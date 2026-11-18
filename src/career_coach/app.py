"""Application-wide Flask entry point, CLI and background task integration."""

import json
import io
import os
import logging
from functools import lru_cache
from pathlib import Path
from uuid import UUID

import click
from celery import Celery
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file
from flask.cli import FlaskGroup
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select, text
from werkzeug.exceptions import HTTPException

from .data_extraction.models import Document, Run, Sync, ValidationRecord, Version, now, uid
from .data_extraction.pipeline import NeedsOCR, extract, gliner_model
from .data_extraction.sources import dataset_records, jobicy_pages
from .data_extraction.storage import Store
from .data_extraction.validation import Finding, ValidationReport, validate_extraction


load_dotenv()
ROOT = Path(__file__).resolve().parents[2]
celery = Celery("career_coach", broker=os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//"))
celery.conf.update(task_ignore_result=True, task_serializer="json", accept_content=["json"],
                   worker_prefetch_multiplier=1, task_acks_late=True,
                   broker_connection_timeout=3, task_publish_retry=False,
                   broker_connection_retry_on_startup=True,
                   task_soft_time_limit=1500, task_time_limit=1800)


@lru_cache(maxsize=1)
def store():
    return Store(os.getenv("DATABASE_URL", "postgresql+psycopg://career:career@localhost:5432/career"),
                 os.getenv("ARCHIVE_ROOT", str(ROOT / "instance" / "archive")))


def run_config():
    return {"taxonomy_dir": str(Path(os.getenv("TAXONOMY_DIR", str(ROOT / "data/taxonomy/onet/31.0"))).resolve()),
            "ontology_path": str(ROOT / "data/ontologies/resume-job.ttl"),
            "embedding_path": os.getenv("EMBEDDING_MODEL_PATH") or None}


def dispatch(run_id):
    try:
        process_run.delay(run_id)
        return True
    except Exception:
        # The queued PostgreSQL row is durable. dispatch-pending recovers broker outages.
        return False


def validate_run(run_id):
    """Validate saved output without changing it or invoking extraction again."""
    database = store()
    with database.session() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise LookupError("Run not found")
        if run.status != "completed" or run.result is None:
            raise ValueError("Only completed extractions can be validated")
        version = session.get(Version, run.version_id)
        document = session.get(Document, version.document_id)
        payload, kind, version_id, config = run.result, document.kind, version.id, run.config
    try:
        report = validate_extraction(payload, kind=kind, version_id=version_id,
            ontology_path=config.get("ontology_path", str(ROOT / "data/ontologies/resume-job.ttl")),
            taxonomy_dir=config.get("taxonomy_dir", str(ROOT / "data/taxonomy/onet/31.0")))
    except Exception:
        logging.getLogger(__name__).exception("Validation execution failed for run %s", run_id)
        report = ValidationReport(status="error", findings=[Finding(severity="error",
            code="validator_error", path="$", message="Validation could not complete; inspect worker/server logs")])
    with database.session.begin() as session:
        record = ValidationRecord(run_id=run_id, report_json=report.model_dump(mode="json", by_alias=True))
        session.add(record)
        session.flush()
        return {"validation_id": record.id, **record.report_json}


@celery.task(name="career_coach.process_run")
def process_run(run_id):
    database = store()
    # Session advisory lock is released on connection loss; a redelivered task
    # can safely restart a processing run after a worker crash.
    with database.engine.connect() as lock:
        acquired = lock.scalar(text("SELECT pg_try_advisory_lock(hashtextextended(:id, 0))"), {"id": run_id})
        lock.commit()
        if not acquired:
            return
        try:
            with database.session.begin() as session:
                run = session.get(Run, run_id)
                if run is None or run.status in {"completed", "failed", "needs_ocr"}:
                    return
                run.status = "processing"
                run.error = None
                version = session.get(Version, run.version_id)
                doc = session.get(Document, version.document_id)
                version_id, format, kind, config = version.id, version.format, doc.kind, run.config
                payload = database.read(version)
            result = extract(payload, format, kind, version_id, config).model_dump(mode="json")
            result["provenance"]["observation"] = config.get("observation", {})
            with database.session.begin() as session:
                run = session.get(Run, run_id)
                run.result = result
                run.status, run.finished_at = "completed", now()
        except Exception as error:
            with database.session.begin() as session:
                run = session.get(Run, run_id)
                if run:
                    run.status = "needs_ocr" if isinstance(error, NeedsOCR) else "failed"
                    # Avoid leaking resume content or provider response bodies into API errors.
                    run.error, run.finished_at = type(error).__name__, now()
            raise
        finally:
            lock.execute(text("SELECT pg_advisory_unlock(hashtextextended(:id, 0))"), {"id": run_id})
            lock.commit()
    # Persist extraction first. A validator or report-storage failure must not
    # discard the extracted evidence or turn extraction completion into failure.
    try:
        validate_run(run_id)
    except Exception:
        logging.getLogger(__name__).exception("Could not persist validation for run %s", run_id)


@celery.task(name="career_coach.sync_jobicy")
def sync_jobicy(sync_id):
    database = store()
    with database.engine.connect() as lock:
        if not lock.scalar(text("SELECT pg_try_advisory_lock(hashtextextended(:id, 0))"), {"id": sync_id}):
            lock.commit()
            return
        lock.commit()
        try:
            with database.session.begin() as session:
                sync = session.get(Sync, sync_id)
                if sync is None or sync.status in {"completed", "failed"}:
                    return
                # A crashed traversal is partial; start a fresh pass only after the cooldown.
                if sync.status == "processing":
                    sync.status, sync.error, sync.finished_at = "failed", "InterruptedTraversal", now()
                    return
                sync.status = "processing"
                config = sync.config

            def archive_response(payload, status):
                blob = database.archive(payload)
                with database.session.begin() as session:
                    sync = session.get(Sync, sync_id)
                    sync.pages = [*sync.pages, {**blob, "http_status": status, "retrieved_at": now().isoformat()}]

            seen = set()
            for jobs in jobicy_pages(config["filters"], archive_response):
                for job in jobs:
                    if not isinstance(job, dict) or not job.get("id") or not job.get("jobTitle"):
                        raise ValueError("Invalid Jobicy job record")
                    external_id = str(job["id"])
                    if external_id in seen:
                        continue
                    seen.add(external_id)
                    metadata = {"source_url": job.get("url"), "source_name": "Jobicy",
                                "retrieved_at": now().isoformat(), "published_at": job.get("pubDate"),
                                "source_status": "discovered_not_rechecked", "sync_id": sync_id}
                    run_id, _, _ = database.ingest(
                        json.dumps(job, ensure_ascii=False, sort_keys=True).encode(), kind="job", source="jobicy",
                        external_id=external_id, format="jobicy_json", metadata=metadata, config=config["pipeline"])
                    with database.session.begin() as session:
                        sync = session.get(Sync, sync_id)
                        sync.run_ids = [*sync.run_ids, run_id]
                    dispatch(run_id)
            with database.session.begin() as session:
                sync = session.get(Sync, sync_id)
                sync.status, sync.finished_at = "completed", now()
        except Exception as error:
            with database.session.begin() as session:
                sync = session.get(Sync, sync_id)
                if sync:
                    sync.status, sync.error, sync.finished_at = "failed", type(error).__name__, now()
            raise
        finally:
            lock.execute(text("SELECT pg_advisory_unlock(hashtextextended(:id, 0))"), {"id": sync_id})
            lock.commit()


class JobicyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int = Field(default=100, ge=1, le=200)
    max_pages: int = Field(default=20, ge=1, le=100)
    geo: str | None = Field(default=None, max_length=100)
    industry: str | None = Field(default=None, max_length=100)
    tag: str | None = Field(default=None, max_length=100)


def valid_id(value):
    return str(UUID(value))


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

    @app.errorhandler(ValueError)
    def bad_request(error):
        if isinstance(error, ValidationError):
            return jsonify(error="Invalid request fields"), 400
        return jsonify(error=str(error)), 400

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    @app.get("/health")
    def health():
        try:
            with store().engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return jsonify(status="ok", postgres="reachable", scope="stages_1_to_5_and_7")
        except Exception:
            return jsonify(status="unavailable", postgres="unreachable"), 503

    @app.post("/documents")
    def upload():
        if "mode" in request.form:
            raise ValueError("Extraction mode is fixed; omit mode")
        file = request.files.get("file")
        if file is None or not file.filename:
            return jsonify(error="A multipart file is required"), 400
        kind = request.form.get("kind", "resume")
        if kind not in {"resume", "job"}:
            raise ValueError("kind must be resume or job")
        format = Path(file.filename).suffix.lower().lstrip(".")
        if format not in {"txt", "pdf", "docx", "html"}:
            raise ValueError("Supported formats: txt, pdf, docx, html")
        payload = file.read()
        if not payload:
            raise ValueError("Empty file")
        external_id = request.form.get("external_id") or uid()
        if len(external_id) > 512:
            raise ValueError("external_id exceeds 512 characters")
        config = run_config()
        run_id, version_id, document_id = store().ingest(
            payload, kind=kind, source="upload", external_id=external_id, format=format,
            metadata={"filename": Path(file.filename).name, "retrieved_at": now().isoformat()}, config=config)
        queued = dispatch(run_id)
        return jsonify(document_id=document_id, version_id=version_id, run_id=run_id,
                       dispatched=queued, status_url=f"/runs/{run_id}"), 202

    @app.get("/documents/<document_id>")
    def document(document_id):
        with store().session() as session:
            doc = session.get(Document, valid_id(document_id))
            if doc is None:
                return jsonify(error="Document not found"), 404
            versions = session.scalars(select(Version).where(Version.document_id == doc.id).order_by(Version.number)).all()
            return jsonify(id=doc.id, kind=doc.kind, source=doc.source, external_id=doc.external_id,
                           versions=[{"id": v.id, "number": v.number, "sha256": v.sha256,
                                      "format": v.format, "metadata": v.metadata_json} for v in versions])

    @app.get("/runs/<run_id>")
    def get_run(run_id):
        with store().session() as session:
            run = session.get(Run, valid_id(run_id))
            if run is None:
                return jsonify(error="Run not found"), 404
            validation = session.scalar(select(ValidationRecord).where(ValidationRecord.run_id == run.id)
                .order_by(ValidationRecord.created_at.desc()).limit(1))
            return jsonify(id=run.id, version_id=run.version_id, status=run.status, error=run.error,
                           created_at=run.created_at.isoformat(),
                           finished_at=run.finished_at.isoformat() if run.finished_at else None,
                           result=run.result, validation=validation.report_json if validation else None,
                           validation_id=validation.id if validation else None)

    @app.post("/runs/<run_id>/validate")
    def validate_saved_run(run_id):
        body = request.get_json() if request.is_json else {}
        if not isinstance(body, dict) or body:
            raise ValueError("Validation accepts no configuration fields")
        try:
            return jsonify(validate_run(valid_id(run_id))), 201
        except LookupError:
            return jsonify(error="Run not found"), 404
        except ValueError as error:
            return jsonify(error=str(error)), 409

    @app.get("/runs/<run_id>/validations")
    def validation_history(run_id):
        with store().session() as session:
            run = session.get(Run, valid_id(run_id))
            if run is None:
                return jsonify(error="Run not found"), 404
            records = session.scalars(select(ValidationRecord).where(ValidationRecord.run_id == run.id)
                .order_by(ValidationRecord.created_at.desc()).limit(20)).all()
            return jsonify(validations=[{"validation_id": r.id, **r.report_json} for r in records])

    @app.get("/runs")
    def list_runs():
        limit = int(request.args.get("limit", "20"))
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        query = select(Run).order_by(Run.created_at.desc()).limit(limit)
        if request.args.get("document_id"):
            query = query.join(Version).where(Version.document_id == valid_id(request.args["document_id"]))
        with store().session() as session:
            runs = session.scalars(query).all()
            return jsonify(runs=[{"id": r.id, "version_id": r.version_id, "status": r.status,
                                  "error": r.error, "created_at": r.created_at.isoformat(),
                                  "status_url": f"/runs/{r.id}"} for r in runs])

    @app.get("/documents/<document_id>/versions/<version_id>/raw")
    def raw_document(document_id, version_id):
        with store().session() as session:
            version = session.get(Version, valid_id(version_id))
            if version is None or version.document_id != valid_id(document_id):
                return jsonify(error="Document version not found"), 404
            payload = store().read(version)
            suffix = "json" if version.format == "jobicy_json" else version.format
            return send_file(io.BytesIO(payload), as_attachment=True,
                             download_name=f"document-{version.id}.{suffix}",
                             mimetype="application/octet-stream")

    @app.post("/runs/<run_id>/reprocess")
    def reprocess(run_id):
        body = request.get_json(silent=False) if request.is_json else {}
        if not isinstance(body, dict) or body:
            raise ValueError("Reprocessing accepts no configuration fields")
        with store().session.begin() as session:
            previous = session.get(Run, valid_id(run_id))
            if previous is None:
                return jsonify(error="Run not found"), 404
            config = run_config()
            config["observation"] = previous.config.get("observation", {})
            run = Run(version_id=previous.version_id, config=config)
            session.add(run)
            session.flush()
            new_id = run.id
        return jsonify(run_id=new_id, dispatched=dispatch(new_id), status_url=f"/runs/{new_id}"), 202

    @app.post("/ingestions/jobicy")
    def ingest_jobicy():
        body = JobicyRequest.model_validate(request.get_json())
        config = {"filters": body.model_dump(), "pipeline": run_config()}
        try:
            sync_id = store().start_sync(config)
        except ValueError as error:
            return jsonify(error=str(error)), 429
        try:
            sync_jobicy.delay(sync_id)
            dispatched = True
        except Exception:
            dispatched = False
        return jsonify(sync_id=sync_id, dispatched=dispatched, status_url=f"/ingestions/{sync_id}"), 202

    @app.get("/ingestions/<sync_id>")
    def get_sync(sync_id):
        with store().session() as session:
            sync = session.get(Sync, valid_id(sync_id))
            if sync is None:
                return jsonify(error="Ingestion not found"), 404
            return jsonify(id=sync.id, status=sync.status, error=sync.error, pages=sync.pages,
                           run_ids=sync.run_ids, created_at=sync.created_at.isoformat(),
                           finished_at=sync.finished_at.isoformat() if sync.finished_at else None)

    @app.cli.command("init-db")
    def init_db():
        """Create the initial review schema; never called on import or server startup."""
        store().initialize()
        click.echo("Initial schema created (existing tables preserved).")

    @app.cli.command("dispatch-pending")
    def dispatch_pending():
        """Republish queued/processing IDs; advisory locks suppress concurrent execution."""
        with store().session() as session:
            ids = session.scalars(select(Run.id).where(Run.status.in_(["queued", "processing"]))).all()
            sync_ids = session.scalars(select(Sync.id).where(Sync.status.in_(["queued", "processing"]))).all()
        dispatched = sum(dispatch(run_id) for run_id in ids)
        for sync_id in sync_ids:
            sync_jobicy.delay(sync_id)
        click.echo(f"Dispatched {dispatched}/{len(ids)} runs and {len(sync_ids)} ingestions.")

    @app.cli.command("ingest-dataset")
    @click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
    @click.option("--kind", type=click.Choice(["resume", "job"]), required=True)
    @click.option("--source", required=True, help="Dataset identity; use the same value for later revisions.")
    @click.option("--text-column", required=True)
    @click.option("--id-column", default=None, help="Defaults to logical row number (one-based).")
    @click.option("--format", "format", type=click.Choice(["txt", "html"]), default="txt", show_default=True)
    @click.option("--limit", type=click.IntRange(min=1), default=None)
    @click.option("--sheet", default=None, help="XLSX worksheet name; defaults to the active sheet.")
    def ingest_dataset(path, kind, source, text_column, id_column, format, limit, sheet):
        """Archive a CSV/XLSX and enqueue its documents; never downloads datasets."""
        if len(source) > 48 or not source or not all(c.isalnum() or c in "_-" for c in source):
            raise click.ClickException("source must be 1-48 letters, digits, underscores or hyphens")
        container_format = path.suffix.lower().lstrip(".")
        if container_format not in {"csv", "xlsx"}:
            raise click.ClickException("Supported dataset containers: csv, xlsx (unzip archives first)")
        database = store()
        blob = database.archive_file(path)
        total, dispatched = 0, 0
        try:
            # Parse the archived snapshot, so row provenance cannot drift if the input is edited.
            for number, external_id, content in dataset_records(database.root / blob["raw_key"],
                    container_format=container_format, text_column=text_column, id_column=id_column, limit=limit, sheet=sheet):
                payload = content.encode("utf-8")
                if len(payload) > 12 * 1024 * 1024:
                    raise ValueError(f"Record {number} exceeds 12 MiB")
                run_id, _, _ = database.ingest(payload, kind=kind, source="dataset:" + source,
                    external_id=external_id, format=format,
                    metadata={"dataset": blob, "filename": path.name, "record_number": number,
                              "text_column": text_column, "container_format": container_format,
                              "sheet": sheet, "retrieved_at": now().isoformat()}, config=run_config())
                dispatched += dispatch(run_id)
                total += 1
                click.echo(json.dumps({"record": number, "run_id": run_id}))
        except (ValueError, OSError) as error:
            raise click.ClickException(f"Import stopped after {total} records; partial imports retained. {error}") from error
        click.echo(f"Archived {total} records; dispatched {dispatched}; recover pending IDs with dispatch-pending.")

    @app.cli.command("warm-model")
    def warm_model():
        """Cache the fixed GLiNER model before the first extraction (optional)."""
        gliner_model()
        click.echo("GLiNER model cached and ready.")

    return app


def main():
    FlaskGroup(create_app=create_app)()
