"""Immutable local blobs and PostgreSQL document/run metadata."""

import hashlib
import os
from datetime import timedelta
from pathlib import Path

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

from .models import Base, Document, Run, Sync, Version, now, uid


class Store:
    def __init__(self, database_url: str, archive_root: str):
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.session = sessionmaker(self.engine, expire_on_commit=False)
        self.root = Path(archive_root).resolve()

    def initialize(self):
        # Initial schema only. Subsequent schema changes need a reviewed migration.
        Base.metadata.create_all(self.engine)

    def archive(self, payload: bytes) -> dict:
        digest = hashlib.sha256(payload).hexdigest()
        key = f"{digest[:2]}/{digest}"
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_name(target.name + "." + uid() + ".part")
            try:
                with temporary.open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        return {"raw_key": key, "sha256": digest, "bytes": len(payload)}

    def archive_file(self, source: Path) -> dict:
        """Stream datasets into the archive without loading the full file into memory."""
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / (uid() + ".part")
        digest = hashlib.sha256()
        size = 0
        try:
            with source.open("rb") as reader, temporary.open("xb") as writer:
                while chunk := reader.read(1024 * 1024):
                    writer.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            key = f"{digest.hexdigest()[:2]}/{digest.hexdigest()}"
            target = self.root / key
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.replace(temporary, target)
            return {"raw_key": key, "sha256": digest.hexdigest(), "bytes": size}
        finally:
            temporary.unlink(missing_ok=True)

    def read(self, version: Version) -> bytes:
        path = (self.root / version.raw_key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Archive path escaped root")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != version.sha256:
            raise ValueError("Raw archive checksum mismatch")
        return payload

    def ingest(self, payload, *, kind, source, external_id, format, metadata, config):
        blob = self.archive(payload)
        with self.session.begin() as session:
            # Serialize revisions of a source identity, including first insertion.
            session.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
                            {"key": source + ":" + external_id})
            doc = session.scalar(select(Document).where(
                Document.source == source, Document.external_id == external_id))
            if doc is None:
                doc = Document(kind=kind, source=source, external_id=external_id)
                session.add(doc)
                session.flush()
            if doc.kind != kind:
                raise ValueError("Source identity already belongs to another document kind")
            latest = session.scalar(select(Version).where(Version.document_id == doc.id)
                                    .order_by(Version.number.desc()).limit(1))
            if latest is not None and latest.sha256 == blob["sha256"] and latest.format == format:
                version = latest
            else:
                version = Version(document_id=doc.id, number=1 if latest is None else latest.number + 1,
                                  sha256=blob["sha256"], raw_key=blob["raw_key"], format=format,
                                  metadata_json=metadata)
                session.add(version)
                session.flush()
            # Each request is an auditable observation/reprocessing run, even if unchanged.
            run = Run(version_id=version.id, config={**config, "observation": metadata})
            session.add(run)
            session.flush()
            return run.id, version.id, doc.id

    def start_sync(self, config):
        with self.session.begin() as session:
            session.execute(text("SELECT pg_advisory_xact_lock(48190275)"))
            previous = session.scalar(select(Sync).where(Sync.source == "jobicy")
                                      .order_by(Sync.created_at.desc()).limit(1))
            if previous and (previous.status in {"queued", "processing"}
                             or previous.created_at > now() - timedelta(hours=1)
                             or (previous.finished_at and previous.finished_at > now() - timedelta(hours=1))):
                raise ValueError("Jobicy sync already pending or within the hourly cooldown")
            sync = Sync(source="jobicy", config=config)
            session.add(sync)
            session.flush()
            return sync.id
