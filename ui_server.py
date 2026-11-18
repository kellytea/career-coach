"""Local browser workspace over the existing matching and agent modules.

Run from the source checkout: python ui_server.py (no extra UI dependencies).
"""

import argparse
from dataclasses import asdict
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlsplit

from career_coach.agent.agent import CareerAgent
from career_coach.agent.grounding import ClaimVerifier
from career_coach.agent.llm import LLMError, VoyagerChatModel
from career_coach.agent.tools import AssistantTools
from career_coach.matching.cli import build_matcher
from career_coach.matching.embeddings import MINILM, MPNET
from career_coach.matching.profiles import load_any_jobs, load_jobs, load_resumes
from career_coach.matching.skills import SkillExtractor
from career_coach.matching.taxonomy import Taxonomy
from ui_pdf import MAX_PDF_BYTES, PDFResumeParser

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "tests/fixtures"
MAX_BODY = 8 * 1024 * 1024
SYSTEM_NAMES = {"S1": "TF-IDF", "S2": "BM25", "S3": "Skill overlap", "S4": "MiniLM",
                "S5": "MPNet", "S6": "BM25 + MPNet", "S7": "BM25 + MPNet + graph"}


def text(data, key, limit=500):
    value = data.get(key, "")
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{key} must be text with at most {limit} characters")
    return value


def parse_upload(content, loader, suffix):
    # Temporary files exist only while the upstream path-based loader reads them.
    with tempfile.TemporaryDirectory(prefix="career-coach-ui-") as directory:
        path = Path(directory) / ("upload" + suffix)
        path.write_text(content, encoding="utf-8")
        try:
            result = loader(path)
        except (KeyError, TypeError, AttributeError, ValueError, IndexError) as error:
            raise ValueError("The file does not match the project's parsed data format") from error
    if not result:
        raise ValueError("The uploaded file contains no usable records")
    return result


class Workspace:
    def __init__(self, app):
        self.app = app
        self.lock = threading.RLock()
        self.touched = time.monotonic()
        self.profiles = app.profiles
        self.matcher = app.matcher
        self.profile_demo = app.profile_demo
        self.jobs_demo = app.jobs_demo
        self.catalog_name = app.catalog_name
        self.profile_processing = None
        self.revision = 0
        self.reset_tools(app.profile_index, app.args.system)

    def reset_tools(self, index=None, system=None):
        self.profile_index = self.profile_index if index is None else index
        self.system = self.system if system is None else system
        self.tools = AssistantTools(self.matcher, self.profiles[self.profile_index], self.app.verifier,
                                    self.app.model, system=self.system)
        self.agent = CareerAgent(self.tools, self.app.model, self.app.verifier) if self.app.model else None
        self.revision += 1

    def state(self):
        profile = self.tools.resume
        return {"profile": asdict(profile), "profile_index": self.profile_index,
                "profiles": [{"index": i, "cv_id": p.cv_id} for i, p in enumerate(self.profiles)],
                "profile_demo": self.profile_demo, "jobs_demo": self.jobs_demo,
                "catalog_name": self.catalog_name, "catalog_size": len(self.matcher.jobs),
                "systems": [{"id": s, "name": SYSTEM_NAMES[s]} for s in self.matcher.available_systems()],
                "system": self.system, "chat_enabled": bool(self.agent), "revision": self.revision,
                "graph_mode": "Ontology facts from the matching module (in memory)",
                "pdf_mode": self.app.args.pdf_mode, "profile_processing": self.profile_processing}

    def upload_pdf(self, content, filename):
        with self.app.pdf_lock:
            document, processing = self.app.pdf_parser.parse(content, filename)
        profiles = parse_upload(document, lambda p: load_resumes(p, self.app.taxonomy), ".json")
        if not profiles[0].skills:
            processing["warnings"].append(
                "No explicit skills were identified. Try a clearly labeled Skills section or load parsed resume JSON")
        self.profiles, self.profile_demo = profiles, False
        self.profile_processing = processing
        self.reset_tools(0)
        return self.state()

    def detail(self, job_id):
        match = self.matcher.explain(self.tools.resume, job_id, self.system)
        return {"explanation": self.tools.explain_match(job_id), "gap": self.tools.skill_gap(job_id),
                "matched_required": [s.label for s in match.graph.matched_required],
                "matched_preferred": [s.label for s in match.graph.matched_preferred],
                "description": match.job.description, "location": match.job.location,
                "source": match.job.source, "resume_evidence": self.tools.resume.evidence,
                "revision": self.revision}

    def dispatch(self, path, data):
        if path == "/api/search":
            count = data.get("top_k", 10)
            if type(count) is not int or not 1 <= count <= 20:
                raise ValueError("top_k must be an integer from 1 to 20")
            result = self.tools.search_jobs(text(data, "query"), count)
            for row in result["results"]:
                match = self.matcher.explain(self.tools.resume, row["job_id"], self.system)
                row.update(location=match.job.location, matched_required_count=len(match.graph.matched_required),
                           required_count=len(match.job.required))
            return {**result, "revision": self.revision}
        if path == "/api/draft":
            return self.tools.draft_outreach(text(data, "job_id"), text(data, "recipient") or "recruiter")
        if path == "/api/profile":
            if "content" in data:
                profiles = parse_upload(text(data, "content", MAX_BODY),
                                        lambda p: load_resumes(p, self.app.taxonomy), ".json")
                if len(profiles) > 100:
                    raise ValueError("Upload at most 100 profiles")
                self.profiles, self.profile_demo = profiles, False
                self.profile_processing = None
                self.reset_tools(0)
            else:
                index = data.get("index")
                if type(index) is not int or not 0 <= index < len(self.profiles):
                    raise ValueError("Choose a profile from the uploaded file")
                self.reset_tools(index)
            return self.state()
        if path == "/api/jobs":
            jobs = parse_upload(text(data, "content", MAX_BODY),
                                lambda p: load_jobs(p, self.app.extractor), ".jsonl")
            if len(jobs) > 500 or len({j.job_id for j in jobs}) != len(jobs):
                raise ValueError("Use at most 500 postings with unique job_id values")
            matcher = self.app.build(jobs)
            self.matcher, self.jobs_demo = matcher, False
            self.catalog_name = "Uploaded job postings"
            self.reset_tools()
            return self.state()
        if path == "/api/system":
            system = text(data, "system")
            if system not in self.matcher.available_systems():
                raise ValueError("This ranking system is not available in this run")
            self.reset_tools(system=system)
            return self.state()
        if path == "/api/reset":
            self.profiles, self.matcher = self.app.profiles, self.app.matcher
            self.profile_demo, self.jobs_demo = self.app.profile_demo, self.app.jobs_demo
            self.catalog_name = self.app.catalog_name
            self.profile_processing = None
            self.reset_tools(self.app.profile_index, self.app.args.system)
            return self.state()
        if path == "/api/chat":
            if self.agent is None:
                raise ValueError("Chat is disabled. Start the server with --llm and Voyager settings")
            message = text(data, "message", 4000).strip()
            if not message:
                raise ValueError("Enter a message")
            return asdict(self.agent.ask(message))
        raise KeyError("Unknown API route")


class Application:
    def __init__(self, args):
        self.args = args
        self.taxonomy = Taxonomy()
        self.extractor = SkillExtractor(self.taxonomy)
        self.verifier = ClaimVerifier(self.extractor)
        self.pdf_parser = PDFResumeParser(self.taxonomy, args.pdf_mode)
        self.pdf_lock = threading.Lock()
        self.profiles = load_resumes(args.resumes, self.taxonomy)
        if not self.profiles:
            raise ValueError("No parsed resumes were found")
        self.profile_index = 0
        if args.cv_id:
            self.profile_index = next((i for i, p in enumerate(self.profiles) if p.cv_id == args.cv_id), -1)
            if self.profile_index < 0:
                raise ValueError("The requested cv-id was not found")
        self.matcher = self.build(load_any_jobs(args.jobs, self.extractor, args.postings_csv))
        if args.system not in self.matcher.available_systems():
            raise ValueError(f"Available ranking systems: {self.matcher.available_systems()}")
        self.profile_demo = args.resumes.resolve() == (FIXTURES / "legacy_resume_output.json").resolve()
        self.jobs_demo = args.jobs.resolve() == (FIXTURES / "sample_jobs.jsonl").resolve()
        self.catalog_name = "Synthetic demo postings" if self.jobs_demo else args.jobs.name
        self.model = VoyagerChatModel() if args.llm else None
        self.sessions = {}
        self.lock = threading.Lock()

    def build(self, jobs):
        return build_matcher(self.args, jobs)

    def session(self, header):
        jar = cookies.SimpleCookie()
        try:
            jar.load(header or "")
            token = jar["cc_session"].value if "cc_session" in jar else None
        except cookies.CookieError:
            token = None
        with self.lock:
            now = time.monotonic()
            self.sessions = {k: v for k, v in self.sessions.items() if now - v.touched < 7200}
            fresh = token not in self.sessions
            if fresh:
                if len(self.sessions) >= 128:
                    raise ValueError("Too many open sessions; restart the local server")
                token = secrets.token_urlsafe(32)
                self.sessions[token] = Workspace(self)
            workspace = self.sessions[token]
            workspace.touched = now
            return token, workspace, fresh


class Handler(BaseHTTPRequestHandler):
    server_version = "CareerCoachUI"

    def log_message(self, format, *args):
        # Do not log profile text, prompts, query strings, or model responses.
        pass

    def respond(self, status, body, content_type="application/json; charset=utf-8", cookie=None):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        if cookie:
            self.send_header("Set-Cookie", f"cc_session={cookie}; HttpOnly; SameSite=Strict; Path=/; Max-Age=7200")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlsplit(self.path)
        static = {"/": ("home.html", "text/html"), "/workspace": ("ui.html", "text/html"),
                  "/workspace/": ("ui.html", "text/html"),
                  "/assets/ui.css": ("ui.css", "text/css"), "/assets/ui.js": ("ui.js", "text/javascript"),
                  "/assets/home.css": ("home.css", "text/css"), "/assets/home.js": ("home.js", "text/javascript"),
                  "/assets/brand.svg": ("brand.svg", "image/svg+xml")}
        if url.path in static:
            name, mime = static[url.path]
            return self.respond(200, (ROOT / name).read_bytes(), mime + "; charset=utf-8")
        if url.path == "/health":
            return self.respond(200, {"status": "ok"})
        if url.path not in {"/api/state", "/api/detail"}:
            return self.respond(404, {"error": "Not found"})
        try:
            token, workspace, fresh = self.server.app.session(self.headers.get("Cookie"))
            with workspace.lock:
                value = workspace.state() if url.path == "/api/state" else workspace.detail(
                    parse_qs(url.query).get("job_id", [""])[0])
            self.respond(200, value, cookie=token if fresh else None)
        except (KeyError, ValueError) as error:
            self.respond(400, {"error": str(error).strip("'\"")})

    def do_POST(self):
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            return self.respond(403, {"error": "Use this app's own browser tab"})
        if urlsplit(self.path).path == "/api/resume/pdf":
            return self.upload_pdf()
        if self.headers.get_content_type() != "application/json":
            return self.respond(415, {"error": "Send application/json"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                return self.respond(413, {"error": "Upload limit is 8 MB"})
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            token, workspace, fresh = self.server.app.session(self.headers.get("Cookie"))
            with workspace.lock:
                result = workspace.dispatch(urlsplit(self.path).path, data)
            self.respond(200, result, cookie=token if fresh else None)
        except KeyError as error:
            self.respond(404, {"error": str(error).strip("'\"")})
        except (ValueError, TypeError, UnicodeError) as error:
            self.respond(400, {"error": str(error)})
        except LLMError:
            self.respond(502, {"error": "Voyager could not complete the request. Try again or use the match tools."})
        except Exception:
            self.respond(500, {"error": "The request could not be completed. Your previous workspace is still available."})

    def upload_pdf(self):
        if self.headers.get_content_type() != "application/pdf":
            return self.respond(415, {"error": "Choose a PDF resume"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_PDF_BYTES:
                return self.respond(413, {"error": "Choose a PDF under 6 MB"})
            content = self.rfile.read(length)
            token, workspace, fresh = self.server.app.session(self.headers.get("Cookie"))
            with workspace.lock:
                result = workspace.upload_pdf(content, self.headers.get("X-Resume-Filename", "resume.pdf"))
            self.respond(200, result, cookie=token if fresh else None)
        except (ValueError, TypeError) as error:
            self.respond(400, {"error": str(error)})
        except Exception:
            self.respond(500, {"error": "The PDF could not be processed. Try another copy or upload parsed resume JSON in the workspace"})


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--resumes", type=Path, default=FIXTURES / "legacy_resume_output.json")
    parser.add_argument("--jobs", type=Path, default=FIXTURES / "sample_jobs.jsonl")
    parser.add_argument("--postings-csv", type=Path)
    parser.add_argument("--cv-id")
    parser.add_argument("--system", default="S2")
    parser.add_argument("--embeddings", choices=("none", "sbert"), default="none")
    parser.add_argument("--minilm", default=MINILM)
    parser.add_argument("--mpnet", default=MPNET)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/embeddings"))
    parser.add_argument("--llm", action="store_true", help="Enable Voyager chat and generated drafts")
    parser.add_argument("--pdf-mode", choices=("local", "full"), default="local",
                        help="PDF extraction: local explicit rules, or full GLiNER + Voyager pipeline")
    return parser.parse_args(argv)


def create_server(args):
    app = Application(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.app = app
    return server


if __name__ == "__main__":
    server = create_server(arguments())
    print(f"CareerCoach is running at http://{server.server_address[0]}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
