# CareerCoach browser workspace

This integration branch is based on the latest team `main` at
`57a70181ed29b980862f0a7800571ca7532fce6a`. It brings the already-tested browser UI
into this repository without changing its appearance or browser behavior.
The matching and agent modules, their tests, and demo fixtures come from the
team’s `matching-agent` branch at `df5021f05dda073821a43aa9d67b14f4266c922d`.

The current parser and job ingestion sources in `career-coach/` are unchanged.
`career_coach/__init__.py` makes that moved source tree available through the
existing import name. The added layout test checks that PDF parsing uses those
current files and the repository’s ontology. The Docker image runs from the
source checkout, consistent with the team’s `package = false` setting.

The main [README](README.md) contains quick-start instructions and the integration
notes, followed by the existing extraction documentation.

**Start the server with either method below, then open
[http://localhost:8501](http://localhost:8501).** This guide lives on
[`ui/resume-home-workspace`](https://github.com/kellytea/career-coach/tree/ui/resume-home-workspace);
the default `main` page retains its existing README until the branch is merged.

## Run in Docker

Use the **`ui/resume-home-workspace` branch**, with Docker Desktop running:

```sh
git clone --branch ui/resume-home-workspace https://github.com/kellytea/career-coach.git
cd career-coach
```

Then, from the cloned folder:

```sh
docker compose -f compose.ui.yaml up --build -d
```

Open http://localhost:8501 for the home page and PDF upload dialog. A successful
upload opens `/workspace` with that resume's extracted facts. **Explore the
workspace** opens the upstream parsed resume fixture and eight **synthetic** job
postings on a fresh session. No API key, Neo4j server, or model download is needed
for the default demo or local PDF parsing. The image uses CPU PyTorch rather than the CUDA wheels
in the upstream Linux lock. Starting the image does not rewrite the committed
`pyproject.toml` or `uv.lock`; the integration changes to those files are described
in the main README.
The port is published only on the local machine.

```sh
docker compose -f compose.ui.yaml run --rm --no-deps ui python -m unittest discover -s tests -v
docker compose -f compose.ui.yaml run --rm --no-deps ui python -m unittest test_ui -v
docker compose -f compose.ui.yaml down
```

## Run without Docker

With Python 3.13+ and uv installed, from the cloned folder:

```sh
uv sync
uv run python ui_server.py
```

If the project's dependencies are already installed in your active Python environment:

```sh
python ui_server.py
```

Use the original project's dependencies and run from this source checkout.
The UI itself uses Python's standard library and browser JavaScript; it adds
no framework dependency or CDN script.

## Use real project output

The home page accepts a **PDF under 6 MB**, up to 30 pages, with selectable text.
The default `--pdf-mode local` adapts the existing `ResumeParser.run`, explicit
entity rules, `TaxonomyResolver`, and `OntologyPopulator` into the same ontology
JSON consumed by `load_resumes`. It makes no GLiNER or LLM calls. This conservative
mode uses explicit skills lists and supported degree labels; employment details
may remain unknown. A note asks users to review their extracted facts, and a
warning appears if no explicit skills were found. The fictional fixture
`tests/fixtures/ui_resume.pdf` is provided for a complete upload demo.

For the original full GLiNER + Voyager extraction sequence, configure the same
Voyager environment settings used by the original pipeline, then run:

```sh
python ui_server.py --pdf-mode full
```

Full mode downloads GLiNER weights on first use and sends extracted resume
context to the configured Voyager endpoint. `--llm` separately enables chat and
model-generated outreach. Full PDF mode is optional and was not tested with a live
model or credentials in the local checks. The upload UI rejects scanned PDFs
without selectable text, locked files, invalid files, and oversized files; it
does not add OCR or password entry. The PDF adapter preserves the previous
profile if parsing fails. PDF bytes are held only during processing and temporary
files are removed afterward.

Upload the parser's JSON (`{"resumes": [{"cv_id": "...", "nodes": [...]}]}`)
in the left panel. If it contains multiple resumes, choose the active one.
Load postings JSONL with `job_id`, `title`, `description`, and optional `company`,
`location`, `url`, `source`, and the matcher's structured skill fields.
Uploads are kept in memory for that browser session, expire after two hours of
inactivity, and are cleared on server restart. Input is temporarily written only
while the upstream file loaders read it. The UI limits uploads to 6 MB / 500 jobs.

For graph exports, use the original loader directly at startup:

```sh
python ui_server.py --resumes path/to/resumes.json \
  --jobs path/to/job_graphs --postings-csv path/to/postings.csv
```

Omitting the source CSV leaves graph-only postings without original description
text. The UI explicitly shows that source evidence is unavailable. It does not
claim to execute live Neo4j queries; ontology facts are those returned by the
existing in-memory matcher. The repository retains the separate `knowledge-graph`
branch, but this UI does not merge that branch or replace the team's graph work.

## How it lines up with the proposal

The page consumes the existing `AssistantTools` and `Matcher` results:

| UI action | Existing code used |
|---|---|
| PDF upload | `ResumeParser`, explicit extraction rules, taxonomy resolver, ontology populator, and `load_resumes`; optional full GLiNER/Voyager sequence |
| Recommendations and keyword search | `AssistantTools.search_jobs` |
| Required/preferred skill coverage, near misses and unknown requirements | `Matcher.explain`, `AssistantTools.explain_match` and `skill_gap` |
| Source evidence | Original posting description, parser evidence excerpts, and separately labeled normalized context facts |
| Recruiter/referral draft | `AssistantTools.draft_outreach`, with the original claim verifier and template fallback |
| Optional conversation | `CareerAgent.ask`, retaining tool calls and verification |

Unknown experience/education stays unknown. Required/preferred skills stay
separate. Raw BM25, graph and fusion scores are never presented as a match
probability. Drafts can be edited and copied; there is no send action. Profile,
catalog and ranking changes reset conversation/tool state and clear draft edits.
Separate browser sessions have separate profiles, shortlist and conversation.

The default S1–S3 choices are the proposal's lexical/set baselines. To enable
real S4–S7 models, install the upstream `matching` extra and run:

```sh
python ui_server.py --embeddings sbert --system S7
```

That path downloads the upstream SentenceTransformer models and uses their
FAISS integration. It is not enabled or claimed as tested by the offline demo.
Hashing embeddings are deliberately not offered as a semantic ranking mode.

For optional Voyager chat and generated drafts, configure the original
`VOYAGER_API_KEY` and `VOYAGER_MODEL` on the server and add `--llm`. Keys never go
to browser JavaScript. Enabling this explicitly sends retrieved profile/posting
facts to the configured Voyager endpoint, as the original agent does. The default
run makes no model requests. Do not publish this local development server as a
multi-user hosted service without adding authentication and deployment controls.

The proposal's benchmark evaluation remains a separate workflow in the existing
matching/evaluation module. UI checks demonstrate integration and behavior;
they do not establish nDCG/MRR improvements, human relevance judgments, or the
proposal's full dataset and outreach evaluation results.
