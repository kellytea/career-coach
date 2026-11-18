# CareerCoach UI

> **Use the UI branch:** these instructions and the UI are saved on
> [`ui/resume-home-workspace`](https://github.com/kellytea/career-coach/tree/ui/resume-home-workspace).
> The default `main` README will stay the same until this branch is merged.

**To open the UI locally, choose either Docker or Python below. Both start a
server, then you open [http://localhost:8501](http://localhost:8501) in your browser.**

This branch adds the tested browser UI to the team repository. The
**`ui/resume-home-workspace` branch** contains the home page, PDF upload, and
matching workspace shown in the demo. Anyone can view or download
this repository without an invitation. The GitHub link shares the source code;
the app runs on your own computer after you start it.

The home page accepts a resume PDF and opens the matching workspace after parsing.
You can also open `/workspace` directly to explore the included demo.
The workspace uses the project's existing matching and agent modules to show
ranked jobs, documented and missing skills, source evidence, and editable outreach
drafts. The current parser files remain in `career-coach/`. Matching and agent code
is reused from the team’s existing `matching-agent` branch.

## Get the UI version

Clone this branch, then run the commands below from the cloned folder:

```sh
git clone --branch ui/resume-home-workspace https://github.com/kellytea/career-coach.git
cd career-coach
```

Alternatively, [download the UI branch as a ZIP](https://github.com/kellytea/career-coach/archive/refs/heads/ui/resume-home-workspace.zip),
unzip it, and open a terminal in the extracted folder.

If you already cloned this team repository, open a terminal in that folder and
switch to the saved UI branch instead:

```sh
git fetch origin
git switch ui/resume-home-workspace
```

## Option A: Docker → localhost:8501

Install and start Docker Desktop. From the project folder containing
`compose.ui.yaml`, run:

```sh
docker compose -f compose.ui.yaml up --build -d
```

Open **[http://localhost:8501](http://localhost:8501)** in your browser. The first
build installs the project dependencies. This separate UI configuration uses CPU
PyTorch and publishes the app only on your local computer.

The home page opens at `/`; the matching screen is
[http://localhost:8501/workspace](http://localhost:8501/workspace).
Check startup or inspect an error with:

```sh
docker compose -f compose.ui.yaml ps
docker compose -f compose.ui.yaml logs ui
```

To stop the UI:

```sh
docker compose -f compose.ui.yaml down
```

## Option B: Python → localhost:8501 (Docker not needed)

Docker is optional. With **Python 3.13+ and uv** installed, run:

```sh
uv sync
uv run python ui_server.py
```

Then open **[http://localhost:8501](http://localhost:8501)**. If the project's
dependencies are already installed in your active Python environment, you can
instead run `python ui_server.py`. Keep that terminal open while using the UI;
press Ctrl+C to stop it.

### If localhost does not open

- Wait for the Docker container to be running, or for the Python terminal to
  print `CareerCoach is running at http://127.0.0.1:8501`.
- Keep Docker/the Python server running while using the page. Opening the GitHub
  link or an HTML file by itself does not start the Python matching server.
- If port 8501 is already occupied, stop the other UI instance. For Python you
  can instead run `uv run python ui_server.py --port 8503`, then open
  [http://localhost:8503](http://localhost:8503).

## What to try

1. On the home page, choose **Upload your resume**, then select or drop a PDF.
   Use selectable text, a file under 6 MB, and at most 30 pages. After parsing,
   the matching workspace opens with your extracted profile. For a fictional
   test resume, use `tests/fixtures/ui_resume.pdf`.
   Or choose **Explore the workspace** to use the included parsed resume and
   **eight synthetic job postings**.
2. Search for a skill or location, such as `Python` or `remote`, and select a job.
3. Review **Why this match** for required/preferred skills and other requirements.
4. Open **Source evidence** to compare posting text with the normalized graph facts.
5. Open **Draft outreach**, create a recruiter/referral draft, then edit and copy it.
6. Use **Upload a new resume PDF** to return to the upload dialog. The left panel
   also accepts the project's **parsed resume JSON** and postings **JSONL**.
   Reset returns to the starting data.

The default run needs **no API key, Neo4j service, or downloaded model weights**.
It uses S2/BM25 ranking, with S1/TF-IDF and S3/skill overlap also available. Chat is
disabled and outreach uses the existing verified template. Nothing is sent
automatically. Unknown resume facts remain labeled “not documented.”

PDF uploads default to the original parser's **local explicit rules**, taxonomy
mapping, and validated ontology population. This mode recognizes explicit skills
lists and supported degree labels; it does not run GLiNER or infer missing
employment facts. Review the extracted profile. Scanned, password-protected,
and unreadable PDFs show a clear error while preserving your previous profile.
The uploaded PDF is removed after parsing; extracted facts remain in the local
session. See [UI-README.md](UI-README.md) for the optional full extraction mode.

For graph-export inputs, real S4–S7 dense retrieval, optional Voyager chat, and
the mapping between UI actions and the proposal, see [UI-README.md](UI-README.md).
The current UI reads ontology facts through the existing in-memory matcher;
live Neo4j query integration is separate work. The full proposal benchmark and
human evaluation are not established by this demo.

## Run the checks

```sh
docker compose -f compose.ui.yaml run --rm --no-deps ui python -m unittest discover -s tests -v
docker compose -f compose.ui.yaml run --rm --no-deps ui python -m unittest test_ui -v
```

The tested UI implementation passed **143 extraction/matching/agent tests and 17 UI integration
tests in Docker**. The UI checks include the current parser import layout and a real PDF flowing into matching and
source evidence, invalid/encrypted PDFs, and session isolation. Those checks use
offline fixtures. Full GLiNER/Voyager PDF extraction, real dense models, live
Voyager responses, and live Neo4j integration were not tested. Without Docker,
use `uv run python -m unittest discover -s tests -v` and
`uv run python -m unittest test_ui -v`.

## Integration with the current repository

This branch starts from `main` at `57a70181ed29b980862f0a7800571ca7532fce6a`.
The team’s existing matching/agent modules and their tests are included from
`matching-agent` at `df5021f05dda073821a43aa9d67b14f4266c922d`, because they are
required by the UI and have not yet been merged into `main`.

The parser and job ingestion files stay unchanged in `career-coach/`. A small
`career_coach/__init__.py` import bridge exposes them under the existing Python
import name used by their CLI, tests, matching module, and UI. This preserves
the recent folder move without copying the parser or reverting the layout.
The project’s `package = false` setting and existing dependencies are retained.
NumPy is declared for matching, and the optional `matching` extra enables the
existing dense-retrieval code. Existing locked package versions are preserved.

The browser UI, styles, logo, and upload behavior match the tested UI fork.
There is no live Neo4j integration or benchmark result added by this branch.

## Original extraction pipeline documentation

Extract resume data into a populated ontology using GLiNER, O*NET, and constrained LLM review.

### Project structure

`career-coach/data_extraction/`

- `main.py` runs the complete pipeline and saves the final JSON.
- `parser.py` extracts PDF text and segments resume sections.
- `entity_extraction.py` seeds explicit skills/degrees, enriches them with GLiNER, and resolves O*NET mappings.
- `llm_mapping.py` reviews explicit entities and unresolved taxonomy mappings.
- `ontology.py` generates Pydantic models from `data/ontologies/resume-job.ttl`.
- `ontology_population.py` populates and validates ontology instances using deterministic rules.

### Quick start

From the project root:

```powershell
uv sync
```

Set these values in `.env`:

```dotenv
VOYAGER_API_KEY=your-api-key
VOYAGER_MODEL=deepseek-v4-1-flash
```

Place resume PDFs in a `resumes/` folder, then run `main()` through its module:

```powershell
uv run python -m career_coach.data_extraction.main resumes
```

### Output

The pipeline writes only `resumes.json` next to the input folder after all resumes succeed.

- `resumes`: the list of populated resume objects.
- `cv_id`: the source PDF filename for each resume.
- `nodes`: validated ontology instances with `id`, `type`, and properties such as `hasSkill`.
- `metadata`: labels, taxonomy identifiers/version, resolution status, original titles, and source evidence.

Relationships reference node IDs; unknown or unsupported relationships remain unasserted.
All degrees listed in Education are linked, without checking completion status.
`yearsExperience` uses only each employment header's month/year range; ongoing roles use the current month.
Explicit unmapped skills retain local IDs and an `unresolved` status.
Intermediate results stay in memory. The pipeline does not write to Neo4j.
