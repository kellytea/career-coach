# Career Coach

Extract resume data into a populated ontology using GLiNER, O*NET, and constrained LLM review.

## Project structure

`career_coach/data_extraction/`

- `main.py` runs the complete pipeline and saves the final JSON.
- `parser.py` extracts PDF text and segments resume sections.
- `entity_extraction.py` seeds explicit skills/degrees, enriches them with GLiNER, and resolves O*NET mappings.
- `llm_mapping.py` reviews explicit entities and unresolved taxonomy mappings.
- `ontology.py` generates Pydantic models from `data/ontologies/resume-job.ttl`.
- `ontology_population.py` populates and validates ontology instances using deterministic rules.

## Quick start

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

## Output

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
