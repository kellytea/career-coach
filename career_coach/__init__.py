"""Expose the team's moved source tree under its existing Python import name.

The parser and job ingestion sources stay in ``career-coach/``. Matching and
agent modules use this importable package alongside those unchanged sources.
Run from the repository root, as configured by the project's package=false mode.
"""

from pathlib import Path

_team_sources = Path(__file__).resolve().parents[1] / "career-coach"
if _team_sources.is_dir():
    __path__.append(str(_team_sources))
