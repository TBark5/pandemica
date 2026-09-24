"""Documentation consistency: README results must match the saved results (rule 4)."""

from pathlib import Path

import pytest

from analysis import report
from src.style import RESULTS_DIR

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"


@pytest.mark.skipif(not (RESULTS_DIR / "m7_summary.json").exists(),
                    reason="results not generated yet (run python run_all.py)")
def test_readme_tables_match_saved_results():
    """The generated tables and headline bullets appear verbatim in README.md."""
    text = README.read_text(encoding="utf-8")
    assert report.build().strip() in text, "README results out of date: run `python run_all.py report`"
    assert report.headlines().strip() in text


def test_docs_have_no_placeholders():
    """No TODO / TBD / placeholder text left in the documentation."""
    docs = ["README.md", "METHODS.md", "INTERVIEW_PREP.md", "RESUME_BULLETS.md",
            "DECISIONS.md", "figures/CAPTIONS.md"]
    for name in docs:
        path = ROOT / name
        if path.exists():
            content = path.read_text(encoding="utf-8")
            for marker in ("TODO", "TBD", "FIXME", "lorem ipsum"):
                assert marker not in content, f"{marker} found in {name}"


def test_every_figure_has_a_caption():
    captions = (ROOT / "figures" / "CAPTIONS.md").read_text(encoding="utf-8")
    for png in (ROOT / "figures").glob("*.png"):
        assert f"`{png.name}`" in captions, f"no caption for {png.name}"
