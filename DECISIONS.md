# Design decisions

Choices made while building PANDEMICA without being able to ask questions.
Each entry: the decision and the reason. Newest at the bottom.

## Phase 0 - Setup
- **Python 3.14 in `.venv`.** The folder already had a PyCharm-created 3.14 venv; all
  pinned packages install from wheels on it, so it was kept.
- **Package layout: `src/` is the package** (`from src.compartmental import ...`), run
  from the project root. Tests use `pythonpath = .` in `pytest.ini`. This avoids an
  editable install step and keeps "clone, install requirements, run" to two commands.
- **Removed the PyCharm `main.py` sample** and git-ignored `.idea/` (IDE settings are
  not part of the project).
- **Analysis scripts live in `analysis/`** (one per module) and write numbers to
  `results/` and figures to `figures/`. `run_all.py` calls them in order. Library code
  in `src/` never writes files, which keeps it easy to test.
- **Okabe-Ito palette** for categorical colors and viridis/magma for continuous
  colormaps: both are colorblind-safe.
- **Data, results and figures are committed** so the README renders on GitHub and the
  project runs offline straight after cloning. The raw files are small (< 2 MB).
