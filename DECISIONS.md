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

## Phase 1 - Core engine (M1)
- **One general ODE right-hand side** covers SIR, SEIR and SEIRD (unused compartments
  stay at zero). Less code, and every model shares the same, tested conservation logic.
- **Force of infection uses the living population** (`beta S I / (N - D)`). Because of
  this, SEIRD's attack rate is slightly higher than SEIR's; the final-size-equation test
  is therefore applied to SIR and SEIR only (the equation assumes a closed population).
- **Solver: RK45 with rtol=atol=1e-8, `max_step=0.5` day.** Runge-Kutta methods keep
  linear invariants (here the population total) exactly apart from rounding, which is why
  conservation holds to ~1e-15. `max_step` stops the solver skipping over step-function
  interventions in M5.
- **Vaccine is perfect and all-or-nothing** (S -> V). Simplest defensible choice.
- **Testing/isolation is an extra removal rate `kappa` from I.** Isolated people are
  counted as removed, so R0 becomes beta / (gamma + kappa).
- **Cumulative infections `C` is tracked as an extra ODE state** so incidence comes from
  differences of C, not from numerical differentiation.
- **The stochastic-vs-deterministic convergence test (rule 10) is added in Phase 4**,
  because it needs the M3 Gillespie code.

## Phase 2 - Data layer
- **Primary fitting dataset: the 1978 English boarding-school influenza outbreak**
  (763 boys, daily number confined to bed). It is real, public, a closed population and
  a single wave, which is exactly what SIR assumes. National COVID-19 series break
  those assumptions (interventions, variants, changing testing), so fitting a
  constant-parameter SIR to them would be hard to defend.
- **JHU CSSE COVID-19 is still used** for the data summary table and exploratory plots,
  and for an early-growth-rate R0 estimate in M2, which only needs the first weeks.
- **"In bed" is treated as the infectious prevalence I(t).** Standard simplification in
  textbook treatments of this dataset; it is listed under Limitations.
- **The R data file is parsed with the pure-Python `rdata` package** (no R needed) and
  converted to CSV once.
- **Synthetic outbreak is always generated**, not only on download failure: SIR with
  beta=1.6, gamma=0.45, I0=2, N=1000, Poisson observation noise, seed 42. It exists to
  test parameter recovery. If the real download fails, `get_fit_dataset()` returns
  it instead and every label says SYNTHETIC.
- **Tests never need the network**: downloads are monkeypatched to fail in the fallback
  tests, and the real-data test is skipped if the cache is missing.
