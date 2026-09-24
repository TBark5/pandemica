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

## Phase 3 - Fitting (M2)
- **Fit SIR to prevalence with free beta, gamma and I0; N = 763 is known.** Both
  beta and gamma are identifiable because the whole rise and fall of I(t) is observed.
- **Parameters are optimised on a log scale** with `scipy.optimize.least_squares`
  (keeps them positive, and puts beta and gamma on comparable scales).
- **Least squares on square-root counts.** A first version used raw counts. A coverage
  check (20 synthetic datasets, 100 bootstrap refits each, run once in a scratch script
  during development, not saved in results/) showed its 95% CI for R0 contained the true
  value only 65% of the time, because count noise is much larger near the peak and the
  plain residual bootstrap ignores that. The square root makes Poisson noise roughly
  equal in size (variance-stabilising transform); the same check gave 90%. The final
  pipeline re-runs both versions (20 datasets x 50 refits each, different seeds from the
  scratch check) and saves them to `results/m2_summary.json`: 95% coverage for sqrt vs 80%
  for raw counts. Only the saved numbers are quoted in the README and other docs.
- **Residual bootstrap (500 refits)**, not case resampling: resampling days would break
  the time structure of an epidemic curve.
- **The shaded band is a confidence band for the fitted curve**, not a prediction
  interval for new observations, so some data points are expected outside it.
- **I0 is weakly identified.** Its CI can miss the true value on synthetic data (the
  day-0 observation is a single small noisy count). This is reported, not hidden.
- **COVID-19 R0 from early growth rates** (secondary analysis): 14-day window from the
  first day the 7-day-average daily cases reach 20, log-linear regression, and
  R0 = (1 + r/sigma)(1 + r/gamma) with an assumed 5.2-day latent period and 5-day
  infectious period. These assumed values drive the result strongly, and early 2020
  case growth also reflects testing ramp-up. The numbers are shown as a demonstration
  of the method, not as estimates of COVID-19's R0.

## Phase 4 - Stochastic (M3) and network (M4)
- **Gillespie direct method in pure Python/NumPy**, with random numbers pre-drawn in
  blocks of 4096. Fast enough: 500 replicates at N = 1000 take about a second.
- **Major vs minor outbreak cut-off: final size above 10% of N.** Final sizes are
  bimodal (see `m3_stochastic_spaghetti.png`), so any cut-off in the empty middle gives
  the same answer.
- **Convergence test uses I0 = 1% of N**, not a fixed I0. With a fixed number of seeds,
  the random delay of the early phase does not shrink as N grows, so the ensemble mean
  is smeared in time and never matches the ODE curve. Scaling I0 with N is the setting
  of Kurtz's law of large numbers, under which the stochastic model converges to the ODE.
- **Replicate counts** (runtime cap): 500 for the main ensemble, 400 per cell of the
  extinction table, 40 per N in the convergence study (N up to 30,000).
- **Network model is discrete-time (1-day steps)** with per-edge infection probability
  1 - exp(-tau) and recovery probability 1 - exp(-gamma). Vectorised with a sparse
  adjacency matrix. A continuous-time network Gillespie would be exact but much slower.
- **All three networks have 2000 nodes and mean degree ~8**, so differences come from
  structure alone. Watts-Strogatz rewiring probability 0.1 (clustered small world),
  Barabasi-Albert m = 4.
- **tau = 0.06, gamma = 0.2** gives an approximate network R0 of ~1.8 on Erdos-Renyi, a
  mid-sized epidemic where network effects are visible.
- **Betweenness uses networkx's sampled estimator (k = 500 sources)** for speed.
- **Vaccination = removing the node** (it starts in R). "Attack rate" is the share of
  *all* nodes ever infected, so vaccinated nodes count as not infected.
- **Superspreaders**: each new infection is credited to a random infectious neighbour.
  This is exact when there is only one and a fair split otherwise.
- **Snapshot figure uses 300-node versions** of each network, because 2000 nodes cannot
  be drawn readably.

## Phase 5 - Interventions (M5) and sensitivity (M6)
- **Baseline for M5: SEIRD, R0 = 2.5, 5-day latent, 7-day infectious, IFR 1%, N = 1M.**
  Generic "COVID-like" illustrative values, not calibrated to any real epidemic.
- **Lockdown is temporary (60 days)**; vaccination and isolation stay on once started.
  This is what makes the lockdown heatmap interesting: a lockdown that ends before
  enough people are immune only delays the epidemic.
- **Simulation horizon 730 days** so waves pushed later by a lockdown are still counted
  in cumulative deaths.
- **Heatmap grid: 16 start days (0-150, step 10) x 13 strengths** per intervention,
  208 ODE runs each, about 30 s per heatmap on CPU.
- **Heatmaps use a log colour scale** because outcomes span several orders of magnitude.
- **M6 samples 7 SEIRD parameters** (beta, infectious and latent periods, IFR,
  vaccination rate, isolation rate, initial infected) with uniform ranges listed in
  `src/sensitivity.py`. 800 LHS samples (~90 s) is a common size for PRCC with 7 inputs.
- **PRCC implemented directly** (rank, regress out the other ranked parameters,
  correlate residuals) instead of adding a sensitivity-analysis library. It is about
  15 lines and is tested on functions with known monotone effects.
- **Periods rather than rates are sampled** (e.g. infectious period 4-10 days), because
  those are the quantities people actually quote and reason about.

## Phase 6 - Spatial (M7)
- **8 fictional regions on a made-up map** (a 2M "Capital" plus cities and towns,
  5.3M people in total). Fictional so nobody mistakes it for a forecast for a real place.
- **Gravity-model travel** F_ij ~ P_i P_j / d_ij^2, symmetric, scaled so 0.2% of the
  total population travels per day. Symmetric flows keep every region's population
  constant.
- **Migration-style coupling** (people move between regions and take their infection
  state with them) rather than commuting-style mixing. Simpler to write and explain.
- **Per-region attack rate = (E + I + R) / P at the end**, not infections counted by
  location. The location-based count credited the first-hit region with its visitors'
  infections and gave misleading differences between regions (caught by a test).
- **Arrival day = first day prevalence exceeds 1 in 10,000.** A deterministic ODE puts a
  tiny fraction of an infected person everywhere immediately, so a threshold is needed.
- **Animation**: 91 frames (days 0-180, every 2 days). GIF at 65 dpi (~1.4 MB, so it
  can go in the README) and MP4 at 120 dpi through the ffmpeg binary bundled with
  `imageio-ffmpeg` (no system install needed). The static snapshot figure is 300 dpi.

## Phase 7 - Visuals
- **Every PNG at 300 dpi** via `src/style.savefig`; one caption per figure in
  `figures/CAPTIONS.md`. Log colour scales where values span orders of magnitude.
- **Fixes made in the review pass**: smoother phase portrait (0.1-day output), text moved
  off curves, overlapping titles and tick labels, arrival-time chart changed from a
  labelled scatter (labels collided) to grouped bars, R_eff panel added to the
  counterfactual figure.
- **`analysis/report.py` writes the README results tables and headline bullets** from
  `results/`, so README numbers always match the saved runs (rule 4). `run_all.py` runs it
  last.

## Phase 8 - Dashboard
- **Streamlit, one tab per module**, tab code split into `dashboard/` so no file exceeds
  ~300 lines. Heavy computations (bootstrap, Gillespie, network runs, LHS) use
  `st.cache_data` and smaller defaults than the saved runs; saved figures are shown next to
  the live controls.
- **`?tab=m1` ... `?tab=m7` deep links** show one module. Added so headless Chrome can
  screenshot each tab for the README; also handy for sharing.
- **Dashboard smoke tests use Streamlit's `AppTest`** (renders all tabs headless, changes a
  slider). A real `streamlit run` launch was also checked (health endpoint + screenshots).
- **`.streamlit/config.toml`** turns off usage statistics and sets a light theme.

## Phase 9 - Quality
- **Fresh-environment check**: a brand-new venv installed from `requirements.txt` alone
  passed the full test suite with no skips. A separate fresh venv ran `run_all.py` end to
  end, and every file in `results/` and `figures/*.png` came out identical to the committed
  versions (only line endings differed). Seeds make the whole pipeline deterministic.
- **Windows path-length gotcha**: installing Streamlit into a very deeply nested folder
  fails with WinError 206 (path > 260 characters). Keep the project path short, or enable
  long paths in Windows.
- **`tests/test_docs.py`** fails if the README results tables drift from `results/`, if
  any doc still contains placeholder markers, or if a figure has no caption.
- **CI workflow** (`.github/workflows/tests.yml`) and **Dockerfile** were added but could
  not be run here: there is no git remote, and Docker is not installed on this machine.

## Stretch - Bayesian estimation (MCMC)
- **emcee ensemble sampler, 24 walkers x 4000 steps, 1000 burn-in, thin 5.** The kept
  chain is more than 50 autocorrelation times long (emcee's own rule of thumb).
- **Negative-binomial likelihood** with a free dispersion k, because the flu counts are
  more spread out than Poisson. Flat priors on log(beta), log(gamma), log(I0), log(k)
  within wide bounds.
- **Fast solver for the likelihood**: `scipy.integrate.odeint` on the 2-state SIR (R is
  implied). It is ~8x faster than the general M1 right-hand side; a test checks the two
  agree. Needed to keep each MCMC run under the ~2-minute cap.
- **Result differs from least squares** (lower median R0, wider interval). The two methods
  make different assumptions about the noise; both are reported rather than picking one.
- `sampler.random_state` is seeded, so MCMC results are reproducible.

## Stretch - Age structure
- **3 age groups with an illustrative, reciprocal contact matrix** (children mostly meet
  children, adults have the most contacts) and steeply age-dependent IFRs. Real survey
  matrices (POLYMOD) would need an extra data download and processing; the illustrative
  matrix is labelled as such in the figure, code and README.
- **R0 from the next-generation matrix**, per-contact transmission probability scaled to
  hit R0 = 2.5.
- **Vaccination priority experiment**: 0-300k doses (1M people) allocated children-first,
  adults-first, 65+-first or pro rata, with spill-over to the next group when one is full.
