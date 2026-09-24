# Methods: the equations behind every module

This file writes out the mathematics each module implements, with pointers to the code.
Notation: N = population size, S/E/I/R/D/V = number of people in each compartment,
rates are per day.

---

## M1 - Compartmental ODE models (`src/compartmental.py`)

### SIR
$$
\frac{dS}{dt} = -\beta \frac{S I}{N}, \qquad
\frac{dI}{dt} = \beta \frac{S I}{N} - \gamma I, \qquad
\frac{dR}{dt} = \gamma I
$$

- $\beta$: transmission rate (contacts per day times the chance each contact transmits).
- $\gamma$: removal rate; $1/\gamma$ is the mean infectious period.
- **Basic reproduction number** $R_0 = \beta/\gamma$: the expected number of people one
  infectious person infects in a fully susceptible population (rate of infecting,
  $\beta$, times mean time infectious, $1/\gamma$).

### SEIR (adds a latent period of mean $1/\sigma$)
$$
\frac{dS}{dt} = -\beta \frac{S I}{N}, \quad
\frac{dE}{dt} = \beta \frac{S I}{N} - \sigma E, \quad
\frac{dI}{dt} = \sigma E - \gamma I, \quad
\frac{dR}{dt} = \gamma I
$$
$R_0$ is still $\beta/\gamma$ (everyone in E eventually becomes infectious). The latent period
slows the epidemic but does not change its final size.

### SEIRD (a fraction $f$ = IFR of removals die)
$$
\frac{dR}{dt} = (1-f)\,\gamma I, \qquad \frac{dD}{dt} = f\,\gamma I
$$
The force of infection uses the living population $N - D$.

### Optional extras (all models)
With vaccination rate $\nu$, waning rates $\omega$ (R to S) and $\omega_V$ (V to S), and an
isolation rate $\kappa$ (extra removal from I due to testing and isolation):
$$
\frac{dS}{dt} = -\beta \frac{S I}{N} - \nu S + \omega R + \omega_V V, \qquad
\frac{dV}{dt} = \nu S - \omega_V V,
$$
$$
\frac{dI}{dt} = (\text{inflow}) - (\gamma + \kappa) I, \qquad R_0 = \frac{\beta}{\gamma + \kappa}.
$$

### Effective reproduction number
$$
R_{\text{eff}}(t) = \frac{\beta(t)}{\gamma + \kappa(t)} \cdot \frac{S(t)}{N}
$$
Prevalence $I(t)$ grows while $R_{\text{eff}} > 1$ and falls when $R_{\text{eff}} < 1$,
so in SIR the peak happens exactly when $R_{\text{eff}} = 1$, i.e. $S = N/R_0$.

### Analytical results used as tests (`tests/test_compartmental.py`)
1. **Conservation.** Every flow leaves one compartment and enters another, so
   $\frac{d}{dt}(S+E+I+R+D+V) = 0$.
2. **Final-size equation.** Dividing $dS/dt$ by $dR/dt$ in SIR gives $dS/dR = -R_0 S/N$, so
   $S = S_0 e^{-R_0 R/N}$. At the end $I = 0$; with $s = S/N$, $s_0 = S_0/N$:
   $$\ln\frac{s_0}{s_\infty} = R_0\,(1 - s_\infty)$$
   The attack fraction is $z = 1 - s_\infty$. For $s_0 \to 1$ this is $z = 1 - e^{-R_0 z}$
   (R0 = 2 gives z = 0.797).
3. **No epidemic when $R_0 < 1$.** $dI/dt = \gamma I (R_0 S/N - 1) < 0$ from the start.
4. **Peak prevalence (SIR).** $I + S - \frac{N}{R_0}\ln S$ is conserved, which gives
   $$i_{\max} = i_0 + s_0 - \frac{1 + \ln(R_0 s_0)}{R_0}.$$
5. **Early growth.** While $S \approx N$, $I(t) \approx I_0 e^{(\beta - \gamma)t}$.
6. **Herd immunity.** If a fraction above $1 - 1/R_0$ is immune, $R_{\text{eff}} < 1$.
7. **Endemic equilibrium (SIRS).** With waning immunity the system settles at $S^* = N/R_0$.

### Numerical method
`scipy.integrate.solve_ivp` with RK45 (adaptive Runge-Kutta 4(5)), `rtol = atol = 1e-8`,
`max_step = 0.5` days. Runge-Kutta methods preserve linear invariants such as the
population total up to rounding error.

---

## M2 - Parameter estimation (`src/fitting.py`)

### Least squares
Given observed prevalence $y_1,\dots,y_n$ at days $t_1,\dots,t_n$ and the SIR solution
$I(t;\theta)$ with $\theta = (\beta, \gamma, I_0)$:
$$
\hat\theta = \arg\min_\theta \sum_{k=1}^{n} \left(\sqrt{I(t_k;\theta)} - \sqrt{y_k}\right)^2
$$
- The **square root** is a variance-stabilising transform: for Poisson counts
  $\mathrm{Var}(y) = \mathrm{E}(y)$, while $\mathrm{Var}(\sqrt{y}) \approx 1/4$ for every mean.
  Ordinary least squares assumes equal noise at every point, so this makes it appropriate.
- The optimiser works on $\log\theta$, which keeps parameters positive
  (`scipy.optimize.least_squares`, trust-region reflective algorithm, with bounds).

### Residual bootstrap for confidence intervals
1. Fit once; keep the fitted curve $\hat{I}_k$ and residuals $e_k = \sqrt{y_k} - \sqrt{\hat I_k}$.
2. For b = 1..B: draw $e^*_k$ from $\{e_k\}$ with replacement, set
   $y^*_k = (\sqrt{\hat I_k} + e^*_k)^2$, refit to get $\theta^*_b$.
3. The 95% CI of any quantity (such as $R_0 = \beta/\gamma$) is the 2.5th to 97.5th percentile
   of its B bootstrap values.

Resampling residuals, not days, keeps the time structure of the epidemic curve.

### Early growth rate to R0 (COVID-19 demonstration)
Fit $\ln(\text{incidence}_t) = a + r t$ by linear regression over 14 days. For an SEIR model
with exponentially distributed latent and infectious periods (Wallinga & Lipsitch 2007):
$$
R_0 = \left(1 + \frac{r}{\sigma}\right)\left(1 + \frac{r}{\gamma}\right), \qquad
T_{\text{double}} = \frac{\ln 2}{r}.
$$

---

## M3 - Gillespie stochastic simulation (`src/stochastic.py`)

The SIR model as a continuous-time Markov chain on whole numbers of people:

| Event | Change | Rate (propensity) |
|---|---|---|
| infection | $S \to S-1,\; I \to I+1$ | $a_1 = \beta S I / N$ |
| recovery | $I \to I-1,\; R \to R+1$ | $a_2 = \gamma I$ |

**Gillespie direct method**, repeated until $I = 0$:
1. $a_0 = a_1 + a_2$.
2. Time to the next event: $\tau \sim \text{Exponential}(a_0)$, i.e. $\tau = -\ln(u_1)/a_0$.
3. The event is an infection with probability $a_1/a_0$, otherwise a recovery.

This produces exact sample paths of the Markov chain; there is no time-step error.

**Extinction probability.** Early on, each infectious person causes a Geometric number of
new infections before recovering: each event is an infection with probability
$\beta/(\beta+\gamma)$. The probability $q$ that one person's chain of infection dies out
solves $q = \frac{\gamma}{\beta+\gamma} + \frac{\beta}{\beta+\gamma} q^2$, whose smallest
root is $q = 1/R_0$ (for $R_0 > 1$). Chains started by $I_0$ people are independent, so
$$P(\text{extinction}) = (1/R_0)^{I_0}.$$
This neat answer relies on the **exponentially distributed infectious period** of the Markov
SIR model (it makes the offspring distribution geometric). In general, $q$ is the smallest root
of $q = G(q)$, where $G$ is the probability generating function of the offspring distribution.
For example, a fixed-length infectious period gives Poisson offspring and a higher extinction
probability for the same $R_0$.

**Convergence to the ODE.** With $I_0$ a fixed fraction of N, the scaled process $I/N$
converges to the ODE solution as $N \to \infty$ (Kurtz's law of large numbers), and random
fluctuations shrink roughly like $1/\sqrt{N}$. With a *fixed* $I_0$ this fails: runs that die
out early and random delays keep the mean away from the ODE curve. The convergence test's
measured log-log slope (see `results/m3_summary.json`) mixes the finite-N bias with Monte Carlo
error from a finite number of replicates, so it need not equal -1/2.

---

## M4 - Network model (`src/network.py`)

Each node $v$ has a state $x_v \in \{S, I, R\}$; $A$ is the adjacency matrix. Each day:
$$
P(v \text{ infected}) = 1 - (1-p)^{k_v}, \qquad p = 1 - e^{-\tau}, \qquad
k_v = \sum_u A_{vu}\,[x_u = I]
$$
and each infectious node recovers with probability $1 - e^{-\gamma}$.

**Network R0.** The transmissibility $T$ is the probability that an infection passes along a
given edge before the infector recovers. In this daily-step model the infector transmits
with probability $p$ per day and recovers with probability $q_r = 1 - e^{-\gamma}$ per day
(and can still transmit on its last day), so
$$
T = 1 - \sum_{n\ge1} (1-p)^n (1-q_r)^{n-1} q_r = 1 - \frac{q_r (1-p)}{1 - (1-q_r)(1-p)},
$$
which tends to the continuous-time value $\tau/(\tau+\gamma)$ for small daily rates. A newly
infected node was reached through one edge, so it has on average
$\langle k^2\rangle/\langle k\rangle - 1$ further neighbours (the mean excess degree), giving
$$
R_0 \approx T \left(\frac{\langle k^2\rangle}{\langle k\rangle} - 1\right).
$$
For equal mean degree $\langle k\rangle$, a heavy-tailed (scale-free) degree distribution has a
much larger $\langle k^2\rangle$, so a higher $R_0$. This is why hubs matter. The formula assumes
a locally tree-like network. On clustered networks (Watts-Strogatz) many neighbours of a new
case are already infected, so it overstates $R_0$.

**Networks.** Erdos-Renyi $G(n, p)$ with $p = \langle k\rangle/(n-1)$ (Poisson degrees);
Watts-Strogatz ring lattice with each node linked to its $\langle k\rangle$ nearest neighbours,
rewired with probability 0.1 (high clustering, short paths); Barabasi-Albert preferential
attachment with $m = \langle k\rangle/2$ (power-law degree tail, $P(k) \sim k^{-3}$).

**Centralities.** Degree $k_v$. Betweenness
$c_B(v) = \sum_{s \ne v \ne t} \sigma_{st}(v)/\sigma_{st}$: the share of shortest paths
between other pairs that pass through $v$.

---

## M5 - Interventions (`src/interventions.py`)

The M1 SEIRD model with time-dependent rates:
$$
\beta(t) = \beta \prod_{\text{lockdowns}} \bigl(1 - s_\ell\, \mathbf{1}[t_\ell \le t < t_\ell + d_\ell]\bigr), \quad
\nu(t) = \sum \nu_j \mathbf{1}[t \ge t_j], \quad
\kappa(t) = \sum \kappa_j \mathbf{1}[t \ge t_j].
$$
A counterfactual is the same model run with a different set of interventions. The heatmaps
evaluate peak $I$ and final $D$ over a grid of start days $t_\ell$ and strengths $s_\ell$.
With isolation, $R_0$ falls below 1 when $\kappa > \beta - \gamma$.

---

## M6 - Global sensitivity analysis (`src/sensitivity.py`)

**Latin hypercube sampling.** For n samples of k parameters, cut each parameter's range into
n equal intervals, draw one point in each interval, and randomly pair intervals across
parameters. Every one-dimensional slice is covered exactly once.

**PRCC** (partial rank correlation coefficient) of parameter $x_j$ with output $y$:
1. Replace every parameter column and $y$ by their ranks.
2. Linearly regress $\text{rank}(x_j)$ on the ranks of all the other parameters; residuals $r_x$.
3. Linearly regress $\text{rank}(y)$ on the same; residuals $r_y$.
4. $\text{PRCC}_j = \text{corr}(r_x, r_y)$.

Significance: $t = \text{PRCC}\sqrt{(n-2-p)/(1-\text{PRCC}^2)}$ with $n - 2 - p$ degrees of
freedom ($p$ = number of other parameters). PRCC captures monotone, possibly non-linear,
effects and removes the influence of the other parameters.

---

## M7 - Metapopulation model (`src/metapopulation.py`)

Regions $i = 1..n$ with populations $P_i$ and positions $\mathbf{x}_i$. Gravity-model travel
flows (people per day), scaled so that a fraction $\phi$ of all people travel each day:
$$
F_{ij} = c\,\frac{P_i P_j}{\lVert \mathbf{x}_i - \mathbf{x}_j\rVert^2}, \qquad
\sum_{i \ne j} F_{ij} = \phi \sum_i P_i, \qquad m_{ij} = F_{ij}/P_i .
$$
For each compartment $X \in \{S, E, I, R\}$:
$$
\frac{dX_i}{dt} = \bigl(\text{local SEIR terms}\bigr) + \sum_j m_{ji} X_j - X_i \sum_j m_{ij},
$$
with local infection $\beta S_i I_i / N_i$. Because $F$ is symmetric, arrivals equal departures
and each $P_i$ stays constant. A travel restriction multiplies $F$ by $(1 - \text{cut})$.
Arrival day in a region is the first day its prevalence $I_i/P_i$ exceeds $10^{-4}$.

---

## Extension - Bayesian estimation with MCMC (`src/bayes.py`)

Observation model: $y_k \sim \text{NegBin}(\mu_k, k)$ with $\mu_k = I(t_k;\beta,\gamma,I_0)$ and
$\text{Var}(y_k) = \mu_k + \mu_k^2/k$ (Poisson as $k \to \infty$). Log-likelihood:
$$
\ell = \sum_k \left[\ln\Gamma(y_k+k) - \ln\Gamma(k) - \ln y_k! + k\ln\frac{k}{k+\mu_k}
+ y_k \ln\frac{\mu_k}{k+\mu_k}\right].
$$
Priors are flat on $\log\beta, \log\gamma, \log I_0, \log k$ within bounds. The posterior
$p(\theta \mid y) \propto p(y \mid \theta)\,p(\theta)$ is sampled with emcee's
affine-invariant ensemble sampler. Credible intervals are posterior percentiles. The
posterior predictive band adds negative-binomial noise to curves drawn from the posterior.

## Extension - Age-structured SEIR (`src/age_structured.py`)

Groups $i = 1..3$ with sizes $N_i$ and contact matrix $C_{ij}$ (daily contacts of a person
in $i$ with people in $j$; reciprocity requires $N_i C_{ij} = N_j C_{ji}$). Force of infection
$\lambda_i = q \sum_j C_{ij} I_j / N_j$, so $dS_i/dt = -\lambda_i S_i$, and so on.

Next-generation matrix (expected infections in $i$ caused by one infectious person in $j$,
at the start of the epidemic):
$$
K_{ij} = \frac{q\, C_{ij} N_i}{N_j\,\gamma}, \qquad R_0 = \rho(K)\ \ (\text{largest eigenvalue}).
$$
$q$ is set so that $\rho(K)$ equals the target R0. With proportionate mixing
($C_{ij} \propto N_j$) the model collapses to one homogeneous SEIR, which is a test.
Deaths in group $i$ = infections in $i$ x $\text{IFR}_i$.
