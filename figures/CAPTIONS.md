# Figure captions

All figures are produced by `python run_all.py` (PNG at 300 dpi; the animation is a GIF and an MP4).
Colours: Okabe-Ito palette (colorblind-safe) for categories; viridis/magma for continuous scales.

## Data
| Figure | Caption |
|---|---|
| `data_outbreaks.png` | Left: the real 1978 English boarding-school influenza outbreak (boys confined to bed each day, N = 763). Right: the SYNTHETIC outbreak generated from a known SIR model with Poisson noise, used to check that the fitting code recovers known parameters. |
| `data_jhu_daily_cases.png` | Daily confirmed COVID-19 cases (7-day average) for six countries from the JHU CSSE archive, showing the several waves that a single constant-parameter SIR model cannot describe. |

## M1 - Compartmental models
| Figure | Caption |
|---|---|
| `m1_epidemic_curves.png` | SIR, SEIR and SEIRD solutions with R0 = 3: a latent period (SEIR) delays and flattens the peak without changing the final size, and SEIRD adds a death compartment. |
| `m1_phase_portrait.png` | SIR trajectories in the susceptible-infectious plane for four R0 values; each trajectory peaks exactly where S/N = 1/R0 (dotted lines). |
| `m1_reff.png` | The effective reproduction number R_eff(t) = R0 S(t)/N falls as people become immune, and prevalence peaks at the moment R_eff crosses 1. |
| `m1_vaccination_waning.png` | Optional compartments: vaccination lowers the peak, while waning immunity produces repeated waves that settle into an endemic level. |

## M2 - Parameter estimation
| Figure | Caption |
|---|---|
| `m2_fit_observed.png` | SIR fitted by least squares to the boarding-school outbreak, with a 95% bootstrap confidence band for the fitted curve (left) and the bootstrap distribution of R0 (right). |
| `m2_fit_synthetic_recovery.png` | The same fitting pipeline applied to SYNTHETIC data: the estimated R0 is compared with the true value used to generate the data. |
| `m2_mcmc_posterior.png` | Bayesian SIR fit by MCMC (emcee) with a negative-binomial likelihood: 95% credible band for the mean curve and posterior predictive band for new data (left), R0 posterior compared with the least-squares bootstrap (middle), and the joint posterior of beta and gamma (right). |
| `m2_covid_growth.png` | Log-linear fits to raw daily COVID-19 case counts over the first 14 days of growth in six countries, converted to R0 with assumed latent and infectious periods (a method demonstration, not an estimate of COVID-19's R0). |

## Stretch goal - Age structure
| Figure | Caption |
|---|---|
| `age_vaccination_priority.png` | Age-structured SEIR with an illustrative contact matrix (left): total infections (middle) and deaths (right) for four ways of allocating a limited number of vaccine doses. |

## M3 - Stochastic model
| Figure | Caption |
|---|---|
| `m3_stochastic_spaghetti.png` | Gillespie replicates of the same SIR process (left): some outbreaks die out early (grey), the rest scatter around the deterministic curve; the final-size histogram (right) is bimodal, with minor and major outbreaks. |
| `m3_extinction_probability.png` | Probability that an outbreak dies out early, from simulation (dots) and from branching-process theory (1/R0)^I0 (dashed lines). |
| `m3_convergence.png` | The gap between the stochastic ensemble mean and the ODE solution shrinks as population size grows (log-log axes). |

## M4 - Network model
| Figure | Caption |
|---|---|
| `m4_network_graphs.png` | 300-node Erdos-Renyi, Watts-Strogatz and Barabasi-Albert networks coloured by infection state at the epidemic peak; node size shows degree. |
| `m4_degree_distribution.png` | Degree distributions of the three 2000-node networks: all have mean degree 8, but only the scale-free network has a heavy tail of hubs. |
| `m4_network_epidemic_curves.png` | Mean and 90% band of 50 epidemics on each network: hubs make the scale-free epidemic fastest, clustering makes the small-world epidemic slow and small. |
| `m4_superspreaders.png` | On the scale-free network, the number of people a node infects rises steeply with its degree and betweenness centrality. |
| `m4_vaccination_strategies.png` | Attack rate against vaccination coverage for random, degree-targeted and betweenness-targeted vaccination on each network (mean +/- SD of 20 runs). |

## M5 - Interventions
| Figure | Caption |
|---|---|
| `m5_counterfactuals.png` | Infectious people, cumulative deaths and R_eff over time for five scenarios: no intervention, a 60-day lockdown, vaccination, testing and isolation, and all three combined. |
| `m5_intervention_heatmaps.png` | Peak infectious and cumulative deaths for every combination of intervention start day and strength (log colour scale), for lockdown, vaccination and isolation. |

## M6 - Sensitivity analysis
| Figure | Caption |
|---|---|
| `m6_tornado.png` | Tornado plot of partial rank correlation coefficients: which SEIRD parameters most strongly raise (red) or lower (blue) cumulative deaths and peak infections. |
| `m6_prcc_matrix.png` | PRCC of each parameter with each of four outcomes, showing for example that the fatality ratio affects deaths but not the size or timing of the epidemic. |

## M7 - Spatial model
| Figure | Caption |
|---|---|
| `m7_spatial_snapshots.png` | The epidemic spreading from the Capital across eight fictional regions linked by gravity-model travel (circle area = population, colour = prevalence on a log scale). |
| `m7_regional_curves.png` | Prevalence in each region with normal travel (left) and with travel cut by 90% (right): the outer regions peak later but just as high. |
| `m7_arrival_times.png` | Day the epidemic arrives in each region, ordered by distance from the Capital, with and without a 90% travel restriction. |
| `m7_spatial_spread.gif` / `.mp4` | Animation of the wave moving across the map (left) with the regional prevalence curves and a moving time marker (right). |
