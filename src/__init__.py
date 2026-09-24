"""PANDEMICA: computational epidemiology modules (M1-M7).

Modules
-------
compartmental   M1  SIR / SEIR / SEIRD ODE models (+ vaccination, waning immunity)
fitting         M2  least-squares parameter estimation with bootstrap CIs
stochastic      M3  Gillespie exact stochastic SIR
network         M4  epidemics on contact networks, targeted vaccination
interventions   M5  lockdown / vaccination / testing counterfactuals
sensitivity     M6  Latin hypercube sampling + PRCC
metapopulation  M7  multi-region spatial model with a travel matrix
data            data download / cache / synthetic fallback
style           shared matplotlib style
"""

__version__ = "1.0.0"
