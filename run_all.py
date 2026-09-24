"""Reproduce every PANDEMICA result and figure from scratch.

Usage:
    python run_all.py            # everything (~10 minutes on a laptop CPU)
    python run_all.py m3 m5      # only some steps

Each step writes numbers to results/ and figures to figures/. Seeds are fixed,
so re-running gives identical outputs.
"""

from __future__ import annotations

import importlib
import sys
import time

STEPS = {
    "m1": ("analysis.m1_compartmental", "M1 compartmental models"),
    "data": ("analysis.data_overview", "Data download / cache / summary"),
    "m2": ("analysis.m2_fitting", "M2 parameter estimation"),
    "m3": ("analysis.m3_stochastic", "M3 Gillespie stochastic model"),
    "m4": ("analysis.m4_network", "M4 network epidemics"),
    "m5": ("analysis.m5_interventions", "M5 interventions"),
    "m6": ("analysis.m6_sensitivity", "M6 sensitivity analysis"),
    "m7": ("analysis.m7_spatial", "M7 metapopulation + animation"),
    "report": ("analysis.report", "Results tables -> results/SUMMARY.md and README.md"),
}


def main(selected: list[str]) -> int:
    unknown = [s for s in selected if s not in STEPS]
    if unknown:
        print(f"Unknown step(s): {unknown}. Choose from: {list(STEPS)}")
        return 2
    total = time.perf_counter()
    for key in selected or list(STEPS):
        module, label = STEPS[key]
        start = time.perf_counter()
        print(f"[{key}] {label} ...", flush=True)
        importlib.import_module(module).main()
        print(f"[{key}] done in {time.perf_counter() - start:.1f} s", flush=True)
    print(f"All done in {time.perf_counter() - total:.1f} s. See results/ and figures/.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
