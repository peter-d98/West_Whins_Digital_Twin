"""
Global_fitting – Fits remaining free tank parameters (UA_adj, optionally
UA_loss[0]) using frozen priors from UA_fitting and ASHP_fitting.

Uses one-step-ahead prediction against measured tank temperatures to avoid
drift accumulation from unmodelled effects (draws, partial ASHP runs, etc.).

Public API
----------
- run_global_fit(cfg)  — fit UA_adj (+ optionally UA_loss[0]) and evaluate
"""

from Global_fitting.fit_global import run_global_fit

__all__ = ["run_global_fit"]
