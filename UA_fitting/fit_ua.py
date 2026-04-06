"""
UA_fitting.fit_ua – Jointly fit UA_loss and UA_adj from idle-window data.

Physics / maths
---------------
During idle periods (Q_st = Q_ashp = Q_imm = 0, no draws), the full energy
balance for node *i* is:

    NODE_CAP_i · dT_i[t] = -UA_loss_i · (T_i[t] - T_amb[t]) · dt_s
                           + UA_adj[i-1] · (T[i-1][t] - T_i[t]) · dt_s  (i > 0)
                           + UA_adj[i]   · (T[i+1][t] - T_i[t]) · dt_s  (i < 3)

This gives 4 equations per time-step transition and 7 unknowns:
  - UA_loss[0..3]  — per-node ambient loss conductance [kW/K]
  - UA_adj[0..2]   — adjacent-node conductance [kW/K]
                     (UA_adj[k] couples node k and node k+1)

All 4 equations at each time-step reference *different* subsets of the same
7 parameters, so fitting them independently would corrupt each estimate with
the un-modelled conduction flux from neighbouring nodes.  Joint fitting solves
all 4M equations (M = total time-step transitions) simultaneously:

    Y  =  X · θ        Y ∈ ℝ^{4M},   X ∈ ℝ^{4M×7},   θ ∈ ℝ^7

Parameter ordering in θ:
    [UA_loss[0], UA_loss[1], UA_loss[2], UA_loss[3],
     UA_adj[0],  UA_adj[1],  UA_adj[2]]

We enforce physical non-negativity (lb = 0) via bounded least squares
(scipy.optimize.lsq_linear) with optional Tikhonov ridge regularisation.

Units
-----
- NODE_CAP_UA : array [kJ/K], shape (4,), bottom→top
- dt_s        : seconds per interval
- UA_loss     : kW/K  (UA · dt_s gives kJ/K, matching NODE_CAP_UA · dT)
- UA_adj      : kW/K
- Temperatures : °C (differences are in K)

Output
------
JSON with keys ``"UA_loss"`` (4 floats), ``"UA_adj"`` (3 floats), metadata.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import lsq_linear

from UA_fitting.config import UAConfig
from UA_fitting.detector import IdleWindow

# ---------------------------------------------------------------------------
# Per-node thermal capacities matching the physical tank geometry
# ---------------------------------------------------------------------------
# Bottom node : 170 L
# Mid, Mid-Hi, Top : 380 L split equally → 126.67 L each
# NODE_CAP_i = volume_i [L] × RHO [kg/L] × CP [kJ/(kg·K)]
_RHO = 1.0          # kg/L  (water density)
_CP  = 4.186        # kJ/(kg·K)
NODE_CAP_UA = np.array([
    170.0          * _RHO * _CP,   # bottom  → 711.62 kJ/K
    (380.0 / 3.0)  * _RHO * _CP,   # mid     → 530.21 kJ/K
    (380.0 / 3.0)  * _RHO * _CP,   # mid-hi  → 530.21 kJ/K
    (380.0 / 3.0)  * _RHO * _CP,   # top     → 530.21 kJ/K
])  # shape (4,), kJ/K, bottom→top

logger = logging.getLogger(__name__)


def fit_ua(
    windows: List[IdleWindow],
    cfg: Optional[UAConfig] = None,
    *,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Jointly fit UA_loss (4 values) and UA_adj (3 values) from idle windows.

    Parameters
    ----------
    windows : list[IdleWindow]
        Validated idle windows (output of ``detect_idle_windows``).
    cfg : UAConfig, optional
        Configuration (defaults used if not provided).
    output_dir : Path, optional
        Directory for ``ua_fit.json``.  Created if needed.  If *None*, the
        result is returned but not saved to disk.

    Returns
    -------
    result : dict
        ``{"UA_loss": [float]*4, "UA_adj": [float]*3, "metadata": {...}}``
    """
    if cfg is None:
        cfg = UAConfig()

    dt_s = cfg.sampling_minutes * 60.0  # seconds per interval

    # -- Step 1: Assemble the joint linear system ----------------------------
    #
    # Parameter vector θ (7 unknowns), indices:
    #   0..3  UA_loss[0..3]   ambient loss per node
    #   4..6  UA_adj[0..2]    inter-node conductance (adj[k] couples node k & k+1)
    #
    # For node i at time t:
    #   dE_i = -UA_loss[i] * (T_i - T_amb) * dt_s
    #          + UA_adj[i-1] * (T[i-1] - T_i) * dt_s   (if i > 0)
    #          + UA_adj[i]   * (T[i+1] - T_i) * dt_s   (if i < 3)
    #
    # This gives one (7,) feature row per (node, timestep) pair.
    # UA_adj[k] between nodes k and k+1 is at parameter index 4+k.
    # For node i:
    #   "from below" predictor uses UA_adj[i-1] at index 4+(i-1) = 3+i
    #   "from above" predictor uses UA_adj[i]   at index 4+i

    y_list: list = []   # one scalar per (node, timestep)
    x_list: list = []   # one (7,) row per (node, timestep)

    for w in windows:
        n = w.n_intervals
        for t in range(n - 1):
            T = w.T_nodes[t]        # (4,) current temperatures [°C]
            T_nxt = w.T_nodes[t + 1]  # (4,) next temperatures [°C]
            T_amb = w.T_amb[t]      # scalar [°C]

            dE = NODE_CAP_UA * (T_nxt - T)  # (4,) measured energy change [kJ]

            for i in range(4):
                row = np.zeros(7)
                row[i] = -(T[i] - T_amb) * dt_s        # UA_loss[i]
                if i > 0:
                    row[3 + i] = (T[i - 1] - T[i]) * dt_s  # UA_adj[i-1]
                if i < 3:
                    row[4 + i] = (T[i + 1] - T[i]) * dt_s  # UA_adj[i]
                y_list.append(dE[i])
                x_list.append(row)

    if len(y_list) == 0:
        logger.error("No time-step transitions available for fitting.")
        return {
            "UA_loss": [0.0] * 4,
            "UA_adj": [0.0] * 3,
            "metadata": {"error": "no_data"},
        }

    Y = np.array(y_list)   # (4M,)
    X = np.array(x_list)   # (4M, 7)

    n_equations = len(Y)
    logger.info(
        "Joint linear system: %d equations, 7 unknowns  (%d windows).",
        n_equations, len(windows),
    )

    # -- Step 2: Solve with non-negative bounded least squares ---------------
    #
    # Enforce θ ≥ 0 (all UA values are physically non-negative).
    # Ridge regularisation:  augment with √α · I  so that the augmented
    # normal equations become (X^T X + α I) θ = X^T Y.

    if cfg.ridge_alpha > 0.0:
        ridge_mat = np.sqrt(cfg.ridge_alpha) * np.eye(7)
        X_aug = np.vstack([X, ridge_mat])
        Y_aug = np.concatenate([Y, np.zeros(7)])
    else:
        X_aug, Y_aug = X, Y

    fit_result = lsq_linear(X_aug, Y_aug, bounds=(0.0, np.inf), method="bvls")
    theta = fit_result.x   # (7,)

    ua_loss = theta[0:4]
    ua_adj  = theta[4:7]

    logger.info("Fitted UA_loss [kW/K]: %s", np.round(ua_loss, 6).tolist())
    logger.info("Fitted UA_adj  [kW/K]: %s", np.round(ua_adj,  6).tolist())

    # -- Step 3: Build result dictionary -------------------------------------
    result = {
        "UA_loss": [round(float(v), 8) for v in ua_loss],
        "UA_adj":  [round(float(v), 8) for v in ua_adj],
        "metadata": {
            "n_windows": len(windows),
            "n_equations": n_equations,
            "train_date_start": str(windows[0].start) if windows else None,
            "train_date_end": str(windows[-1].end) if windows else None,
            "thresholds": {
                "ashp_off_kwh": cfg.ashp_off_kwh,
                "st_off_kwh": cfg.st_off_kwh,
                "imm_off_kwh": cfg.imm_off_kwh,
                "min_idle_intervals": cfg.min_idle_intervals,
                "draw_delta_c": cfg.draw_delta_c,
                "ridge_alpha": cfg.ridge_alpha,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "units": "kW/K (per node, bottom→top)",
            "node_cap_kj_per_k": [round(float(v), 2) for v in NODE_CAP_UA],
            "dt_s": dt_s,
        },
    }

    # -- Step 4: Save to disk if output_dir provided -------------------------
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "ua_fit.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info("Saved UA fit to %s", out_path)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _inter_node_energy(
    T: np.ndarray,
    ua_adj: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Compute the inter-node conduction energy contribution [kJ].

    Parameters
    ----------
    T : (4,) current node temperatures [°C].
    ua_adj : (3,) inter-node conductances [kW/K].
    dt_s : time-step [s].

    Returns
    -------
    energy : (4,) array of conduction energy per node [kJ].
    """
    energy = np.zeros(4)
    for i in range(4):
        if i > 0:
            energy[i] += ua_adj[i - 1] * (T[i - 1] - T[i]) * dt_s
        if i < 3:
            energy[i] += ua_adj[i] * (T[i + 1] - T[i]) * dt_s
    return energy
