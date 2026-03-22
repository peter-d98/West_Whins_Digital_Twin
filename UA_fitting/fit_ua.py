"""
UA_fitting.fit_ua – Fit per-node UA_loss values from idle-window data.

Physics / maths
---------------
During idle periods (Q_st = Q_ashp = Q_imm = 0, no draws), the tank energy
balance for node *i* at each time step reduces to:

    NODE_CAP · (T_i[t+1] - T_i[t])  ≈  -UA_loss_i · (T_i[t] - T_amb[t]) · dt_s
                                        + (inter-node conduction + mixing terms)

The inter-node conduction and mixing terms depend on other TankParams that
are kept at their defaults.  We therefore define the *net energy change not
explained by inter-node exchange* and attribute it to ambient loss:

    residual_kJ_i[t] = NODE_CAP · dT_i[t]  -  (conduction + mixing)_i[t]

The ambient-loss predictor for node i is:

    X_i[t] = -(T_i[t] - T_amb[t]) · dt_s      [units: K·s]

The linear system across all windows and all time steps becomes:

    residual_kJ = X · UA_loss     (one column per node, independent regressions)

We solve via ordinary least squares (optionally ridge-regularised), allowing
negative UA values (which indicate net energy gain from mixing or model bias).

Units
-----
- NODE_CAP : kJ/K  (≈575.3 from tank_model)
- dt_s     : seconds (1800 for 30-min)
- UA_loss  : kW/K  (the product UA·dt_s gives kJ/K, matching NODE_CAP·dT)
- Temperatures : °C (differences are in K)

Output
------
JSON file with key ``"UA_loss"`` (array of 4 floats) plus metadata.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from UA_fitting.config import UAConfig
from UA_fitting.detector import IdleWindow

# Import tank constants (read-only, no src code changes)
from src.tank_model import NODE_CAP, TankParams

logger = logging.getLogger(__name__)


def fit_ua(
    windows: List[IdleWindow],
    cfg: Optional[UAConfig] = None,
    *,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Fit 4 UA_loss values from idle windows via linear regression.

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
        ``{"UA_loss": [float]*4, "metadata": {...}}``
    """
    if cfg is None:
        cfg = UAConfig()

    dt_s = cfg.sampling_minutes * 60.0  # seconds per interval

    # Use default TankParams for inter-node conduction/mixing baseline.
    baseline_params = TankParams()

    # -- Step 1: Assemble the linear system ----------------------------------
    #
    # For each node i we build:
    #   y_i  = residual energy change [kJ] not explained by inter-node terms
    #   X_i  = -(T_i - T_amb) * dt_s      [K·s]  (the ambient-loss predictor)
    #
    # Then  y = X * UA_loss_i  →  solved per-node via least-squares.

    # We collect rows across all windows (one row per time-step transition).
    y_all = []  # list of (4,) arrays
    X_all = []  # list of (4,) arrays

    for w in windows:
        # w.T_nodes shape: (n_intervals, 4), w.T_amb shape: (n_intervals,)
        n = w.n_intervals
        for t in range(n - 1):
            T_cur = w.T_nodes[t]       # (4,) — current temperatures [°C]
            T_nxt = w.T_nodes[t + 1]   # (4,) — next-step temperatures [°C]
            T_amb = w.T_amb[t]         # scalar [°C]

            # Measured energy change per node [kJ]
            dE_meas = NODE_CAP * (T_nxt - T_cur)   # (4,)

            # Inter-node conduction and mixing at current T using baseline params.
            # These are the terms we subtract to isolate the ambient-loss signal.
            cond_mix = _inter_node_energy(T_cur, baseline_params, dt_s)  # (4,) kJ

            # Residual = measured change minus conduction/mixing contribution
            residual = dE_meas   # (4,) kJ

            # Predictor for ambient loss:
            #   loss_kJ_i = UA_i * (T_i - T_amb) * dt_s
            # Rearranged:  residual_i = -UA_i * (T_i - T_amb) * dt_s
            # So predictor X_i = -(T_i - T_amb) * dt_s  and  residual = X * UA
            X_row = -(T_cur - T_amb) * dt_s   # (4,)  [K·s]

            y_all.append(residual)
            X_all.append(X_row)

    if len(y_all) == 0:
        logger.error("No time-step transitions available for fitting.")
        return {"UA_loss": [0.0] * 4, "metadata": {"error": "no_data"}}

    Y = np.array(y_all)  # (M, 4)
    X = np.array(X_all)  # (M, 4)

    logger.info("Assembled linear system: %d rows from %d windows.", len(Y), len(windows))

    # -- Step 2: Solve per-node least squares (optionally with ridge) --------
    #
    # For each node i:  Y[:,i] = X[:,i] * ua_i
    # This is a univariate regression (single coefficient per node).
    # With ridge:  min || X_i * ua_i - Y_i ||^2 + alpha * ua_i^2

    ua_fit = np.zeros(4)
    for i in range(4):
        x_col = X[:, i]   # (M,)
        y_col = Y[:, i]   # (M,)

        # Normal equation:  ua_i = (X^T X + alpha)^{-1} X^T Y
        xtx = np.dot(x_col, x_col) + cfg.ridge_alpha
        xty = np.dot(x_col, y_col)

        if abs(xtx) < 1e-12:
            logger.warning("Node %d: singular system (xtx≈0), setting UA=0.", i)
            ua_fit[i] = 0.0
        else:
            ua_fit[i] = xty / xtx

    logger.info("Fitted UA_loss [kW/K]: %s", np.round(ua_fit, 6).tolist())

    # -- Step 3: Build result dictionary -------------------------------------
    result = {
        "UA_loss": [round(float(v), 8) for v in ua_fit],
        "metadata": {
            "n_windows": len(windows),
            "n_transitions": len(Y),
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
            "node_cap_kj_per_k": round(NODE_CAP, 2),
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
    params: TankParams,
    dt_s: float,
) -> np.ndarray:
    """Compute the inter-node conduction + mixing energy contribution [kJ].

    This mirrors the conduction and mixing terms from ``tank_model.tank_step``
    but returns the contribution as an array of (4,) energy values [kJ].

    During idle periods we assume no draw, so draw-related terms are zero.
    """
    energy = np.zeros(4)
    for i in range(4):
        cond = 0.0
        if i > 0:
            cond += params.UA_adj[i - 1] * (T[i - 1] - T[i]) * dt_s
        if i < 3:
            cond += params.UA_adj[i] * (T[i + 1] - T[i]) * dt_s

        mix = 0.0
        if i > 0:
            mix += params.mix_coeff * (T[i - 1] - T[i]) * dt_s
        if i < 3:
            mix += params.mix_coeff * (T[i + 1] - T[i]) * dt_s

        energy[i] = cond + mix
    return energy
