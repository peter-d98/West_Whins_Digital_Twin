"""
Parameter identification for the Stage-1 digital twin.

Two-step procedure:
  1. Fit ASHP maps on intervals with no immersion and low ST.
  2. Fit tank parameters (with ASHP heat derived from the map).

Joint refinement with regularisation is also available.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import ashp_model, solar_thermal, tank_model
from .tank_model import NODE_CAP, TankParams

logger = logging.getLogger(__name__)

# Minimum power (kWh) used to avoid division-by-zero in COP calculations
_MIN_POWER_KWH = 1e-3


@dataclass
class IdentificationResult:
    """Container for fitted parameters and diagnostics."""
    tank_params: tank_model.TankParams
    ashp_params: ashp_model.ASHPParams
    hx_effectiveness: float
    cost_history: list[float]


def compute_ashp_runs_1min(
    df: pd.DataFrame,
    st_col: str = "st_kwh",
    min_run_minutes: int = 10,
    min_dtop_run_c: float = 0.5,
    min_dtsum_run_c: float = 2.0,
    min_warming_nodes: int = 2,
) -> pd.DataFrame:
    """Find qualifying ASHP-only DHW runs in 1-minute resolution data.

    An interval is considered ASHP-only DHW if:
            - 2-minute rolling ASHP electricity ``ashp_inst_kwh`` sum > 0.10 kWh
                (to reduce meter quantisation artefacts)
            - ``imm_tot_inst_kwh < 0.05`` (immersion heater off)
      - solar-thermal energy is negligible (``st_col < 0.002``)
      - all four tank temperatures are finite at this row and the previous row

    Consecutive qualifying intervals are grouped into runs.  Only runs of
    length ≥ ``min_run_minutes`` are retained.  For each qualifying run the
    function computes:

    * ``Q_kwh`` — total heat delivered to the tank over the run [kWh],
      using the 4-node energy balance:
      ``Q_kJ = NODE_CAP × Σ(T[end,i] − T[start-1,i]) + Σ(UA_loss × ΔT_amb × 60s)``
    * ``P_kwh`` — total ASHP electricity over the run [kWh]
    * ``n_minutes`` — run length [minutes]
    * ``T_out_c`` — outdoor air temperature at run end [°C]
    * ``T_sink_c`` — ASHP sink-proxy temperature at run end [°C]

    Parameters
    ----------
    df : pd.DataFrame
        1-minute cleaned DataFrame with tank temperatures, ASHP electricity,
        immersion electricity, and ambient temperature columns.
    st_col : str
        Name of the solar-thermal energy column (default ``"st_kwh"``).
    min_run_minutes : int
        Minimum consecutive qualifying minutes to form a valid run (default 10).
    min_dtop_run_c : float
        Minimum required top-node temperature rise over the full run
        (``T_top[end] - T_top[start-1]``). Enforces bulk-charge behavior.
    min_dtsum_run_c : float
        Minimum net sum of node temperature rises over the run
        (``sum(T[end] - T[start-1])``). Helps reject setpoint-maintenance and
        mixed non-DHW runs with little net tank charging.
    min_warming_nodes : int
        Minimum number of tank nodes that must show positive net temperature
        rise over the run.

    Returns
    -------
    pd.DataFrame
        One row per qualifying run, indexed by run-end timestamp.
        Columns: ``Q_kwh``, ``P_kwh``, ``n_minutes``, ``T_out_c``, ``T_sink_c``,
        ``dtop_run_c``.
    """
    tank_cols = ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]
    dt_s = 60.0

    ashp_2min_kwh = df["ashp_inst_kwh"].fillna(0).rolling(2, min_periods=1).sum()
    ashp_on  = ashp_2min_kwh > 0.10   # 2-min rolling sum > 0.10 kWh ≈ > 3 kW active charging
    imm_off  = df["imm_tot_inst_kwh"].fillna(0) < 0.05  # clear margin above immersion-on level (~0.10 kWh/min)
    st_low   = (
        df[st_col].fillna(0) < 0.002
        if st_col in df.columns
        else pd.Series(True, index=df.index)
    )

    T = df[tank_cols].values
    finite_now  = np.all(np.isfinite(T), axis=1)
    finite_prev = np.roll(finite_now, 1)
    finite_prev[0] = False

    mask = (
        ashp_on & imm_off & st_low
        & pd.Series(finite_now & finite_prev, index=df.index)
    )

    ua_loss_default = TankParams().UA_loss
    T_amb = df["t_amb_c"].fillna(df["t_amb_c"].median()).values if "t_amb_c" in df.columns else np.zeros(len(df))

    mask_arr = mask.values

    # Group consecutive True intervals into (start, end) index pairs
    run_pairs: list[tuple[int, int]] = []
    in_run = False
    run_start = 0
    for k in range(len(mask_arr)):
        if mask_arr[k]:
            if not in_run:
                run_start = k
                in_run = True
        else:
            if in_run:
                run_end = k - 1
                if run_end - run_start + 1 >= min_run_minutes:
                    run_pairs.append((run_start, run_end))
                in_run = False
    if in_run:
        run_end = len(mask_arr) - 1
        if run_end - run_start + 1 >= min_run_minutes:
            run_pairs.append((run_start, run_end))

    logger.info("Total candidate runs ≥ %d min: %d", min_run_minutes, len(run_pairs))

    
    rejected_dtop = 0
    rejected_dtsum = 0
    rejected_nodes = 0
    records = []
    end_timestamps = []

    for k_start, k_end in run_pairs:
        # k_start == 0 should never happen (finite_prev[0] = False ensures it),
        # but guard explicitly to avoid accessing T[-1] on unexpected data.
        if k_start == 0:
            logger.warning("Run starting at index 0 skipped (no predecessor row).")
            continue
            
        n = k_end - k_start + 1

        # Run-level thermal-quality guards for bulk-charge behavior.
        dT_nodes = T[k_end, :] - T[k_start - 1, :]
        dtop_run_c = float(dT_nodes[3])
        dtsum_run_c = float(np.sum(dT_nodes))
        n_warming_nodes = int(np.sum(dT_nodes > 0.0))

        if dtop_run_c < min_dtop_run_c:
            rejected_dtop += 1
            continue
        if dtsum_run_c < min_dtsum_run_c:
            rejected_dtsum += 1
            continue
        if n_warming_nodes < min_warming_nodes:
            rejected_nodes += 1
            continue

        # Energy balance: T change over full run (pre-run state = T[k_start-1])
        # Storage term: per-node capacity × per-node temperature change
        storage_kJ = 0.0
        for i in range(4):
            storage_kJ += NODE_CAP[i] * dT_nodes[i]

        # Standing losses over run
        loss_kJ = 0.0
        for k in range(k_start, k_end + 1):
            for i in range(4):
                loss_kJ += ua_loss_default[i] * (T[k, i] - T_amb[k]) * dt_s

        Q_kJ  = storage_kJ + loss_kJ
        Q_kwh = max(Q_kJ / 3600.0, 0.0)

        P_kwh = float(df["ashp_inst_kwh"].iloc[k_start: k_end + 1].sum())

        end_ts   = df.index[k_end]
        t_out_c  = (float(df["t_out_c"].iloc[k_end])
                    if "t_out_c" in df.columns else np.nan)
        t_sink_c = float(ashp_model.sink_proxy(
            df["tank_mid_c"].iloc[k_end],
            df["tank_top_c"].iloc[k_end],
        ))

        records.append({
            "Q_kwh":      Q_kwh,
            "P_kwh":      P_kwh,
            "n_minutes":  n,
            "T_out_c":    t_out_c,
            "T_sink_c":   t_sink_c,
            "dtop_run_c": dtop_run_c,
            "dtsum_run_c": dtsum_run_c,
            "n_warming_nodes": n_warming_nodes,
        })
        end_timestamps.append(end_ts)

    logger.info(
        "Qualifying runs after run-level filters: %d (rejected dTop<%.2fC: %d, dTsum<%.2fC: %d, warming_nodes<%d: %d)",
        len(records),
        min_dtop_run_c,
        rejected_dtop,
        min_dtsum_run_c,
        rejected_dtsum,
        min_warming_nodes,
        rejected_nodes,
    )

    if not records:
        return pd.DataFrame(
            columns=[
                "Q_kwh",
                "P_kwh",
                "n_minutes",
                "T_out_c",
                "T_sink_c",
                "dtop_run_c",
                "dtsum_run_c",
                "n_warming_nodes",
            ]
        )

    result = pd.DataFrame(records, index=end_timestamps)
    return result


def back_calculate_ashp_heat(
    df: pd.DataFrame,
    st_col: str = "st_kwh",
    dt_s: float = 1800.0,
    min_run_minutes: int = 15,
    ashp_on_threshold: float = 0.05,
) -> pd.Series:
    """Back-calculate ASHP heat delivery [kWh] for ASHP-only intervals.

    Returns a Series of length ``len(df)`` with NaN for non-ASHP-only
    intervals (or non-qualifying-run rows in 1-minute mode).

    **30-minute mode** (``dt_s=1800``, default):
    An interval is ASHP-only if:
      - ``ashp_inst_kwh > ashp_on_threshold`` (default 0.05)
      - ``tank_top_c.diff() > 0.05`` (rising top temperature)
      - ``imm_tot_inst_kwh < 0.01``
      - ``st_col < 0.05``
      - all four tank temperatures are finite at this row and the previous row

    Each qualifying row gets an individual back-calculated Q value.

    **1-minute mode** (``dt_s=60``):
    Uses run-based aggregation via :func:`compute_ashp_runs_1min`.
    Qualifying runs of ≥ ``min_run_minutes`` consecutive ASHP-only minutes
    are identified; the total run Q is stored at the **run-end** timestamp
    (NaN everywhere else).  Also logs run statistics and the distribution
    of back-calculated COP values.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with tank temperatures, ASHP, immersion, and ST data.
    st_col : str
        Name of the solar-thermal energy column.
    dt_s : float
        Interval length in seconds.  Use ``60.0`` for 1-minute data.
    min_run_minutes : int
        Minimum run length to include in 1-minute mode (default 15).
    ashp_on_threshold : float
        Minimum ``ashp_inst_kwh`` to consider the ASHP running (30-min mode
        only; 1-min mode always uses 0.005 kWh).

    Returns
    -------
    pd.Series
        Back-calculated ASHP heat [kWh]; NaN for excluded intervals/rows.
    """
    if dt_s < 120:
        return _back_calc_runs_1min(df, st_col=st_col,
                                    min_run_minutes=min_run_minutes)
    return _back_calc_30min(df, st_col=st_col, dt_s=dt_s,
                            ashp_on_threshold=ashp_on_threshold)


def _back_calc_runs_1min(
    df: pd.DataFrame,
    st_col: str = "st_kwh",
    min_run_minutes: int = 15,
) -> pd.Series:
    """1-minute mode: return Q_total at run-end timestamps, NaN elsewhere."""
    Q_back = pd.Series(np.nan, index=df.index)

    runs_df = compute_ashp_runs_1min(df, st_col=st_col,
                                     min_run_minutes=min_run_minutes)

    if len(runs_df) == 0:
        logger.warning(
            "No qualifying ASHP runs found in 1-min data; returning all-NaN."
        )
        return Q_back

    # Log run statistics
    lens = runs_df["n_minutes"]
    cops = runs_df["Q_kwh"] / runs_df["P_kwh"].clip(lower=_MIN_POWER_KWH)
    logger.info(
        "Qualifying ASHP runs: %d  |  length min=%d median=%.1f max=%d min",
        len(runs_df), lens.min(), lens.median(), lens.max(),
    )
    logger.info(
        "Back-calculated COP:  mean=%.2f  median=%.2f",
        cops.mean(), cops.median(),
    )

    # Store Q_total at each run-end timestamp
    for end_ts, row in runs_df.iterrows():
        if end_ts in Q_back.index:
            Q_back.loc[end_ts] = row["Q_kwh"]

    return Q_back


def _back_calc_30min(
    df: pd.DataFrame,
    st_col: str = "st_kwh",
    dt_s: float = 1800.0,
    ashp_on_threshold: float = 0.1,
) -> pd.Series:
    """30-minute mode: per-interval back-calculation (original logic)."""
    tank_cols = ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]

    ashp_on = df["ashp_inst_kwh"].fillna(0) > ashp_on_threshold
    hx_on   = df["tank_top_c"].fillna(0).diff() > 0.05
    imm_off = df["imm_tot_inst_kwh"].fillna(0) < 0.01
    st_low  = (
        df[st_col].fillna(0) < 0.05
        if st_col in df.columns
        else pd.Series(True, index=df.index)
    )

    T = df[tank_cols].values
    finite_now  = np.all(np.isfinite(T), axis=1)
    finite_prev = np.roll(finite_now, 1)
    finite_prev[0] = False

    mask = ashp_on & hx_on & imm_off & st_low & pd.Series(finite_now & finite_prev, index=df.index)

    n_ashp_only = mask.sum()
    logger.info("ASHP-only intervals found: %d", n_ashp_only)

    Q_back = pd.Series(np.nan, index=df.index)

    if n_ashp_only < 50:
        logger.warning(
            "Insufficient ASHP-only intervals (%d < 50) for back-calculation; "
            "fallback to a = b * 3.0 retained.",
            n_ashp_only,
        )
        return Q_back

    ua_loss_default = TankParams().UA_loss
    T_amb = df["t_amb_c"].fillna(df["t_amb_c"].median()).values

    idx = np.where(mask.values)[0]
    for k in idx:
        storage_kJ = 0.0
        loss_sum   = 0.0
        for i in range(4):
            storage_kJ += NODE_CAP[i] * (T[k, i] - T[k - 1, i])
            loss_sum   += ua_loss_default[i] * (T[k - 1, i] - T_amb[k]) * dt_s

        Q_kJ = storage_kJ + loss_sum
        Q_back.iloc[k] = max(Q_kJ / 3600.0, 0.0)

    return Q_back


def prepare_inputs(df: pd.DataFrame, ashp_p: ashp_model.ASHPParams, dt_h: float = 5 / 60) -> dict:
    """Build arrays needed for tank simulation from a cleaned DataFrame.

    Returns dict with keys: T_meas (N,4), Q_st, Q_ashp, Q_imm, T_amb (all N,).
    """
    T_meas = df[["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]].values

    # ST energy
    if "st_kwh" in df.columns:
        Q_st = df["st_kwh"].fillna(0).values
    else:
        Q_st = solar_thermal.compute_st_energy(df, dt_minutes=dt_h * 60).values

    # ASHP heat from map
    T_sink = ashp_model.sink_proxy(df["tank_mid_c"].values, df["tank_top_c"].values)
    # Use measured electrical × COP from map to derive heat delivered
    cop = ashp_model.predict_cop(df["t_out_c"].values, T_sink, ashp_p)
    P_meas = df["ashp_inst_kwh"].fillna(0).values
    Q_ashp = P_meas * cop  # kWh heat = kWh elec × COP

    # Immersion
    Q_imm = df["imm_tot_inst_kwh"].fillna(0).values

    T_amb = df["t_amb_c"].fillna(df["t_amb_c"].median()).values

    return dict(
        T_meas=T_meas,
        Q_st=Q_st,
        Q_ashp=Q_ashp,
        Q_imm=Q_imm,
        T_amb=T_amb,
    )


def fit_tank_params(
    inputs: dict,
    *,
    max_nfev: int = 300,
    reg_weight: float = 0.01,
    dt_s: float = 300.0,
) -> tank_model.TankParams:
    """Fit tank parameters using one-step-ahead (teacher-forced) residuals.

    Each step resets to the measured state, so the residuals are the
    one-step prediction errors.  This avoids error accumulation and gives
    stable parameter estimates.
    """
    T_meas = inputs["T_meas"]
    Q_st   = inputs["Q_st"]
    Q_ashp = inputs["Q_ashp"]
    Q_imm  = inputs["Q_imm"]
    T_amb  = inputs["T_amb"]
    N = len(Q_st)
    steps = N - 1

    p0 = tank_model.TankParams()
    x0 = p0.to_vector()
    lb = tank_model.TankParams.lower_bounds()
    ub = tank_model.TankParams.upper_bounds()

    # Clamp x0 within bounds
    x0 = np.clip(x0, lb + 1e-8, ub - 1e-8)

    def residuals(x):
        p = tank_model.TankParams.from_vector(x)
        # One-step-ahead: predict T[k+1] from measured T[k]
        T_pred = np.zeros((steps, 4))
        for k in range(steps):
            T_pred[k] = tank_model.tank_step(
                T_meas[k],
                float(Q_st[k]), float(Q_ashp[k]),
                float(Q_imm[k]), float(T_amb[k]), p, dt_s=dt_s,
            )
        err = (T_pred - T_meas[1: steps + 1]).ravel()
        # Regularisation toward defaults
        reg = reg_weight * (x - p0.to_vector())
        return np.concatenate([err, reg])

    result = least_squares(
        residuals, x0,
        bounds=(lb, ub),
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=max_nfev,
        verbose=0,
    )
    logger.info("Tank fit cost: %.2f, nfev: %d", result.cost, result.nfev)
    return tank_model.TankParams.from_vector(result.x)


def run_identification(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.7,
    max_nfev: int = 300,
    ashp_params_path: Path | None = None,
    dt_s: float = 300.0,
) -> tuple[IdentificationResult, pd.DataFrame, pd.DataFrame]:
    """Full identification pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from :func:`src.data_loader.load_and_clean`.
    train_frac : float
        Fraction of data to use for training (default 0.7).
    max_nfev : int
        Maximum function evaluations for the tank optimiser (default 300).
    ashp_params_path : Path or None
        Optional path to a pre-fitted ASHP parameters JSON file (as produced
        by ``run_ashp_1min.py``).  If the file exists, ASHP fitting is
        skipped entirely and the parameters are loaded from JSON.  Expected
        format::

            {"ashp": {"a": [...], "b": [...]}}

        If ``None`` or the file does not exist, ASHP parameters are fitted
        from the 30-minute training data as usual.

    Returns
    -------
    result : IdentificationResult
    df_train : training slice
    df_val : validation slice
    """
    # Compute ST energy column
    df = df.copy()
    df["st_kwh"] = solar_thermal.compute_st_energy(df, dt_minutes=dt_s / 60.0)

    # Train/val split by time
    split_idx = int(len(df) * train_frac)
    df_train = df.iloc[:split_idx].copy()
    df_val   = df.iloc[split_idx:].copy()

    logger.info("Train: %d rows, Val: %d rows", len(df_train), len(df_val))

    # Step 1: ASHP parameter identification
    if ashp_params_path is not None and Path(ashp_params_path).exists():
        # Load pre-fitted ASHP params from JSON (e.g. from run_ashp_1min.py)
        with open(ashp_params_path, "r") as fh:
            ashp_data = json.load(fh)
        ashp_d = ashp_data["ashp"]
        ashp_p = ashp_model.ASHPParams(
            c=np.array(ashp_d["c"]) if "c" in ashp_d else None,
            a=np.array(ashp_d["a"]),
            b=np.array(ashp_d["b"]),
        )
        logger.info(
            "Loaded pre-fitted ASHP params from %s (skipping ASHP map fitting).",
            ashp_params_path,
        )
    else:
        if ashp_params_path is not None:
            logger.warning(
                "ASHP params file not found: %s — fitting ASHP from 30-min data.",
                ashp_params_path,
            )

        T_sink_train = ashp_model.sink_proxy(
            df_train["tank_mid_c"].values,
            df_train["tank_top_c"].values,
        )
        # Pass 1: fit power (b) coefficients only
        ashp_p = ashp_model.fit_ashp_maps(
            T_out=df_train["t_out_c"].values,
            T_sink=T_sink_train,
            Q_meas_kwh=None,
            P_meas_kwh=df_train["ashp_inst_kwh"].values,
        )

        # Pass 2: back-calculate heat and re-fit both a and b
        Q_back = back_calculate_ashp_heat(df_train, dt_s=dt_s)
        if Q_back.notna().sum() >= 50:
            ashp_p = ashp_model.fit_ashp_maps(
                T_out=df_train["t_out_c"].values,
                T_sink=T_sink_train,
                Q_meas_kwh=Q_back.values,
                P_meas_kwh=df_train["ashp_inst_kwh"].values,
            )
            logger.info("ASHP capacity fitted from back-calculated heat data.")
        else:
            logger.warning("Insufficient ASHP-only intervals for back-calculation; "
                           "fallback to a = b * 3.0 retained.")

    # Step 2: Fit tank on training data
    train_inputs = prepare_inputs(df_train, ashp_p, dt_h=dt_s / 3600.0)
    tank_p = fit_tank_params(train_inputs, max_nfev=max_nfev, dt_s=dt_s)

    result = IdentificationResult(
        tank_params=tank_p,
        ashp_params=ashp_p,
        hx_effectiveness=1.0,
        cost_history=[],
    )
    return result, df_train, df_val
