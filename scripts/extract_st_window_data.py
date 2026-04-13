#!/usr/bin/env python3
"""Extract ST-window interval data with GTI and write a diagnostics CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows-csv",
        type=Path,
        default=Path("ST_fitting/diagnostics/st_windows.csv"),
        help="CSV with window start/end timestamps.",
    )
    parser.add_argument(
        "--full-ds-csv",
        type=Path,
        default=Path("data/FullDS_Findhorn_5min.csv"),
        help="Main 5-min plant dataset.",
    )
    parser.add_argument(
        "--gti-csv",
        type=Path,
        default=Path("data/hist_GTI_5min.csv"),
        help="5-min GTI timeseries CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("ST_fitting/diagnostics/st_windows_with_gti.csv"),
        help="Path for extracted interval-level output CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    windows = pd.read_csv(args.windows_csv)
    full_ds = pd.read_csv(args.full_ds_csv)
    gti = pd.read_csv(args.gti_csv)

    windows["start"] = pd.to_datetime(windows["start"])
    windows["end"] = pd.to_datetime(windows["end"])

    full_ds["timestamp"] = pd.to_datetime(full_ds["Time"], dayfirst=True)
    gti["timestamp"] = pd.to_datetime(gti["time"])

    full_cols = {
        "ST Power [kW]": "ST Power [kW]",
        "ST Flow T [°C]": "ST Flow T [C]",
        "ST Tot Energy [MWh]": "ST Tot Energy [MWh]",
        "PV Inst [kW]": "PV Inst [kW]",
        "Tank Bottom [°C]": "Tank Bottom [C]",
    }

    missing = [col for col in full_cols if col not in full_ds.columns]
    if missing:
        raise KeyError(f"Missing expected columns in FullDS dataset: {missing}")

    full_sel = full_ds[["timestamp", *full_cols.keys()]].rename(columns=full_cols)
    gti_sel = gti[["timestamp", "GTI"]]

    rows = []
    for w in windows.itertuples(index=False):
        mask = (full_sel["timestamp"] >= w.start) & (full_sel["timestamp"] <= w.end)
        chunk = full_sel.loc[mask].copy()
        chunk.insert(0, "window_id", w.window_id)
        chunk.insert(1, "window_start", w.start)
        chunk.insert(2, "window_end", w.end)
        chunk.insert(3, "Q_st_meas_kwh", w.Q_st_meas_kwh)
        chunk.insert(4, "expected_n_intervals", int(w.n_intervals))
        chunk.insert(5, "actual_n_intervals_window", int(mask.sum()))
        rows.append(chunk)

    out = pd.concat(rows, ignore_index=True)
    out = out.merge(gti_sel, on="timestamp", how="left", validate="many_to_one")

    expected_total = int(windows["n_intervals"].sum())
    actual_total = len(out)
    if actual_total != expected_total:
        raise ValueError(
            f"Extracted {actual_total} rows, but expected {expected_total} from windows file."
        )

    if out["GTI"].isna().any():
        n_missing = int(out["GTI"].isna().sum())
        raise ValueError(f"GTI missing for {n_missing} extracted intervals after timestamp merge.")

    out = out.sort_values(["window_id", "timestamp"]).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    print(f"Wrote {actual_total} intervals to {args.output_csv}")


if __name__ == "__main__":
    main()
