"""ETL: Load, clean, and prepare ERCOT LMP data."""

import pandas as pd
import numpy as np


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop unnamed index column
    if df.columns[0] == "" or df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])
    # Parse datetime
    df["ObsTime"] = pd.to_datetime(df["ObsTime"])
    df = df.sort_values("ObsTime").reset_index(drop=True)
    return df


def fill_temporal_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Create complete hourly index and forward-fill small gaps."""
    df = df.set_index("ObsTime")
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    n_missing = len(full_idx) - len(df)
    pct_missing = n_missing / len(full_idx) * 100
    print(f"  Temporal gaps: {n_missing} missing hours ({pct_missing:.2f}%)")

    df = df.reindex(full_idx)
    df.index.name = "ObsTime"

    # Forward-fill load/gen variables (change slowly); limit to 3 hours
    fill_cols = [
        "North_Load", "South_Load", "West_Load", "Houston_Load",
        "System_Load", "System_Solar", "System_Wind",
        "LZ_S_H_Wind", "LZ_N_Wind", "LZ_W_Wind",
        "Outages", "Generation", "Available_Gen", "Net_Load",
        "Responsive_Load", "Responsive_Offline_Gen",
        "outage_severity", "reserve_margin", "reserve_ratio",
    ]
    for c in fill_cols:
        if c in df.columns:
            df[c] = df[c].ffill(limit=3)

    # Interpolate price columns (limit 3 hours)
    for c in ["RT_LMP", "DA_LMP"]:
        if c in df.columns:
            df[c] = df[c].interpolate(method="linear", limit=3)

    # Fill Ramp after ffill of loads
    if "Ramp" in df.columns:
        df["Ramp"] = df["Ramp"].ffill(limit=3)

    df = df.reset_index()
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unnecessary columns, flag anomalies."""
    # Drop constant / redundant columns
    drop_cols = ["PNODE", "FlowDate", "ObsDate"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Drop time_of_day (will be re-derived via cyclical encoding)
    if "time_of_day" in df.columns:
        df = df.drop(columns=["time_of_day"])

    # Flag censored prices (ERCOT $9000 cap)
    df["is_price_cap"] = (df["RT_LMP"] >= 9000).astype(int)

    # Flag Winter Storm Uri period
    df["is_uri"] = (
        (df["ObsTime"] >= "2021-02-14") & (df["ObsTime"] <= "2021-02-19")
    ).astype(int)

    # Ensure numeric dtypes
    num_cols = df.select_dtypes(include="object").columns.tolist()
    if "ObsTime" in num_cols:
        num_cols.remove("ObsTime")
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def run_etl(path: str) -> pd.DataFrame:
    """Full ETL pipeline."""
    print("[ETL] Loading raw data...")
    df = load_raw(path)
    print(f"  Shape: {df.shape}")

    print("[ETL] Filling temporal gaps...")
    df = fill_temporal_gaps(df)
    print(f"  Shape after gap fill: {df.shape}")

    print("[ETL] Cleaning...")
    df = clean(df)
    print(f"  Shape after clean: {df.shape}")

    remaining_nulls = df.isnull().sum()
    remaining_nulls = remaining_nulls[remaining_nulls > 0]
    if len(remaining_nulls) > 0:
        print(f"  Remaining nulls:\n{remaining_nulls}")

    # Drop rows with any remaining nulls (from unfilled gaps at edges)
    before = len(df)
    df = df.dropna(subset=["RT_LMP", "System_Load"])
    print(f"  Dropped {before - len(df)} rows with null RT_LMP/System_Load")

    print(f"[ETL] Done. Final shape: {df.shape}")
    return df
