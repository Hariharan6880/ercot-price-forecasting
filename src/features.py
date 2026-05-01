"""Feature engineering: derive all features from cleaned ERCOT data."""

import numpy as np
import pandas as pd


def add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scarcity, stress, and renewable features."""
    # Scarcity Proximity: relative threshold scaled by system size
    threshold = 0.07 * df["System_Load"]
    df["Scarcity_Proximity"] = np.clip(1 - df["reserve_margin"] / threshold, 0, None)

    # Load Stress: demand under constraint
    df["Load_Stress"] = df["System_Load"] * df["Scarcity_Proximity"]

    # Renewable Contribution
    df["Renewable_Contribution"] = (
        (df["System_Wind"] + df["System_Solar"]) / df["System_Load"]
    )

    # West Wind Share (concentration risk)
    df["West_Wind_Share"] = df["LZ_W_Wind"] / (df["System_Wind"] + 1)

    # Houston Load Share
    df["Houston_Load_Share"] = df["Houston_Load"] / df["System_Load"]

    return df


def add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag, diff, and rolling features for temporal dynamics."""
    # 3-hour cumulative ramp
    df["Ramp_3hr"] = df["Ramp"].rolling(3, min_periods=1).sum()

    # Hours to Scarcity
    ramp_clipped = df["Ramp"].clip(lower=100)
    df["Hours_to_Scarcity"] = (df["reserve_margin"] / ramp_clipped).clip(upper=24)

    # Dynamic Stress: reserve ratio + trajectory
    df["Dynamic_Stress"] = df["reserve_ratio"] + 2 * df["reserve_ratio"].diff()

    # Reserve margin changes
    df["RM_change_1hr"] = df["reserve_margin"].diff(1)
    df["RM_change_3hr"] = df["reserve_margin"].diff(3)

    # Exhaustion Rate
    df["Exhaustion_Rate"] = df["Ramp_3hr"] / df["reserve_margin"].clip(lower=100)

    # Lagged features
    df["Load_lag_1"] = df["System_Load"].shift(1)
    df["Load_lag_3"] = df["System_Load"].shift(3)
    df["Wind_lag_1"] = df["System_Wind"].shift(1)
    df["Wind_change_3hr"] = df["System_Wind"].diff(3)
    df["Price_lag_1"] = df["RT_LMP"].shift(1)
    df["Price_lag_3"] = df["RT_LMP"].shift(3)

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical time encodings and calendar features."""
    hour = df["ObsTime"].dt.hour
    month = df["ObsTime"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["is_weekend"] = (df["ObsTime"].dt.dayofweek >= 5).astype(int)
    df["day_of_year"] = df["ObsTime"].dt.dayofyear

    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create target variables for regression and classification."""
    # Log-transformed price using signed log to handle negatives naturally
    # asinh(x) ≈ log(2x) for large x, handles negatives, zero-preserving
    df["log_RT_LMP"] = np.arcsinh(df["RT_LMP"])

    # Spike binary
    df["is_spike"] = (df["RT_LMP"] > 300).astype(int)

    # Regime based on reserve margin
    df["regime"] = np.where(
        df["reserve_margin"] > 5000, 0,
        np.where(df["reserve_margin"] > 1000, 1, 2)
    )

    return df


def get_feature_columns() -> list:
    """Return the list of feature columns for modeling."""
    primary = [
        "System_Load", "System_Wind", "System_Solar",
        "LZ_W_Wind", "LZ_S_H_Wind", "LZ_N_Wind",
        "Available_Gen", "outage_severity", "Ramp",
        "reserve_margin", "Scarcity_Proximity",
    ]
    engineered = [
        "Load_Stress", "Renewable_Contribution",
        "West_Wind_Share", "Houston_Load_Share",
        "Dynamic_Stress", "Hours_to_Scarcity", "Exhaustion_Rate",
    ]
    lags = [
        "Ramp_3hr", "RM_change_1hr", "RM_change_3hr",
        "Wind_change_3hr", "Price_lag_1", "Price_lag_3",
        "Load_lag_1",
    ]
    temporal = [
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "is_weekend", "day_of_year",
    ]
    flags = ["is_price_cap", "is_uri"]
    return primary + engineered + lags + temporal + flags


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature engineering steps."""
    print("[Features] Adding core features...")
    df = add_core_features(df)

    print("[Features] Adding lag/rolling features...")
    df = add_lag_rolling_features(df)

    print("[Features] Adding temporal features...")
    df = add_temporal_features(df)

    print("[Features] Adding targets...")
    df = add_targets(df)

    # Drop rows with NaN from lag/diff operations
    before = len(df)
    feature_cols = get_feature_columns()
    df = df.dropna(subset=feature_cols)
    print(f"[Features] Dropped {before - len(df)} rows with NaN from lags")
    print(f"[Features] Done. Final shape: {df.shape}")

    return df
