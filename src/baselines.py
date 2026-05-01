"""Baseline models for comparison: Naive, Linear Regression, Ridge."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


def run_baselines(train, val, test, feature_cols):
    """Train and evaluate baseline models, return comparison dict."""
    X_train = train[feature_cols].values
    X_test = test[feature_cols].values
    y_train = train["RT_LMP"].values
    y_test = test["RT_LMP"].values

    results = {}

    # ── Baseline 1: Naive Lag-1 (predict previous hour's price) ────
    naive_pred = test["Price_lag_1"].values
    naive_mae = mean_absolute_error(y_test, naive_pred)
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
    naive_median = np.median(np.abs(y_test - naive_pred))
    results["Naive (lag-1)"] = {"MAE": naive_mae, "RMSE": naive_rmse, "Median_AE": naive_median}

    # ── Baseline 2: Hourly Mean (predict average price for that hour-of-day) ──
    train_hourly_mean = train.groupby(train["ObsTime"].dt.hour)["RT_LMP"].mean()
    hourly_pred = test["ObsTime"].dt.hour.map(train_hourly_mean).values
    hourly_mae = mean_absolute_error(y_test, hourly_pred)
    hourly_rmse = np.sqrt(mean_squared_error(y_test, hourly_pred))
    hourly_median = np.median(np.abs(y_test - hourly_pred))
    results["Hourly Mean"] = {"MAE": hourly_mae, "RMSE": hourly_rmse, "Median_AE": hourly_median}

    # ── Baseline 3: Linear Regression ──────────────────────────────
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    lr_median = np.median(np.abs(y_test - lr_pred))
    results["Linear Regression"] = {"MAE": lr_mae, "RMSE": lr_rmse, "Median_AE": lr_median}

    # ── Baseline 4: Ridge Regression ───────────────────────────────
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    ridge_mae = mean_absolute_error(y_test, ridge_pred)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
    ridge_median = np.median(np.abs(y_test - ridge_pred))
    results["Ridge Regression"] = {"MAE": ridge_mae, "RMSE": ridge_rmse, "Median_AE": ridge_median}

    return results


def print_baseline_comparison(baseline_results, xgb_mae, xgb_rmse, xgb_median):
    """Print baseline comparison table."""
    print(f"\n{'='*70}")
    print("  BASELINE COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Model':<22} {'MAE ($/MWh)':>12} {'RMSE':>12} {'Median AE':>12} {'vs XGBoost':>12}")
    print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for name, metrics in baseline_results.items():
        improvement = (metrics["MAE"] - xgb_mae) / metrics["MAE"] * 100
        print(f"  {name:<22} ${metrics['MAE']:>10.2f} ${metrics['RMSE']:>10.2f} ${metrics['Median_AE']:>10.2f} {improvement:>+10.1f}%")

    print(f"  {'XGBoost (ours)':<22} ${xgb_mae:>10.2f} ${xgb_rmse:>10.2f} ${xgb_median:>10.2f} {'baseline':>12}")
    print(f"{'='*70}")

    # Print key takeaway
    best_baseline_mae = min(m["MAE"] for m in baseline_results.values())
    best_baseline_name = min(baseline_results, key=lambda k: baseline_results[k]["MAE"])
    pct_vs_best = (best_baseline_mae - xgb_mae) / best_baseline_mae * 100

    print(f"\n  >> XGBoost outperforms best baseline ({best_baseline_name}) by {pct_vs_best:.1f}% in MAE")
    print(f"  >> XGBoost outperforms Linear Regression by "
          f"{(baseline_results['Linear Regression']['RMSE'] - xgb_rmse) / baseline_results['Linear Regression']['RMSE'] * 100:.1f}% in RMSE")
