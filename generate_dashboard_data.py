"""Generate predictions CSV for the Streamlit dashboard."""

import sys
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.etl import run_etl
from src.features import build_features, get_feature_columns
from src.models import temporal_split
from src.baselines import run_baselines
from xgboost import XGBRegressor, XGBClassifier

DATA_PATH = "localMarginalPrice-Ercot.csv"
MODEL_DIR = "models"


def main():
    # Load and process data
    df = run_etl(DATA_PATH)
    df = build_features(df)
    feature_cols = get_feature_columns()
    feature_cols = [c for c in feature_cols if c in df.columns]

    train, val, test = temporal_split(df)

    # Load saved models
    regressor = XGBRegressor()
    regressor.load_model(os.path.join(MODEL_DIR, "regressor.json"))

    spike_clf = XGBClassifier()
    spike_clf.load_model(os.path.join(MODEL_DIR, "spike_clf.json"))

    regime_clf = XGBClassifier()
    regime_clf.load_model(os.path.join(MODEL_DIR, "regime_clf.json"))

    regime_regressors = {}
    for rid in [0, 1, 2]:
        path = os.path.join(MODEL_DIR, f"regime_regressor_{rid}.json")
        if os.path.exists(path):
            m = XGBRegressor()
            m.load_model(path)
            regime_regressors[rid] = m

    # Generate predictions for ALL data (train + val + test)
    for subset, label in [(train, "train"), (val, "val"), (test, "test")]:
        X = subset[feature_cols]

        # Regressor
        pred_log = regressor.predict(X)
        subset = subset.copy()
        subset["predicted_price"] = np.sinh(pred_log)
        subset["prediction_error"] = subset["RT_LMP"] - subset["predicted_price"]
        subset["abs_error"] = np.abs(subset["prediction_error"])

        # Spike classifier
        subset["spike_probability"] = spike_clf.predict_proba(X)[:, 1]
        subset["spike_predicted"] = (subset["spike_probability"] > 0.5).astype(int)

        # Regime classifier
        regime_probs = regime_clf.predict_proba(X)
        subset["regime_prob_normal"] = regime_probs[:, 0]
        subset["regime_prob_stressed"] = regime_probs[:, 1]
        subset["regime_prob_scarcity"] = regime_probs[:, 2]
        subset["predicted_regime"] = np.argmax(regime_probs, axis=1)

        # Ensemble prediction
        ens_pred = np.zeros(len(X))
        for rid, m in regime_regressors.items():
            p = np.clip(m.predict(X), -7, 10)
            ens_pred += regime_probs[:, rid] * np.sinh(p)
        subset["ensemble_price"] = ens_pred

        # Naive baseline (for comparison in dashboard)
        subset["naive_prediction"] = subset["Price_lag_1"]

        # Assign split label
        subset["split"] = label

        if label == "train":
            all_preds = subset
        else:
            all_preds = pd.concat([all_preds, subset], ignore_index=True)

    # Select columns for dashboard
    dashboard_cols = [
        "ObsTime", "split", "RT_LMP", "predicted_price", "ensemble_price",
        "prediction_error", "abs_error", "naive_prediction",
        "spike_probability", "spike_predicted", "is_spike",
        "regime", "predicted_regime",
        "regime_prob_normal", "regime_prob_stressed", "regime_prob_scarcity",
        "System_Load", "System_Wind", "System_Solar", "reserve_margin",
        "Scarcity_Proximity", "Load_Stress", "Available_Gen", "Ramp",
    ]
    dashboard_cols = [c for c in dashboard_cols if c in all_preds.columns]
    output = all_preds[dashboard_cols].copy()

    # Save
    os.makedirs("dashboard_data", exist_ok=True)
    output.to_csv("dashboard_data/predictions.csv", index=False)
    print(f"Saved predictions: {output.shape} to dashboard_data/predictions.csv")

    # Also run baselines and save summary
    baseline_results = run_baselines(train, val, test, feature_cols)

    # Test set metrics
    test_mask = output["split"] == "test"
    test_data = output[test_mask]
    xgb_mae = test_data["abs_error"].mean()
    xgb_median = test_data["abs_error"].median()
    xgb_rmse = np.sqrt((test_data["prediction_error"] ** 2).mean())

    metrics = {
        "xgb_mae": round(xgb_mae, 2),
        "xgb_median_ae": round(xgb_median, 2),
        "xgb_rmse": round(xgb_rmse, 2),
    }
    for name, m in baseline_results.items():
        key = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
        metrics[f"{key}_mae"] = round(m["MAE"], 2)
        metrics[f"{key}_rmse"] = round(m["RMSE"], 2)

    import json
    with open("dashboard_data/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to dashboard_data/metrics.json")


if __name__ == "__main__":
    main()
