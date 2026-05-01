"""ERCOT Electricity Price Modeling Pipeline — Main Orchestrator."""

import sys
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.etl import run_etl
from src.features import build_features, get_feature_columns
from src.models import temporal_split, train_all_models, ensemble_predict
from src.evaluate import (
    evaluate_regressor, evaluate_ensemble,
    evaluate_spike_classifier, evaluate_regime_classifier,
    shap_analysis, plot_feature_importance, error_analysis,
)
from src.baselines import run_baselines, print_baseline_comparison

DATA_PATH = "localMarginalPrice-Ercot.csv"


def main():
    print("=" * 70)
    print("  ERCOT HB_NORTH Electricity Price Modeling Pipeline")
    print("=" * 70)

    # ── Phase 1: ETL ────────────────────────────────────────────────
    df = run_etl(DATA_PATH)

    # ── Phase 2: Feature Engineering ────────────────────────────────
    df = build_features(df)

    # ── Phase 3: Feature Selection ──────────────────────────────────
    feature_cols = get_feature_columns()
    # Verify all features exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"WARNING: Missing features: {missing}")
        feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"\n[Features] Using {len(feature_cols)} features: {feature_cols[:10]}... ")

    # ── Phase 4: Temporal Split ─────────────────────────────────────
    train, val, test = temporal_split(df)

    # ── Phase 4.5: Baseline Models ──────────────────────────────────
    baseline_results = run_baselines(train, val, test, feature_cols)

    # ── Phase 5: Train Models ───────────────────────────────────────
    models = train_all_models(train, val, feature_cols)

    # ── Phase 6: Evaluate on Validation Set ─────────────────────────
    print("\n" + "=" * 70)
    print("  VALIDATION SET EVALUATION")
    print("=" * 70)

    X_val = val[feature_cols]
    evaluate_regressor(
        models["regressor"], X_val,
        val["log_RT_LMP"].values, val["RT_LMP"].values, val, label="Val"
    )
    evaluate_spike_classifier(models["spike_clf"], X_val, val["is_spike"].values, label="Val")
    evaluate_regime_classifier(models["regime_clf"], X_val, val["regime"].values, label="Val")

    # Ensemble
    ens_pred_val = ensemble_predict(
        models["regime_clf"], models["regime_regressors"], val, feature_cols
    )
    evaluate_ensemble(ens_pred_val, val["log_RT_LMP"].values, val["RT_LMP"].values, val, label="Val")

    # ── Phase 7: Evaluate on Test Set ───────────────────────────────
    print("\n" + "=" * 70)
    print("  TEST SET EVALUATION")
    print("=" * 70)

    X_test = test[feature_cols]
    reg_metrics = evaluate_regressor(
        models["regressor"], X_test,
        test["log_RT_LMP"].values, test["RT_LMP"].values, test, label="Test"
    )
    evaluate_spike_classifier(models["spike_clf"], X_test, test["is_spike"].values, label="Test")
    evaluate_regime_classifier(models["regime_clf"], X_test, test["regime"].values, label="Test")

    # Ensemble on test
    ens_pred_test = ensemble_predict(
        models["regime_clf"], models["regime_regressors"], test, feature_cols
    )
    ens_metrics = evaluate_ensemble(
        ens_pred_test, test["log_RT_LMP"].values, test["RT_LMP"].values, test, label="Test"
    )

    # Error analysis (global regressor)
    pred_raw_test = np.sinh(models["regressor"].predict(X_test))
    error_analysis(test["RT_LMP"].values, pred_raw_test, test, label="Test")

    # ── Baseline Comparison ─────────────────────────────────────────
    print_baseline_comparison(
        baseline_results,
        xgb_mae=reg_metrics["mae_raw"],
        xgb_rmse=reg_metrics["rmse_raw"],
        xgb_median=reg_metrics["median_ae"],
    )

    # ── Phase 8: Feature Importance & SHAP ──────────────────────────
    print("\n" + "=" * 70)
    print("  FEATURE IMPORTANCE & SHAP")
    print("=" * 70)

    plot_feature_importance(models["regressor"], feature_cols, "regressor")
    plot_feature_importance(models["spike_clf"], feature_cols, "spike_clf")

    shap_analysis(models["regressor"], X_test, feature_cols, "regressor")

    # ── Phase 9: Save Models ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SAVING MODELS")
    print("=" * 70)

    os.makedirs("models", exist_ok=True)
    models["regressor"].save_model("models/regressor.json")
    models["spike_clf"].save_model("models/spike_clf.json")
    models["regime_clf"].save_model("models/regime_clf.json")
    for rid, m in models["regime_regressors"].items():
        m.save_model(f"models/regime_regressor_{rid}.json")

    # Save feature column list for inference
    import json
    with open("models/feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    print(f"  Saved {3 + len(models['regime_regressors'])} models + feature config to models/")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Global Regressor MAE: ${reg_metrics['mae_raw']:.2f}/MWh")
    print(f"  Ensemble MAE:         ${ens_metrics['mae_raw']:.2f}/MWh")
    print(f"  Output directory:     output/")
    print(f"  Plots saved:          regressor, spike PR curve, regime confusion,")
    print(f"                        SHAP summary, feature importance, error analysis")
    print("=" * 70)


if __name__ == "__main__":
    main()
