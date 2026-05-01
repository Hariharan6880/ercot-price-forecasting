"""Evaluation: metrics, SHAP, error analysis, and visualization."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    classification_report, confusion_matrix,
    precision_recall_curve, fbeta_score, roc_auc_score,
    average_precision_score,
)
import shap

OUTPUT_DIR = "output"


def _ensure_output_dir():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Regression Metrics ──────────────────────────────────────────────

def evaluate_regressor(model, X, y_log, y_raw, df, label="Test"):
    """Evaluate the global regressor on log-transformed and raw scales."""
    _ensure_output_dir()
    pred_log = model.predict(X)
    pred_raw = np.sinh(pred_log)  # invert arcsinh

    mae_log = mean_absolute_error(y_log, pred_log)
    mae_raw = mean_absolute_error(y_raw, pred_raw)
    rmse_raw = np.sqrt(mean_squared_error(y_raw, pred_raw))
    median_ae = np.median(np.abs(y_raw - pred_raw))

    print(f"\n{'='*60}")
    print(f"[Regressor] {label} Set Metrics")
    print(f"  MAE (log scale):  {mae_log:.4f}")
    print(f"  MAE ($/MWh):      {mae_raw:.2f}")
    print(f"  Median AE:        {median_ae:.2f}")
    print(f"  RMSE ($/MWh):     {rmse_raw:.2f}")

    # Regime-stratified MAE
    regimes = df["regime"].values
    for rid, name in [(0, "Normal (RM>5k)"), (1, "Stressed (1k-5k)"), (2, "Scarcity (<1k)")]:
        mask = regimes == rid
        if mask.sum() > 0:
            r_mae = mean_absolute_error(y_raw[mask], pred_raw[mask])
            print(f"  MAE Regime {name}: ${r_mae:.2f} (n={mask.sum()})")

    # Spike capture rate
    actual_spikes = y_raw > 300
    pred_spikes = pred_raw > 300
    if actual_spikes.sum() > 0:
        capture = (actual_spikes & pred_spikes).sum() / actual_spikes.sum() * 100
        print(f"  Spike capture rate (>$300): {capture:.1f}%")

    # Predicted vs Actual plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(y_raw, pred_raw, alpha=0.1, s=5)
    lims = [min(y_raw.min(), pred_raw.min()), max(y_raw.max(), pred_raw.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual RT_LMP ($/MWh)")
    ax.set_ylabel("Predicted RT_LMP ($/MWh)")
    ax.set_title(f"Predicted vs Actual — {label}")

    ax = axes[1]
    # Zoom into normal range
    mask_zoom = y_raw < 200
    ax.scatter(y_raw[mask_zoom], pred_raw[mask_zoom], alpha=0.1, s=5)
    ax.plot([y_raw[mask_zoom].min(), 200], [y_raw[mask_zoom].min(), 200], "r--")
    ax.set_xlabel("Actual RT_LMP ($/MWh)")
    ax.set_ylabel("Predicted RT_LMP ($/MWh)")
    ax.set_title(f"Predicted vs Actual (Zoomed <$200) — {label}")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/regressor_{label.lower()}.png", dpi=150)
    plt.close()

    return {"mae_log": mae_log, "mae_raw": mae_raw, "rmse_raw": rmse_raw, "median_ae": median_ae}


# ── Ensemble Regressor Metrics ──────────────────────────────────────

def evaluate_ensemble(pred_log, y_log, y_raw, df, label="Test"):
    """Evaluate ensemble predictions."""
    _ensure_output_dir()
    pred_raw = np.sinh(pred_log)

    mae_raw = mean_absolute_error(y_raw, pred_raw)
    rmse_raw = np.sqrt(mean_squared_error(y_raw, pred_raw))
    median_ae = np.median(np.abs(y_raw - pred_raw))

    print(f"\n{'='*60}")
    print(f"[Ensemble] {label} Set Metrics")
    print(f"  MAE ($/MWh):      {mae_raw:.2f}")
    print(f"  Median AE:        {median_ae:.2f}")
    print(f"  RMSE ($/MWh):     {rmse_raw:.2f}")

    regimes = df["regime"].values
    for rid, name in [(0, "Normal"), (1, "Stressed"), (2, "Scarcity")]:
        mask = regimes == rid
        if mask.sum() > 0:
            r_mae = mean_absolute_error(y_raw[mask], pred_raw[mask])
            print(f"  MAE Regime {name}: ${r_mae:.2f} (n={mask.sum()})")

    return {"mae_raw": mae_raw, "rmse_raw": rmse_raw, "median_ae": median_ae}


# ── Spike Classifier Metrics ───────────────────────────────────────

def evaluate_spike_classifier(model, X, y_true, label="Test"):
    """Evaluate spike classifier with F2, PR-AUC, and threshold analysis."""
    _ensure_output_dir()
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    f2 = fbeta_score(y_true, y_pred, beta=2)
    pr_auc = average_precision_score(y_true, y_prob)

    print(f"\n{'='*60}")
    print(f"[Spike Classifier] {label} Set Metrics")
    print(f"  F2 Score:   {f2:.4f}")
    print(f"  PR-AUC:     {pr_auc:.4f}")
    if len(np.unique(y_true)) > 1:
        roc = roc_auc_score(y_true, y_prob)
        print(f"  ROC-AUC:    {roc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Spike"]))

    # Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — Spike Detection ({label})")
    ax.axhline(y=y_true.mean(), color="r", linestyle="--", label="Baseline")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/spike_pr_curve_{label.lower()}.png", dpi=150)
    plt.close()

    return {"f2": f2, "pr_auc": pr_auc}


# ── Regime Classifier Metrics ──────────────────────────────────────

def evaluate_regime_classifier(model, X, y_true, label="Test"):
    """Evaluate regime classifier with confusion matrix and per-class F1."""
    _ensure_output_dir()
    y_pred = model.predict(X)

    print(f"\n{'='*60}")
    print(f"[Regime Classifier] {label} Set Metrics")
    print(classification_report(
        y_true, y_pred,
        target_names=["Normal (RM>5k)", "Stressed (1k-5k)", "Scarcity (<1k)"]
    ))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Stressed", "Scarcity"],
                yticklabels=["Normal", "Stressed", "Scarcity"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Regime Confusion Matrix — {label}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/regime_confusion_{label.lower()}.png", dpi=150)
    plt.close()


# ── SHAP Analysis ──────────────────────────────────────────────────

def shap_analysis(model, X, feature_cols, model_name="regressor", max_samples=2000):
    """Compute and plot SHAP values for a model."""
    _ensure_output_dir()
    print(f"\n[SHAP] Computing for {model_name} (sampling {min(len(X), max_samples)} rows)...")

    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=42)
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols,
                      show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_summary_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Feature importance bar
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols,
                      plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_importance_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[SHAP] Saved to {OUTPUT_DIR}/shap_*_{model_name}.png")
    return shap_values


# ── Feature Importance ─────────────────────────────────────────────

def plot_feature_importance(model, feature_cols, model_name="regressor", top_n=20):
    """Plot XGBoost built-in feature importance (gain)."""
    _ensure_output_dir()
    imp = model.feature_importances_
    idx = np.argsort(imp)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(idx)), imp[idx])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_cols[i] for i in idx])
    ax.set_xlabel("Gain")
    ax.set_title(f"Feature Importance (Gain) — {model_name}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feat_importance_{model_name}.png", dpi=150)
    plt.close()


# ── Error Analysis ─────────────────────────────────────────────────

def error_analysis(y_raw, pred_raw, df, label="Test"):
    """Time series of errors and worst misses."""
    _ensure_output_dir()
    errors = y_raw - pred_raw
    abs_errors = np.abs(errors)

    # Worst misses
    worst_idx = np.argsort(abs_errors)[-10:][::-1]
    print(f"\n[Error Analysis] Top 10 worst misses ({label}):")
    print(f"  {'ObsTime':<22} {'Actual':>10} {'Predicted':>10} {'Error':>10} {'Regime':>8}")
    for i in worst_idx:
        t = df.iloc[i]["ObsTime"]
        print(f"  {str(t):<22} ${y_raw[i]:>9.2f} ${pred_raw[i]:>9.2f} ${errors[i]:>9.2f} {int(df.iloc[i]['regime']):>8}")

    # Error time series
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    times = df["ObsTime"].values

    axes[0].plot(times, y_raw, alpha=0.7, linewidth=0.5, label="Actual")
    axes[0].plot(times, pred_raw, alpha=0.7, linewidth=0.5, label="Predicted")
    axes[0].set_ylabel("RT_LMP ($/MWh)")
    axes[0].legend()
    axes[0].set_title(f"Price: Actual vs Predicted — {label}")

    axes[1].bar(times, errors, width=0.04, alpha=0.5, color="red")
    axes[1].set_ylabel("Error ($/MWh)")
    axes[1].set_title("Prediction Error Over Time")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/error_timeseries_{label.lower()}.png", dpi=150)
    plt.close()
