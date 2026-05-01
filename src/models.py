"""Models: XGBoost regression, spike classification, regime classification, ensemble."""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import TimeSeriesSplit


def temporal_split(df: pd.DataFrame):
    """Split data temporally: Train 2019-2022, Val 2023, Test 2024+."""
    train = df[df["ObsTime"] < "2023-01-01"].copy()
    val = df[(df["ObsTime"] >= "2023-01-01") & (df["ObsTime"] < "2024-01-01")].copy()
    test = df[df["ObsTime"] >= "2024-01-01"].copy()

    print(f"[Split] Train: {len(train)} rows ({train['ObsTime'].min()} to {train['ObsTime'].max()})")
    print(f"[Split] Val:   {len(val)} rows ({val['ObsTime'].min()} to {val['ObsTime'].max()})")
    print(f"[Split] Test:  {len(test)} rows ({test['ObsTime'].min()} to {test['ObsTime'].max()})")

    # Regime distribution
    for name, subset in [("Train", train), ("Val", val), ("Test", test)]:
        regime_pct = subset["regime"].value_counts(normalize=True).sort_index()
        spike_pct = subset["is_spike"].mean() * 100
        print(f"  {name} regimes: {dict(regime_pct.round(3))}, spike%: {spike_pct:.1f}%")

    return train, val, test


def train_regressor(X_train, y_train, X_val, y_val):
    """Train XGBoost regressor on arcsinh-transformed price."""
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=30,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    return model


def train_spike_classifier(X_train, y_train, X_val, y_val):
    """Train XGBoost binary classifier for price spikes (>$300)."""
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        gamma=1,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    return model


def train_regime_classifier(X_train, y_train, X_val, y_val):
    """Train XGBoost 3-class regime classifier."""
    # Compute sample weights inversely proportional to class frequency
    counts = np.bincount(y_train, minlength=3)
    weights = len(y_train) / (3 * counts + 1)
    sample_weight = np.array([weights[c] for c in y_train])

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=800,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    return model


def train_regime_specific_regressors(df_train, df_val, feature_cols):
    """Train separate regression models for each regime."""
    models = {}
    for regime_id, label in [(0, "Normal"), (1, "Stressed"), (2, "Scarcity")]:
        rt = df_train[df_train["regime"] == regime_id]
        rv = df_val[df_val["regime"] == regime_id]

        if len(rt) < 50:
            print(f"  Regime {label}: only {len(rt)} training samples, skipping")
            continue

        # Adjust complexity for regime
        depth = {0: 4, 1: 6, 2: 7}[regime_id]
        n_est = {0: 500, 1: 1000, 2: 1200}[regime_id]

        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=n_est,
            max_depth=depth,
            learning_rate=0.05,
            min_child_weight=10 if regime_id == 2 else 30,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            random_state=42,
            n_jobs=-1,
        )

        X_rt, y_rt = rt[feature_cols], rt["log_RT_LMP"]
        if len(rv) > 10:
            X_rv, y_rv = rv[feature_cols], rv["log_RT_LMP"]
            model.set_params(early_stopping_rounds=30)
            model.fit(X_rt, y_rt, eval_set=[(X_rv, y_rv)], verbose=0)
        else:
            model.fit(X_rt, y_rt, verbose=0)

        print(f"  Regime {label}: trained on {len(rt)} samples")
        models[regime_id] = model

    return models


def ensemble_predict(regime_model, regime_regressors, X, feature_cols):
    """Regime-aware ensemble: weight regime-specific predictions by regime probabilities."""
    regime_probs = regime_model.predict_proba(X[feature_cols])
    predictions = np.zeros(len(X))

    for regime_id, model in regime_regressors.items():
        pred = model.predict(X[feature_cols])
        # Clip arcsinh predictions to valid range (arcsinh(9000) ≈ 9.8)
        pred = np.clip(pred, -7, 10)
        predictions += regime_probs[:, regime_id] * pred

    return predictions


def train_all_models(train, val, feature_cols):
    """Train all models and return them."""
    X_train = train[feature_cols]
    X_val = val[feature_cols]

    print("\n[Model A] Price Regressor (Huber loss, log-transformed target)...")
    reg_model = train_regressor(
        X_train, train["log_RT_LMP"], X_val, val["log_RT_LMP"]
    )

    print("\n[Model B] Spike Classifier (binary, F2-optimized)...")
    spike_model = train_spike_classifier(
        X_train, train["is_spike"], X_val, val["is_spike"]
    )

    print("\n[Model C] Regime Classifier (3-class)...")
    regime_model = train_regime_classifier(
        X_train, train["regime"], X_val, val["regime"]
    )

    print("\n[Model D] Regime-Specific Regressors...")
    regime_regressors = train_regime_specific_regressors(
        train, val, feature_cols
    )

    return {
        "regressor": reg_model,
        "spike_clf": spike_model,
        "regime_clf": regime_model,
        "regime_regressors": regime_regressors,
    }
