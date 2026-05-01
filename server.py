"""FastAPI server for ERCOT electricity price prediction.

Exposes endpoints:
  POST /predict        — single-hour prediction from raw grid measurements
  POST /predict/batch  — multi-hour batch prediction
  GET  /health         — health check

Requires trained models in the models/ directory (run main.py first).
Start with:  uvicorn server:app --host 0.0.0.0 --port 8000
"""

import json
import os
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import XGBRegressor, XGBClassifier

# ── Load models & config at startup ────────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

def _load_models():
    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(
            "models/ directory not found. Run main.py first to train and save models."
        )

    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)

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

    return {
        "regressor": regressor,
        "spike_clf": spike_clf,
        "regime_clf": regime_clf,
        "regime_regressors": regime_regressors,
        "feature_cols": feature_cols,
    }


models = _load_models()
FEATURE_COLS = models["feature_cols"]

app = FastAPI(
    title="ERCOT Price Prediction API",
    description="Predict real-time electricity prices at HB_NORTH using XGBoost models",
    version="1.0.0",
)


# ── Request / Response schemas ─────────────────────────────────────

class GridState(BaseModel):
    """Raw grid measurements for a single hour."""
    ObsTime: str = Field(..., description="ISO datetime, e.g. '2024-08-15 14:00:00'")
    System_Load: float = Field(..., description="Total system load (MW)")
    System_Wind: float = Field(..., description="Total wind generation (MW)")
    System_Solar: float = Field(..., description="Total solar generation (MW)")
    LZ_W_Wind: float = Field(..., description="West zone wind (MW)")
    LZ_S_H_Wind: float = Field(..., description="South-Houston zone wind (MW)")
    LZ_N_Wind: float = Field(..., description="North zone wind (MW)")
    Houston_Load: float = Field(..., description="Houston load (MW)")
    Available_Gen: float = Field(..., description="Available generation capacity (MW)")
    Net_Load: float = Field(..., description="Net load = System_Load - Wind - Solar (MW)")
    Outages: float = Field(..., description="Generation outages (MW)")
    Ramp: float = Field(..., description="Load ramp (MW/hr)")
    outage_severity: float = Field(..., description="Outage severity ratio")
    reserve_margin: float = Field(..., description="Available_Gen - Net_Load (MW)")
    reserve_ratio: float = Field(..., description="Net_Load / Available_Gen")

    # Lag values (from previous hours — caller must supply)
    Price_lag_1: float = Field(..., description="RT_LMP from 1 hour ago ($/MWh)")
    Price_lag_3: float = Field(..., description="RT_LMP from 3 hours ago ($/MWh)")
    Load_lag_1: float = Field(..., description="System_Load from 1 hour ago (MW)")
    Load_lag_3: float = Field(..., description="System_Load from 3 hours ago (MW)")
    Wind_lag_1: float = Field(..., description="System_Wind from 1 hour ago (MW)")

    # Rolling / diff values (caller computes or we compute if history given)
    Ramp_3hr: float = Field(..., description="Sum of Ramp over last 3 hours")
    RM_change_1hr: float = Field(..., description="reserve_margin change vs 1hr ago")
    RM_change_3hr: float = Field(..., description="reserve_margin change vs 3hr ago")
    Wind_change_3hr: float = Field(..., description="System_Wind change vs 3hr ago")

    # Optional flags
    is_price_cap: int = Field(0, description="1 if current price at $9000 cap")
    is_uri: int = Field(0, description="1 if Winter Storm Uri period")


class PredictionResponse(BaseModel):
    ObsTime: str
    predicted_price: float = Field(..., description="Predicted RT_LMP ($/MWh)")
    ensemble_price: float = Field(..., description="Regime-ensemble predicted price ($/MWh)")
    spike_probability: float = Field(..., description="Probability of price spike (>$300)")
    is_spike_predicted: bool
    predicted_regime: str
    regime_probabilities: dict


class BatchRequest(BaseModel):
    observations: list[GridState]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


# ── Feature computation ────────────────────────────────────────────

def _compute_features(obs: GridState) -> pd.DataFrame:
    """Build the full feature vector from a single GridState observation."""
    dt = pd.Timestamp(obs.ObsTime)

    # Core derived features
    threshold = 0.07 * obs.System_Load
    scarcity_proximity = max(0, 1 - obs.reserve_margin / threshold) if threshold > 0 else 0
    load_stress = obs.System_Load * scarcity_proximity
    renewable_contribution = (
        (obs.System_Wind + obs.System_Solar) / obs.System_Load
        if obs.System_Load > 0 else 0
    )
    west_wind_share = obs.LZ_W_Wind / (obs.System_Wind + 1)
    houston_load_share = obs.Houston_Load / obs.System_Load if obs.System_Load > 0 else 0

    # Dynamic stress (approx — uses reserve_ratio diff from RM_change)
    rr_change = (obs.RM_change_1hr / (obs.Available_Gen + 1)) if obs.Available_Gen > 0 else 0
    dynamic_stress = obs.reserve_ratio + 2 * rr_change

    # Hours to scarcity
    ramp_clipped = max(obs.Ramp, 100)
    hours_to_scarcity = min(obs.reserve_margin / ramp_clipped, 24)

    # Exhaustion rate
    exhaustion_rate = obs.Ramp_3hr / max(obs.reserve_margin, 100)

    # Temporal features
    hour_sin = np.sin(2 * np.pi * dt.hour / 24)
    hour_cos = np.cos(2 * np.pi * dt.hour / 24)
    month_sin = np.sin(2 * np.pi * dt.month / 12)
    month_cos = np.cos(2 * np.pi * dt.month / 12)
    is_weekend = 1 if dt.dayofweek >= 5 else 0
    day_of_year = dt.dayofyear

    row = {
        "System_Load": obs.System_Load,
        "System_Wind": obs.System_Wind,
        "System_Solar": obs.System_Solar,
        "LZ_W_Wind": obs.LZ_W_Wind,
        "LZ_S_H_Wind": obs.LZ_S_H_Wind,
        "LZ_N_Wind": obs.LZ_N_Wind,
        "Available_Gen": obs.Available_Gen,
        "outage_severity": obs.outage_severity,
        "Ramp": obs.Ramp,
        "reserve_margin": obs.reserve_margin,
        "Scarcity_Proximity": scarcity_proximity,
        "Load_Stress": load_stress,
        "Renewable_Contribution": renewable_contribution,
        "West_Wind_Share": west_wind_share,
        "Houston_Load_Share": houston_load_share,
        "Dynamic_Stress": dynamic_stress,
        "Hours_to_Scarcity": hours_to_scarcity,
        "Exhaustion_Rate": exhaustion_rate,
        "Ramp_3hr": obs.Ramp_3hr,
        "RM_change_1hr": obs.RM_change_1hr,
        "RM_change_3hr": obs.RM_change_3hr,
        "Wind_change_3hr": obs.Wind_change_3hr,
        "Price_lag_1": obs.Price_lag_1,
        "Price_lag_3": obs.Price_lag_3,
        "Load_lag_1": obs.Load_lag_1,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "is_weekend": is_weekend,
        "day_of_year": day_of_year,
        "is_price_cap": obs.is_price_cap,
        "is_uri": obs.is_uri,
    }

    return pd.DataFrame([row])[FEATURE_COLS]


def _predict_single(obs: GridState) -> PredictionResponse:
    """Run all models on a single observation."""
    X = _compute_features(obs)

    # Global regressor (arcsinh space → sinh to invert)
    pred_log = models["regressor"].predict(X)[0]
    predicted_price = float(np.sinh(pred_log))

    # Spike classifier
    spike_prob = float(models["spike_clf"].predict_proba(X)[0, 1])
    is_spike = spike_prob > 0.5

    # Regime classifier
    regime_probs = models["regime_clf"].predict_proba(X)[0]
    regime_id = int(np.argmax(regime_probs))
    regime_names = {0: "Normal", 1: "Stressed", 2: "Scarcity"}

    # Ensemble prediction
    ens_pred = 0.0
    for rid, m in models["regime_regressors"].items():
        p = float(np.sinh(np.clip(m.predict(X)[0], -7, 10)))
        ens_pred += regime_probs[rid] * p

    return PredictionResponse(
        ObsTime=obs.ObsTime,
        predicted_price=round(predicted_price, 2),
        ensemble_price=round(ens_pred, 2),
        spike_probability=round(spike_prob, 4),
        is_spike_predicted=is_spike,
        predicted_regime=regime_names[regime_id],
        regime_probabilities={
            regime_names[i]: round(float(regime_probs[i]), 4) for i in range(3)
        },
    )


# ── API Endpoints ──────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "n_features": len(FEATURE_COLS),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(obs: GridState):
    """Predict electricity price for a single hour."""
    try:
        return _predict_single(obs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    """Predict prices for multiple hours."""
    if len(req.observations) > 500:
        raise HTTPException(status_code=400, detail="Max 500 observations per batch")
    try:
        results = [_predict_single(obs) for obs in req.observations]
        return BatchResponse(predictions=results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
