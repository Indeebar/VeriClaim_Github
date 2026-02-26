import joblib
import pandas as pd
from models.fraud_classifier.feature_eng import engineer_features

_artifact = None


def load_fraud_model(path='models/fraud_classifier/xgb_fraud_model.pkl'):
    global _artifact
    _artifact = joblib.load(path)
    print(f'[XGB] Fraud model loaded from {path}')


def predict_fraud(claim_dict: dict, damage_pred: dict = None) -> dict:
    if _artifact is None:
        raise RuntimeError(
            'Model not loaded. Call load_fraud_model() before predict_fraud().'
        )

    df = pd.DataFrame([claim_dict])
    damage_preds = [damage_pred] if damage_pred else None
    df = engineer_features(df, damage_preds=damage_preds)

    feat_cols = _artifact['feature_cols']
    available = [c for c in feat_cols if c in df.columns]
    X = df[available].fillna(0)

    fraud_prob = float(_artifact['model'].predict_proba(X)[0][1])

    if fraud_prob >= 0.7:
        risk = 'HIGH'
        recommendation = 'Flag for manual investigation immediately.'
    elif fraud_prob >= 0.4:
        risk = 'MEDIUM'
        recommendation = 'Assign to senior adjuster for review.'
    else:
        risk = 'LOW'
        recommendation = 'Proceed with standard claim processing.'

    return {
        'fraud_probability': round(fraud_prob, 4),
        'fraud_flag':        fraud_prob >= 0.5,
        'risk_level':        risk,
        'recommendation':    recommendation
    }