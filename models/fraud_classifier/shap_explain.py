import shap
import joblib
import pandas as pd

_explainer   = None
_feature_cols = None


def load_explainer(path='models/fraud_classifier/xgb_fraud_model.pkl'):
    global _explainer, _feature_cols
    artifact      = joblib.load(path)
    _explainer    = shap.TreeExplainer(artifact['model'])
    _feature_cols = artifact['feature_cols']
    print(f'[SHAP] Explainer loaded')


def explain(features_dict: dict) -> dict:
    if _explainer is None:
        raise RuntimeError('Explainer not loaded. Call load_explainer() first.')

    row      = pd.DataFrame([features_dict])[_feature_cols].fillna(0)
    shap_vals = _explainer.shap_values(row)[0]
    top_3    = sorted(
        zip(_feature_cols, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    return {
        'top_factors': [
            {'feature': k, 'impact': round(float(v), 4)}
            for k, v in top_3
        ]
    }