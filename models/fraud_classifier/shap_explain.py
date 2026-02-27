import shap
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

_explainer    = None
_feature_cols = None

STRING_COLS = [
    'Month', 'DayOfWeek', 'Make', 'AccidentArea', 'DayOfWeekClaimed',
    'MonthClaimed', 'Sex', 'MaritalStatus', 'Fault', 'PolicyType',
    'VehicleCategory', 'VehiclePrice', 'Days_Policy_Accident', 'Days_Policy_Claim',
    'PastNumberOfClaims', 'AgeOfVehicle', 'AgeOfPolicyHolder', 'PoliceReportFiled',
    'WitnessPresent', 'AgentType', 'NumberOfSuppliments', 'AddressChange_Claim',
    'NumberOfCars', 'BasePolicy'
]


def load_explainer(path='models/fraud_classifier/xgb_fraud_model.pkl'):
    global _explainer, _feature_cols
    artifact      = joblib.load(path)
    _explainer    = shap.TreeExplainer(artifact['model'])
    _feature_cols = artifact['feature_cols']
    print('[SHAP] Explainer loaded')


def explain(features_dict: dict) -> dict:
    if _explainer is None:
        raise RuntimeError('Explainer not loaded. Call load_explainer() first.')

    row = {col: features_dict.get(col, 0) for col in _feature_cols}
    df  = pd.DataFrame([row])

    le = LabelEncoder()
    for col in STRING_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    df = df.astype(int)

    shap_vals = _explainer.shap_values(df)[0]
    top_3 = sorted(
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