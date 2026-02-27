import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

_artifact = None

LABEL_ENCODERS = {}

FEATURE_COLS = [
    'Month', 'WeekOfMonth', 'DayOfWeek', 'Make', 'AccidentArea',
    'DayOfWeekClaimed', 'MonthClaimed', 'WeekOfMonthClaimed', 'Sex',
    'MaritalStatus', 'Age', 'Fault', 'PolicyType', 'VehicleCategory',
    'VehiclePrice', 'RepNumber', 'Deductible', 'DriverRating',
    'Days_Policy_Accident', 'Days_Policy_Claim', 'PastNumberOfClaims',
    'AgeOfVehicle', 'AgeOfPolicyHolder', 'PoliceReportFiled', 'WitnessPresent',
    'AgentType', 'NumberOfSuppliments', 'AddressChange_Claim', 'NumberOfCars',
    'Year', 'BasePolicy'
]

STRING_COLS = [
    'Month', 'DayOfWeek', 'Make', 'AccidentArea', 'DayOfWeekClaimed',
    'MonthClaimed', 'Sex', 'MaritalStatus', 'Fault', 'PolicyType',
    'VehicleCategory', 'VehiclePrice', 'Days_Policy_Accident', 'Days_Policy_Claim',
    'PastNumberOfClaims', 'AgeOfVehicle', 'AgeOfPolicyHolder', 'PoliceReportFiled',
    'WitnessPresent', 'AgentType', 'NumberOfSuppliments', 'AddressChange_Claim',
    'NumberOfCars', 'BasePolicy'
]


def load_fraud_model(path='models/fraud_classifier/xgb_fraud_model.pkl'):
    global _artifact
    _artifact = joblib.load(path)
    print(f'[XGB] Fraud model loaded from {path}')


def predict_fraud(claim_dict: dict, damage_pred: dict = None) -> dict:
    if _artifact is None:
        raise RuntimeError('Model not loaded. Call load_fraud_model() first.')

    model = _artifact['model']

    # Build row with only the 31 expected feature columns
    row = {}
    for col in FEATURE_COLS:
        row[col] = claim_dict.get(col, 0)

    df = pd.DataFrame([row])

    # Encode string columns to numeric
    le = LabelEncoder()
    for col in STRING_COLS:
        df[col] = le.fit_transform(df[col].astype(str))

    # Force everything to int
    df = df.astype(int)

    fraud_prob = float(model.predict_proba(df)[0][1])

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