import pandas as pd
import numpy as np

TIER1_CITIES = {
    'mumbai', 'delhi', 'bangalore', 'bengaluru', 'hyderabad',
    'chennai', 'kolkata', 'pune', 'ahmedabad'
}
TIER2_CITIES = {
    'jaipur', 'lucknow', 'surat', 'nagpur', 'indore',
    'bhopal', 'visakhapatnam', 'patna', 'vadodara', 'ludhiana'
}
FESTIVAL_MONTHS = {10, 11}  # October and November — Diwali / Navratri


def get_city_tier(city: str) -> int:
    c = str(city).lower().strip()
    if c in TIER1_CITIES:
        return 1
    if c in TIER2_CITIES:
        return 2
    return 3


def get_hour_bin(hour) -> int:
    try:
        h = int(hour)
    except (ValueError, TypeError):
        return 2
    if 2 <= h <= 4:
        return 0   # highest risk window
    if h >= 22 or h <= 1:
        return 1   # night
    return 2       # day


def engineer_features(df: pd.DataFrame, damage_preds: list = None) -> pd.DataFrame:
    df = df.copy()

    if 'incident_city' in df.columns:
        df['city_tier'] = df['incident_city'].apply(get_city_tier)

    if 'incident_date' in df.columns:
        df['_incident_dt'] = pd.to_datetime(df['incident_date'], errors='coerce')
        df['incident_month'] = df['_incident_dt'].dt.month
        df['is_festival_season'] = df['incident_month'].isin(FESTIVAL_MONTHS).astype(int)
        df.drop(columns=['_incident_dt'], inplace=True)

    if 'policy_bind_date' in df.columns and 'incident_date' in df.columns:
        bind_dt     = pd.to_datetime(df['policy_bind_date'], errors='coerce')
        incident_dt = pd.to_datetime(df['incident_date'],    errors='coerce')
        df['policy_age_days'] = (incident_dt - bind_dt).dt.days.fillna(365)

    if 'incident_hour_of_day' in df.columns:
        df['incident_hour_bin'] = df['incident_hour_of_day'].apply(get_hour_bin)

    if 'total_claim_amount' in df.columns and 'vehicle_claim' in df.columns:
        df['claim_to_value_ratio'] = (
            df['total_claim_amount'] / (df['vehicle_claim'] + 1)
        ).clip(0, 10)

    # Fuse DL damage prediction as a feature
    if damage_preds is not None:
        df['damage_severity_idx'] = [p['severity_idx'] for p in damage_preds]
        df['damage_confidence']   = [p['confidence']   for p in damage_preds]
        if 'total_claim_amount' in df.columns:
            max_claim = df['total_claim_amount'].max()
            if max_claim > 0:
                claim_norm = df['total_claim_amount'] / max_claim
                df['damage_claim_mismatch'] = (
                    (1 - df['damage_severity_idx'] / 2) * claim_norm
                ).clip(0, 1)

    return df