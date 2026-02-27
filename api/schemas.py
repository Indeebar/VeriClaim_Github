from pydantic import BaseModel
from typing import Optional, List


class ClaimInput(BaseModel):
    # Columns matching the actual trained XGBoost model
    Month:                  str   = 'Jan'
    WeekOfMonth:            int   = 1
    DayOfWeek:              str   = 'Monday'
    Make:                   str   = 'Honda'
    AccidentArea:           str   = 'Urban'
    DayOfWeekClaimed:       str   = 'Monday'
    MonthClaimed:           str   = 'Jan'
    WeekOfMonthClaimed:     int   = 1
    Sex:                    str   = 'Male'
    MaritalStatus:          str   = 'Single'
    Age:                    int   = 35
    Fault:                  str   = 'Policy Holder'
    PolicyType:             str   = 'Sport - Liability'
    VehicleCategory:        str   = 'Sport'
    VehiclePrice:           str   = '20000 to 29000'
    RepNumber:              int   = 1
    Deductible:             int   = 300
    DriverRating:           int   = 1
    Days_Policy_Accident:   str   = 'more than 30'
    Days_Policy_Claim:      str   = 'more than 30'
    PastNumberOfClaims:     str   = 'none'
    AgeOfVehicle:           str   = '3 years'
    AgeOfPolicyHolder:      str   = '26 to 30'
    PoliceReportFiled:      str   = 'No'
    WitnessPresent:         str   = 'No'
    AgentType:              str   = 'External'
    NumberOfSuppliments:    str   = 'none'
    AddressChange_Claim:    str   = '1 year'
    NumberOfCars:           str   = '1 vehicle'
    Year:                   int   = 2024
    BasePolicy:             str   = 'Liability'

    # Extra fields used by other modules but not XGBoost
    incident_description:   Optional[str] = None


class SHAPFactor(BaseModel):
    feature: str
    impact:  float


class FraudPredictionResponse(BaseModel):
    fraud_probability:   float
    fraud_flag:          bool
    risk_level:          str          # LOW / MEDIUM / HIGH
    recommendation:      str

    # DL module output
    damage_severity:     str
    damage_confidence:   float

    # NLP module output
    anomaly_score:       Optional[float]
    triggered_keywords:  Optional[List[str]]

    # XGBoost SHAP output
    top_shap_factors:    List[SHAPFactor]