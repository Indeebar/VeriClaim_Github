import json
import io
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image

from api.schemas import ClaimInput, FraudPredictionResponse, SHAPFactor
from models.damage_classifier.predict import predict_damage
from models.claim_nlp.anomaly_score import score_text
from models.fraud_classifier.predict import predict_fraud
from models.fraud_classifier.shap_explain import explain

router = APIRouter()


@router.post('/predict/fraud', response_model=FraudPredictionResponse)
async def predict_fraud_endpoint(
    image:      UploadFile = File(...),
    claim_data: str        = Form(...)
):
    # Parse claim JSON
    try:
        claim_dict = json.loads(claim_data)
        claim      = ClaimInput(**claim_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Invalid claim_data: {e}')

    # Step 1 — DL: damage severity from image
    try:
        img_bytes    = await image.read()
        pil_img      = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        damage_result = predict_damage(pil_img)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f'Image processing failed: {e}')

    # Step 2 — NLP: anomaly score from incident description
    nlp_result = {
        'anomaly_score':      0.0,
        'triggered_keywords': [],
        'top_fraud_pattern':  None
    }
    if claim.incident_description:
        try:
            nlp_result = score_text(claim.incident_description)
        except Exception:
            pass  # NLP failure is non-fatal, use defaults

    # Step 3 — XGBoost: fraud probability
    try:
        fraud_result = predict_fraud(
            claim_dict   = claim.dict(),
            damage_pred  = damage_result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Fraud model error: {e}')

    # Step 4 — SHAP explanation
    try:
        shap_result  = explain(claim.dict())
        shap_factors = [
            SHAPFactor(feature=f['feature'], impact=f['impact'])
            for f in shap_result['top_factors']
        ]
    except Exception:
        shap_factors = []

    return FraudPredictionResponse(
        fraud_probability  = fraud_result['fraud_probability'],
        fraud_flag         = fraud_result['fraud_flag'],
        risk_level         = fraud_result['risk_level'],
        recommendation     = fraud_result['recommendation'],
        damage_severity    = damage_result['severity'],
        damage_confidence  = damage_result['confidence'],
        anomaly_score      = nlp_result.get('anomaly_score'),
        triggered_keywords = nlp_result.get('triggered_keywords', []),
        top_shap_factors   = shap_factors
    )