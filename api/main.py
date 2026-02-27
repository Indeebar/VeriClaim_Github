from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routers.claim import router as claim_router

from models.damage_classifier.predict import load_model
from models.claim_nlp.embed import load_nlp_model
from models.fraud_classifier.predict import load_fraud_model
from models.fraud_classifier.shap_explain import load_explainer


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all models once at startup
    print('Loading models...')
    load_model('models/damage_classifier/best_model.pt')
    load_nlp_model('models/claim_nlp/fraud_patterns.json')
    load_fraud_model('models/fraud_classifier/xgb_fraud_model.pkl')
    load_explainer('models/fraud_classifier/xgb_fraud_model.pkl')
    print('All models loaded. API ready.')
    yield
    # Shutdown — nothing to clean up


app = FastAPI(
    title       = 'VeriClaim API',
    description = 'AI-powered motor insurance fraud detection for India',
    version     = '1.0.0',
    lifespan    = lifespan
)

app.include_router(
    claim_router,
    prefix = '/api/v1',
    tags   = ['Fraud Detection']
)


@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'vericlaim'}