# VeriClaim — AI-Powered Motor Insurance Fraud Detection

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-ff4b4b)

A production-grade, multi-modal AI system for detecting fraudulent motor insurance claims. Built for the Indian insurance market, VeriClaim combines computer vision, natural language processing, and machine learning to flag suspicious claims in real time.

---

## Demo

![VeriClaim Demo](https://i.imgur.com/placeholder.png)

> Upload a vehicle damage photo, fill in claim details, and get an instant fraud probability score with full explainability.

---

## What It Does

A fraud investigator submits a vehicle damage photo and claim details. VeriClaim runs three AI models in parallel and returns:

- **Fraud Probability** — 0 to 100% with LOW / MEDIUM / HIGH risk classification
- **Damage Severity** — minor, moderate, or severe from the vehicle photo
- **NLP Anomaly Score** — semantic analysis of the incident description
- **Top Risk Factors** — SHAP-based explanation of what drove the score
- **Recommendation** — actionable next step for the investigator

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                    │
│          Dark professional UI · localhost:8501           │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP POST multipart/form-data
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Backend                       │
│              REST API · localhost:8000                   │
│          POST /api/v1/predict/fraud                      │
└──────────┬──────────────┬──────────────┬────────────────┘
           │              │              │
           ▼              ▼              ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│ EfficientNet │  │  DistilBERT  │  │     XGBoost      │
│     B0       │  │  Sentence    │  │   + SMOTE +      │
│ Damage       │  │  Transformer │  │   SHAP           │
│ Classifier   │  │  Anomaly     │  │   Fraud          │
│ .pt model    │  │  Scorer      │  │   Classifier     │
│              │  │  (local)     │  │   .pkl model     │
└──────────────┘  └──────────────┘  └──────────────────┘
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch 2.1 + EfficientNet-B0 via timm |
| NLP | sentence-transformers + all-MiniLM-L6-v2 |
| ML Classifier | XGBoost + SMOTE (imbalanced-learn) |
| Explainability | SHAP TreeExplainer |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Training | Google Colab T4 GPU |

---

## Models

### 1. Damage Classifier (EfficientNet-B0)
- **Architecture:** EfficientNet-B0 pretrained on ImageNet, fine-tuned on car damage images
- **Training:** Google Colab T4 GPU · 20 epochs · early stopping · transfer learning
- **Dataset:** [anujms/car-damage-detection](https://www.kaggle.com/datasets/anujms/car-damage-detection) (~2300 images)
- **Classes:** minor · moderate · severe
- **Val Accuracy:** ~64.6%
- **Note:** Dataset has binary labels only (damaged/undamaged). Severity split is approximate. Model contributes damage signal to the fusion layer.

### 2. NLP Anomaly Scorer (DistilBERT)
- **Architecture:** Dual-layer scoring — semantic cosine similarity + weighted keyword matching
- **Model:** all-MiniLM-L6-v2 via sentence-transformers (~80MB, downloads on first run)
- **Patterns:** 15 known fraud patterns · 19 high-risk keywords
- **Output:** Anomaly score 0.0–1.0 (higher = more suspicious)
- **No training required** — uses pretrained embeddings with fraud-specific pattern matching

### 3. Fraud Classifier (XGBoost)
- **Architecture:** XGBoost with SMOTE oversampling for class imbalance
- **Training:** Google Colab CPU · 5-fold cross-validation
- **Dataset:** [shivamb/vehicle-claim-fraud-detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection) (15,420 rows)
- **CV AUC:** 0.8464
- **Features:** 31 features including vehicle details, policy info, incident characteristics
- **Explainability:** SHAP TreeExplainer for top risk factor attribution

---

## Project Structure

```
VeriClaim/
├── api/
│   ├── main.py                  # FastAPI app with lifespan model loading
│   ├── schemas.py               # Pydantic request/response models
│   └── routers/
│       └── claim.py             # POST /api/v1/predict/fraud endpoint
├── models/
│   ├── damage_classifier/
│   │   ├── model.py             # EfficientNet-B0 class definition
│   │   └── predict.py           # Inference: load_model(), predict_damage()
│   ├── claim_nlp/
│   │   ├── embed.py             # SentenceTransformer loading and embedding
│   │   ├── anomaly_score.py     # Dual-layer fraud scoring
│   │   └── fraud_patterns.json  # 15 patterns + 19 keywords
│   └── fraud_classifier/
│       ├── feature_eng.py       # Feature engineering pipeline
│       ├── predict.py           # Inference: load_fraud_model(), predict_fraud()
│       └── shap_explain.py      # SHAP explanation generation
├── notebooks/
│   ├── train_damage_classifier.ipynb   # Colab: EfficientNet training
│   └── train_fraud_classifier.ipynb    # Colab: XGBoost training
├── app.py                       # Streamlit frontend
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

> **Note:** `best_model.pt` and `xgb_fraud_model.pkl` are not committed to this repo.
> They are trained on Google Colab and placed manually. See Setup below.

---

## Local Setup

### Prerequisites
- Python 3.12
- Git
- Kaggle account (for dataset download on Colab)

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/VeriClaim.git
cd VeriClaim
```

### 2. Create virtual environment
```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add trained model files
The following files are NOT in the repo and must be obtained separately:

```
models/damage_classifier/best_model.pt      # ~16MB
models/fraud_classifier/xgb_fraud_model.pkl # ~1MB
```

To train them yourself:
- Open `notebooks/train_damage_classifier.ipynb` in Google Colab (T4 GPU)
- Open `notebooks/train_fraud_classifier.ipynb` in Google Colab (CPU)
- Run all cells — both notebooks download their datasets from Kaggle automatically
- Download the output files and place them in the paths above

### 5. Run the API
```bash
uvicorn api.main:app --port 8000
```

You should see:
```
[DL] Damage classifier loaded
[NLP] Loaded 15 fraud patterns and 19 keywords
[XGB] Fraud model loaded
[SHAP] Explainer loaded
All models loaded. API ready.
```

### 6. Run the Streamlit frontend
Open a second terminal:
```bash
streamlit run app.py
```

Go to `http://localhost:8501`

---

## API Reference

### POST /api/v1/predict/fraud

**Request:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| image | file | Vehicle damage photo (.jpg, .png) |
| claim_data | JSON string | Claim details (see schema below) |

**Response:**
```json
{
  "fraud_probability": 0.72,
  "fraud_flag": true,
  "risk_level": "HIGH",
  "recommendation": "Flag for manual investigation immediately.",
  "damage_severity": "severe",
  "damage_confidence": 0.61,
  "anomaly_score": 0.85,
  "triggered_keywords": ["total loss", "fire", "no witnesses"],
  "top_shap_factors": [
    {"feature": "Year", "impact": -1.46},
    {"feature": "BasePolicy", "impact": 0.78},
    {"feature": "AccidentArea", "impact": 0.49}
  ]
}
```

### GET /health
```json
{"status": "ok", "service": "vericlaim"}
```

Interactive API docs available at `http://localhost:8000/docs`

---

## Key Design Decisions

**Why Colab for training?**
Training deep learning models on a laptop CPU is impractical. Google Colab provides free T4 GPU access. This is standard practice — models are trained in the cloud and weights are downloaded for local inference. The trained `.pt` and `.pkl` files are the artifacts, not the training code.

**Why XGBoost over a neural network for fraud classification?**
XGBoost is interpretable via SHAP, fast at inference time, and performs extremely well on tabular data. Neural networks on tabular data rarely outperform gradient boosting and add unnecessary complexity. Explainability is critical in insurance fraud detection.

**Why sentence-transformers for NLP?**
The NLP module needs to catch semantic fraud signals, not just keyword matches. A claim saying "vehicle became inoperable due to thermal event" should score similarly to "car caught fire". Sentence embeddings handle this naturally without fine-tuning.

**Why Streamlit over React/Next.js?**
This is an internal fraud investigation tool, not a consumer product. Streamlit allows rapid prototyping of data-heavy interfaces in pure Python, keeping the stack consistent and reducing context switching.

---

## Limitations and Future Work

- **Damage severity accuracy (64.6%):** The training dataset has only binary labels (damaged/undamaged). Severity is approximated by splitting the damaged class randomly. A properly labeled severity dataset would significantly improve this module.
- **XGBoost dataset:** The training dataset does not have India-specific features. A real deployment would use proprietary claims data with geographic and temporal fraud patterns.
- **No authentication:** The API has no auth layer. Production deployment would require API key validation or JWT.
- **Single image input:** Real fraud detection would benefit from multiple photos and video evidence.

---

## License

MIT License — see LICENSE for details.

---

## Author

Built as a portfolio project demonstrating multi-modal AI system design, MLOps practices, and production API development.
