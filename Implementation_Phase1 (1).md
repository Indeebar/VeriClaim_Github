# VeriClaim — Implementation Phase 1 (Final)
# Project Setup and Repository Structure

---

## What You Are Building

VeriClaim is an AI-powered motor insurance fraud detection system for India.

A fraud investigator uploads a photo of the damaged vehicle and submits claim details.
The system returns a fraud probability score with an explanation.

How it works:
- EfficientNet-B0 looks at the car photo and outputs damage severity
- DistilBERT reads the incident description and outputs an anomaly score
- XGBoost combines all signals into a final fraud probability with SHAP explanation

---

## Where Everything Happens

Training on Google Colab (you do this separately, not through the agent):
- Train EfficientNet-B0 on car damage images  ->  download best_model.pt from Drive
- Train XGBoost on insurance claims data      ->  download xgb_fraud_model.pkl from Drive
- Place both files manually into the project folders shown below

Local project (what the agent builds):
- model.py     : EfficientNet architecture definition so best_model.pt can be loaded
- predict.py   : loads the .pt or .pkl file and runs inference
- FastAPI app  : receives image + claim data, runs all 3 models, returns fraud score

There is NO train.py in the local project.
There is NO local dataset download.
All data lives on Colab and Google Drive only.

Model formats:
- PyTorch  ->  .pt   (NOT .h5, that is TensorFlow and will not work here)
- XGBoost  ->  .pkl

---

## Datasets (Download These on Colab Only)

Car damage images:
  https://www.kaggle.com/datasets/anujms/car-damage-detection
  Used in: notebooks/train_damage_classifier.ipynb

Insurance claims tabular data:
  https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection
  Used in: notebooks/train_fraud_classifier.ipynb

Do not download these locally. They are only needed inside Colab.

---

## Phase 1 Goal

By end of Phase 1:
- All folders and the correct local files exist inside VeriClaim
- Virtual environment created and all dependencies installed
- Git initialized and pushed to GitHub
- .gitignore blocks model files and data permanently
- No datasets downloaded locally
- No model training done yet

---

## Step 1.1 — Fix the Directory Structure

The agent has already run setup_structure.py but it created wrong files.
Run this corrected script now. It will delete the wrong files and create the right ones.

Save as fix_structure.py inside VeriClaim and run it:

```python
import os

# Delete these files — they should NOT exist in the local project
wrong_files = [
    "models/damage_classifier/dataset.py",
    "models/fraud_classifier/train.py",
]
for f in wrong_files:
    if os.path.exists(f):
        os.remove(f)
        print(f"Deleted:      {f}")
    else:
        print(f"Already gone: {f}")

# These are the only files that should exist in models/
correct_files = [
    "models/damage_classifier/__init__.py",
    "models/damage_classifier/model.py",
    "models/damage_classifier/predict.py",
    "models/claim_nlp/__init__.py",
    "models/claim_nlp/embed.py",
    "models/claim_nlp/anomaly_score.py",
    "models/claim_nlp/fraud_patterns.json",
    "models/fraud_classifier/__init__.py",
    "models/fraud_classifier/feature_eng.py",
    "models/fraud_classifier/predict.py",
    "models/fraud_classifier/shap_explain.py",
    "api/__init__.py",
    "api/main.py",
    "api/schemas.py",
    "api/routers/__init__.py",
    "api/routers/claim.py",
    "notebooks/train_damage_classifier.ipynb",
    "notebooks/train_fraud_classifier.ipynb",
    "requirements.txt",
    "Dockerfile",
    "docker-compose.yml",
    ".env.example",
    "README.md",
]

for f in correct_files:
    d = os.path.dirname(f)
    if d:
        os.makedirs(d, exist_ok=True)
    if not os.path.exists(f):
        open(f, "w").close()
        print(f"Created:      {f}")
    else:
        print(f"Exists:       {f}")

print("\nStructure fixed.")
```

Run it:
    python fix_structure.py

After it runs, confirm these DO NOT exist:
    models/damage_classifier/dataset.py
    models/fraud_classifier/train.py

And confirm these DO exist:
    models/damage_classifier/model.py
    models/damage_classifier/predict.py
    models/fraud_classifier/predict.py
    models/fraud_classifier/shap_explain.py
    notebooks/train_damage_classifier.ipynb
    notebooks/train_fraud_classifier.ipynb

---

## Step 1.2 — Update .gitignore

Replace the contents of .gitignore with exactly this:

```
data/
*.pt
*.pkl
*.h5
venv/
.env
__pycache__/
*.pyc
*.egg-info/
.DS_Store
*.pem
.aws/
kaggle.json
.ipynb_checkpoints/
```

---

## Step 1.3 — Verify Dependencies

Check what is installed:
    pip list | findstr -i "torch timm xgboost fastapi sentence"

Install anything missing from this list:
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
    pip install timm==0.9.12
    pip install sentence-transformers==2.3.1
    pip install xgboost==2.0.3
    pip install scikit-learn==1.3.2
    pip install imbalanced-learn==0.11.0
    pip install shap==0.44.0
    pip install pandas==2.1.4
    pip install numpy==1.26.3
    pip install Pillow==10.2.0
    pip install fastapi==0.109.0
    pip install uvicorn==0.27.0
    pip install python-multipart==0.0.6
    pip install pydantic==2.5.3
    pip install joblib==1.3.2
    pip install python-dotenv==1.0.0
    pip install requests==2.31.0

After everything is installed:
    pip freeze > requirements.txt

Verify:
    python -c "import torch; import timm; import xgboost; import fastapi; from sentence_transformers import SentenceTransformer; print('All OK')"

Must print: All OK

---

## Step 1.4 — Create .env.example

Create .env.example in VeriClaim root:

    S3_BUCKET=vericlaim-models
    AWS_ACCESS_KEY_ID=your_key_here
    AWS_SECRET_ACCESS_KEY=your_secret_here

---

## Step 1.5 — Commit and Push

    git add .
    git status

Confirm git status does NOT show: data/, *.pt, *.pkl, kaggle.json, dataset.py, train.py
If any appear, fix .gitignore first.

    git commit -m "Phase 1 complete: correct project structure and dependencies"
    git push origin main

---

## Phase 1 Verification Checklist

    [ ] fix_structure.py ran without errors
    [ ] models/damage_classifier/dataset.py does NOT exist
    [ ] models/fraud_classifier/train.py does NOT exist
    [ ] models/damage_classifier/model.py exists
    [ ] models/damage_classifier/predict.py exists
    [ ] models/fraud_classifier/predict.py exists
    [ ] models/fraud_classifier/shap_explain.py exists
    [ ] notebooks/train_damage_classifier.ipynb exists
    [ ] notebooks/train_fraud_classifier.ipynb exists
    [ ] (venv) shows in terminal prompt
    [ ] python imports check prints All OK
    [ ] requirements.txt is not empty
    [ ] .gitignore contains: data/, *.pt, *.pkl, kaggle.json
    [ ] git status shows no data/ or model files
    [ ] GitHub VeriClaim repo shows latest commit

---

## What Phase 2 Covers

Phase 2 has two parts running in parallel:

YOU do on Google Colab:
    Open notebooks/train_damage_classifier.ipynb in Colab
    Set runtime to T4 GPU
    Run all cells — it downloads dataset inside Colab, trains the model, saves to Drive
    Download best_model.pt from Google Drive
    Place it at: models/damage_classifier/best_model.pt

AGENT does locally:
    Write models/damage_classifier/model.py   (EfficientNet-B0 class definition)
    Write models/damage_classifier/predict.py (loads best_model.pt, runs inference)

Phase 2 ends when best_model.pt is placed and predict.py runs without errors.
