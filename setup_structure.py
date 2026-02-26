# Run this as: python setup_structure.py
import os

dirs = [
    "data/raw/damage_images",
    "data/raw/claims_tabular",
    "data/processed",
    "models/damage_classifier",
    "models/claim_nlp",
    "models/fraud_classifier",
    "api/routers",
    "notebooks",
    "scripts",
    ".github/workflows",
]

files = [
    "models/damage_classifier/__init__.py",
    "models/damage_classifier/dataset.py",
    "models/damage_classifier/model.py",
    "models/damage_classifier/predict.py",
    "models/claim_nlp/__init__.py",
    "models/claim_nlp/embed.py",
    "models/claim_nlp/anomaly_score.py",
    "models/claim_nlp/fraud_patterns.json",
    "models/fraud_classifier/__init__.py",
    "models/fraud_classifier/feature_eng.py",
    "models/fraud_classifier/train.py",
    "models/fraud_classifier/shap_explain.py",
    "models/fraud_classifier/predict.py",
    "api/__init__.py",
    "api/main.py",
    "api/schemas.py",
    "api/routers/__init__.py",
    "api/routers/claim.py",
    "notebooks/train_damage_classifier.ipynb",
    "scripts/download_data.sh",
    "requirements.txt",
    "Dockerfile",
    "docker-compose.yml",
    ".env.example",
    "README.md",
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"Created dir:  {d}")

for f in files:
    if not os.path.exists(f):
        open(f, 'w').close()
        print(f"Created file: {f}")

print("\nStructure created successfully.")