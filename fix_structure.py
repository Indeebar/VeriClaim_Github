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
