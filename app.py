import streamlit as st
import json
from PIL import Image
import numpy as np
import sys
from pathlib import Path

# ───────────────── PATH SETUP (VERY IMPORTANT FOR STREAMLIT CLOUD) ─────────────
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

# Import your internal modules
from models.damage_classifier.predict import predict_damage
from models.fraud_classifier.predict import predict_fraud
from models.claim_nlp.anomaly_score import get_anomaly_score

# ───────────────── PAGE CONFIG ─────────────────
st.set_page_config(
    page_title="VeriClaim",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 VeriClaim - AI Fraud Detection System")

# ───────────────── MODEL LOADING (CACHED = FAST) ─────────────────
@st.cache_resource
def load_models():
    # Your predict modules should internally load:
    # - best_model.pt
    # - xgb_fraud_model.pkl
    return True  # placeholder since your modules handle loading

load_models()

# ───────────────── INPUT UI ─────────────────
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload Vehicle Damage Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=400)

with col2:
    st.subheader("Claim Details")

    make = st.selectbox("Vehicle Make", [
        "Honda", "Toyota", "Ford", "BMW", "Hyundai", "Other"
    ])

    accident_area = st.selectbox("Accident Area", ["Urban", "Rural"])
    fault = st.selectbox("Fault", ["Policy Holder", "Third Party"])
    age = st.slider("Driver Age", 18, 80, 30)
    past_claims = st.selectbox(
        "Past Claims", ["none", "1", "2", "3 to 5", "more than 5"]
    )

description = st.text_area(
    "Incident Description",
    placeholder="Describe the incident..."
)

analyse = st.button("Analyse Claim", use_container_width=True)

# ───────────────── PREDICTION PIPELINE ─────────────────
if analyse:
    if not uploaded_file:
        st.error("Please upload an image.")
        st.stop()

    with st.spinner("Running AI Fraud Analysis..."):

        # 1️⃣ DAMAGE PREDICTION (PyTorch Model)
        damage_result = predict_damage(image)

        damage_severity = damage_result["severity"]
        damage_conf = damage_result["confidence"]

        # 2️⃣ NLP ANOMALY SCORE
        anomaly_score = get_anomaly_score(description)

        # 3️⃣ STRUCTURED CLAIM DATA (same as your backend)
        claim_data = {
            "Make": make,
            "AccidentArea": accident_area,
            "Fault": fault,
            "Age": age,
            "PastNumberOfClaims": past_claims,
            "incident_description": description,
            "damage_severity": damage_severity,
            "damage_confidence": damage_conf,
            "anomaly_score": anomaly_score,
        }

        # 4️⃣ FRAUD PREDICTION (XGBoost + SHAP)
        fraud_result = predict_fraud(claim_data)

        fraud_prob = fraud_result["fraud_probability"]
        risk = fraud_result["risk_level"]
        recommendation = fraud_result["recommendation"]
        shap_factors = fraud_result.get("top_shap_factors", [])

    # ───────────────── OUTPUT ─────────────────
    st.success("Analysis Complete")

    colA, colB, colC = st.columns(3)

    colA.metric("Fraud Probability", f"{int(fraud_prob*100)}%")
    colB.metric("Risk Level", risk)
    colC.metric("Damage Severity", damage_severity)

    st.write("### Recommendation")
    st.info(recommendation)

    st.write("### NLP Anomaly Score")
    st.progress(int(anomaly_score * 100))

    if shap_factors:
        st.write("### Top Risk Factors (SHAP)")
        for factor in shap_factors:
            st.write(f"{factor['feature']} → Impact: {factor['impact']:.4f}")
