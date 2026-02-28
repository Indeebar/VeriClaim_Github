import streamlit as st
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# ───────────────── PATH FIX (CRITICAL FOR STREAMLIT CLOUD) ─────────────────
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ───────────────── SAFE IMPORTS ─────────────────
# Import AFTER path fix
from models.damage_classifier.predict import predict_damage, load_model as load_damage_model
from models.fraud_classifier.predict import predict_fraud
from models.claim_nlp.anomaly_score import score_text as get_anomaly_score

# ───────────────── PAGE CONFIG ─────────────────
st.set_page_config(
    page_title="VeriClaim",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 VeriClaim - AI Fraud Detection System")

# ───────────────── MODEL LOADING (CORRECT WAY) ─────────────────
@st.cache_resource(show_spinner=True)
def load_all_models():
    """
    Properly load models ONCE for Streamlit Cloud.
    Prevents:
    - Model not loaded error
    - Repeated torch loading
    - Cold start crashes
    """
    try:
        # 🔥 THIS is what your predict.py expects
        load_damage_model()
        return True
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return False

# Load models at startup
models_loaded = load_all_models()

# ───────────────── INPUT UI ─────────────────
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload Vehicle Damage Image",
        type=["jpg", "jpeg", "png"]
    )

    image = None
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=400)

with col2:
    st.subheader("Claim Details")

    make = st.selectbox(
        "Vehicle Make",
        ["Honda", "Toyota", "Ford", "BMW", "Hyundai", "Other"]
    )

    accident_area = st.selectbox("Accident Area", ["Urban", "Rural"])
    fault = st.selectbox("Fault", ["Policy Holder", "Third Party"])
    age = st.slider("Driver Age", 18, 80, 30)
    past_claims = st.selectbox(
        "Past Claims",
        ["none", "1", "2", "3 to 5", "more than 5"]
    )

description = st.text_area(
    "Incident Description",
    placeholder="Describe the incident..."
)

analyse = st.button("Analyse Claim", use_container_width=True)

# ───────────────── PREDICTION PIPELINE ─────────────────
if analyse:

    if not models_loaded:
        st.error("Models failed to load. Check deployment logs.")
        st.stop()

    if image is None:
        st.error("Please upload an image.")
        st.stop()

    with st.spinner("Running AI Fraud Analysis... (First run may take ~30s)"):

        # 1️⃣ DAMAGE PREDICTION (CV MODEL)
        try:
            damage_result = predict_damage(image)
            damage_severity = damage_result["severity"]
            damage_conf = damage_result["confidence"]
        except Exception as e:
            st.error(f"Damage model failed: {e}")
            st.stop()

        # 2️⃣ NLP ANOMALY SCORE (FIXED IMPORT)
        try:
            anomaly_result = get_anomaly_score(description)
            anomaly_score = anomaly_result["anomaly_score"]
            triggered_keywords = anomaly_result.get("triggered_keywords", [])
        except Exception as e:
            st.warning(f"NLP module issue: {e}")
            anomaly_score = 0.0
            triggered_keywords = []

        # 3️⃣ STRUCTURED CLAIM DATA
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

        # 4️⃣ FRAUD PREDICTION (XGBOOST + SHAP)
        try:
            fraud_result = predict_fraud(claim_data)
            fraud_prob = fraud_result["fraud_probability"]
            risk = fraud_result["risk_level"]
            recommendation = fraud_result["recommendation"]
            shap_factors = fraud_result.get("top_shap_factors", [])
        except Exception as e:
            st.error(f"Fraud model failed: {e}")
            st.stop()

    # ───────────────── OUTPUT ─────────────────
    st.success("Analysis Complete")

    colA, colB, colC = st.columns(3)

    colA.metric("Fraud Probability", f"{int(fraud_prob * 100)}%")
    colB.metric("Risk Level", risk)
    colC.metric("Damage Severity", damage_severity)

    st.write("### Recommendation")
    st.info(recommendation)

    st.write("### NLP Anomaly Score")
    st.progress(int(anomaly_score * 100))

    if triggered_keywords:
        st.write("### Triggered Keywords")
        st.write(", ".join(triggered_keywords))

    if shap_factors:
        st.write("### Top Risk Factors (SHAP)")
        for factor in shap_factors:
            st.write(f"{factor['feature']} → Impact: {factor['impact']:.4f}")
