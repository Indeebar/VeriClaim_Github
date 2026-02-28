import streamlit as st
from PIL import Image
import sys
from pathlib import Path

# ───────────────── PATH SETUP (CRITICAL FOR STREAMLIT CLOUD) ─────────────
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# ───────────────── SAFE IMPORTS (PREVENT CLOUD CRASH) ─────────────────
try:
    from models.damage_classifier.predict import predict_damage
    from models.fraud_classifier.predict import predict_fraud
    from models.claim_nlp.anomaly_score import score_text as get_anomaly_score
except Exception as e:
    st.error(f"Import Error: {e}")
    st.stop()

# ───────────────── PAGE CONFIG ─────────────────
st.set_page_config(
    page_title="VeriClaim",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 VeriClaim - AI Fraud Detection System")

# ───────────────── MODEL WARMUP (CORRECTED) ─────────────────
@st.cache_resource(show_spinner=True)
def warmup_models():
    """
    Proper warmup: explicitly load models instead of calling predict blindly.
    Prevents 'Model not loaded' RuntimeError on Streamlit Cloud.
    """
    try:
        # Import internal load functions if available
        from models.damage_classifier import predict as damage_module
        from models.fraud_classifier import predict as fraud_module

        # If your modules have load_model(), call them
        if hasattr(damage_module, "load_model"):
            damage_module.load_model()

        if hasattr(fraud_module, "load_model"):
            fraud_module.load_model()

        return True

    except Exception as e:
        print(f"Warmup Warning (non-fatal): {e}")
        return False


# Call once (cached)
warmup_models()

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
    if image is None:
        st.error("Please upload an image.")
        st.stop()

    with st.spinner("Running AI Fraud Analysis..."):

        # 1️⃣ DAMAGE MODEL (SAFE EXECUTION)
        try:
            damage_result = predict_damage(image)
            damage_severity = damage_result.get("severity", "unknown")
            damage_conf = damage_result.get("confidence", 0.0)
        except Exception as e:
            st.error(f"Damage model failed: {e}")
            st.stop()

        # 2️⃣ NLP ANOMALY (SAFE)
        try:
            anomaly_result = get_anomaly_score(description or "")
            anomaly_score = anomaly_result.get("anomaly_score", 0.0)
            triggered_keywords = anomaly_result.get("triggered_keywords", [])
        except Exception as e:
            st.warning(f"NLP module fallback used: {e}")
            anomaly_score = 0.0
            triggered_keywords = []

        # 3️⃣ CLAIM STRUCTURE
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

        # 4️⃣ FRAUD MODEL (SAFE)
        try:
            fraud_result = predict_fraud(claim_data)
            fraud_prob = fraud_result.get("fraud_probability", 0.0)
            risk = fraud_result.get("risk_level", "Unknown")
            recommendation = fraud_result.get("recommendation", "N/A")
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
        st.write("### Triggered Fraud Keywords")
        st.write(", ".join(triggered_keywords))

    if shap_factors:
        st.write("### Top Risk Factors (SHAP)")
        for factor in shap_factors:
            st.write(f"{factor['feature']} → Impact: {factor['impact']:.4f}")
