import streamlit as st
import sys
from pathlib import Path
from PIL import Image

# ── Path fix (critical for Streamlit Cloud) ───────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = 'VeriClaim',
    page_icon  = '🔍',
    layout     = 'wide',
    initial_sidebar_state = 'collapsed'
)

# ── Dark professional theme ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    background-color: #080c14;
    color: #c8d8e8;
    font-family: 'Rajdhani', sans-serif;
}
#MainMenu, footer, header {visibility: hidden;}
.block-container { padding: 2rem 3rem; max-width: 1400px; }

.vericlaim-header { border-bottom: 1px solid #1e3a5f; padding-bottom: 1.2rem; margin-bottom: 2rem; }
.vericlaim-title { font-family: 'Share Tech Mono', monospace; font-size: 2.4rem; color: #00d4ff; letter-spacing: 0.15em; margin: 0; }
.vericlaim-subtitle { color: #4a7a9b; font-size: 0.95rem; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 0.3rem; }

.section-label { font-family: 'Share Tech Mono', monospace; font-size: 0.75rem; color: #00d4ff; letter-spacing: 0.2em; text-transform: uppercase; border-left: 3px solid #00d4ff; padding-left: 0.6rem; margin-bottom: 1rem; margin-top: 1.5rem; }

.metric-card { background: #0d1824; border: 1px solid #1e3a5f; border-radius: 4px; padding: 1.2rem 1.5rem; margin-bottom: 1rem; }
.metric-label { font-size: 0.75rem; color: #4a7a9b; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.3rem; }
.metric-value { font-family: 'Share Tech Mono', monospace; font-size: 2rem; font-weight: bold; }

.risk-HIGH   { color: #ff4444; }
.risk-MEDIUM { color: #ffaa00; }
.risk-LOW    { color: #00cc66; }

.prob-bar-container { background: #0d1824; border: 1px solid #1e3a5f; border-radius: 2px; height: 8px; width: 100%; margin: 0.5rem 0 1rem 0; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 2px; }

.keyword-tag { display: inline-block; background: #1a0a0a; border: 1px solid #ff4444; color: #ff6666; font-family: 'Share Tech Mono', monospace; font-size: 0.75rem; padding: 0.2rem 0.6rem; border-radius: 2px; margin: 0.2rem; }

.shap-row { display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid #0d1824; font-size: 0.9rem; }
.shap-positive { color: #ff6644; }
.shap-negative { color: #00cc66; }

.stButton > button { background: linear-gradient(135deg, #003d5c, #006494); color: #00d4ff; border: 1px solid #00d4ff; border-radius: 2px; font-family: 'Share Tech Mono', monospace; font-size: 1rem; letter-spacing: 0.15em; padding: 0.7rem 2.5rem; width: 100%; }
.stButton > button:hover { background: linear-gradient(135deg, #006494, #0088cc); box-shadow: 0 0 20px rgba(0,212,255,0.3); }

.scan-line { height: 1px; background: linear-gradient(90deg, transparent, #00d4ff, transparent); margin: 1.5rem 0; opacity: 0.4; }

.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #00cc66; margin-right: 0.5rem; animation: pulse 2s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="vericlaim-header">
    <div class="vericlaim-title">⬡ VERICLAIM</div>
    <div class="vericlaim-subtitle">AI-Powered Motor Insurance Fraud Detection System</div>
</div>
""", unsafe_allow_html=True)

# ── Model loading (cached — runs once) ───────────────────────────────────────
@st.cache_resource(show_spinner='Loading AI models...')
def load_all_models():
    from models.damage_classifier.predict import load_model as load_damage
    from models.claim_nlp.embed import load_nlp_model
    from models.fraud_classifier.predict import load_fraud_model
    from models.fraud_classifier.shap_explain import load_explainer

    base = Path(__file__).resolve().parent
    load_damage(str(base / 'models/damage_classifier/best_model.pt'))
    load_nlp_model(str(base / 'models/claim_nlp/fraud_patterns.json'))
    load_fraud_model(str(base / 'models/fraud_classifier/xgb_fraud_model.pkl'))
    load_explainer(str(base / 'models/fraud_classifier/xgb_fraud_model.pkl'))
    return True

models_loaded = load_all_models()

# ── Layout ────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap='large')

# ══ LEFT PANEL ════════════════════════════════════════════════════════════════
with left_col:

    st.markdown('<div class="section-label">01 / Vehicle Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        'Upload damage photo',
        type=['jpg', 'jpeg', 'png'],
        label_visibility='collapsed'
    )
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, use_column_width=True)

    st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">02 / Claim Details</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        make             = st.selectbox('Vehicle Make', ['Honda', 'Toyota', 'Ford', 'Chevrolet', 'BMW', 'Mercedes', 'Nissan', 'Hyundai', 'Volkswagen', 'Other'])
        accident_area    = st.selectbox('Accident Area', ['Urban', 'Rural'])
        fault            = st.selectbox('Fault', ['Policy Holder', 'Third Party'])
        vehicle_category = st.selectbox('Vehicle Category', ['Sport', 'Sedan', 'SUV', 'Utility', 'Hatchback'])
        age              = st.number_input('Driver Age', min_value=18, max_value=80, value=35)
        deductible       = st.selectbox('Deductible', [300, 400, 500, 700])

    with col_b:
        police_report  = st.selectbox('Police Report Filed', ['No', 'Yes'])
        witness        = st.selectbox('Witness Present', ['No', 'Yes'])
        agent_type     = st.selectbox('Agent Type', ['External', 'Internal'])
        base_policy    = st.selectbox('Base Policy', ['Liability', 'Collision', 'All Perils'])
        driver_rating  = st.selectbox('Driver Rating', [1, 2, 3, 4])
        past_claims    = st.selectbox('Past Claims', ['none', '1', '2', '3 to 5', 'more than 5'])

    st.markdown('<div class="section-label">03 / Incident Description</div>', unsafe_allow_html=True)
    description = st.text_area(
        'Describe what happened',
        placeholder='e.g. Vehicle caught fire at 3am on a remote highway, no witnesses present...',
        height=100,
        label_visibility='collapsed'
    )

    st.markdown('<br>', unsafe_allow_html=True)
    analyse_btn = st.button('⬡  ANALYSE CLAIM', use_container_width=True)

# ══ RIGHT PANEL ═══════════════════════════════════════════════════════════════
with right_col:

    if not analyse_btn:
        st.markdown("""
        <div style="height:500px; display:flex; flex-direction:column; align-items:center;
                    justify-content:center; border:1px dashed #1e3a5f; border-radius:4px;
                    color:#1e3a5f; font-family:'Share Tech Mono',monospace; font-size:0.85rem;
                    letter-spacing:0.1em;">
            <div style="font-size:3rem; margin-bottom:1rem; opacity:0.3;">⬡</div>
            <div>AWAITING CLAIM SUBMISSION</div>
            <div style="margin-top:0.5rem; font-size:0.7rem; opacity:0.5;">
                UPLOAD IMAGE AND FILL FORM TO BEGIN ANALYSIS
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        if not uploaded_file:
            st.error('Please upload a vehicle damage image before analysing.')
            st.stop()

        with st.spinner('Analysing claim...'):
            try:
                from models.damage_classifier.predict import predict_damage
                from models.claim_nlp.anomaly_score import score_text
                from models.fraud_classifier.predict import predict_fraud
                from models.fraud_classifier.shap_explain import explain

                # Step 1 — Damage severity from image
                pil_img       = Image.open(uploaded_file).convert('RGB')
                damage_result = predict_damage(pil_img)

                # Step 2 — NLP anomaly score
                nlp_result = {'anomaly_score': 0.0, 'triggered_keywords': []}
                if description:
                    try:
                        nlp_result = score_text(description)
                    except Exception:
                        pass

                # Step 3 — Build claim dict with 31 XGBoost features
                claim_dict = {
                    'Month':                'Jan',
                    'WeekOfMonth':          1,
                    'DayOfWeek':            'Monday',
                    'Make':                 make,
                    'AccidentArea':         accident_area,
                    'DayOfWeekClaimed':     'Monday',
                    'MonthClaimed':         'Jan',
                    'WeekOfMonthClaimed':   1,
                    'Sex':                  'Male',
                    'MaritalStatus':        'Single',
                    'Age':                  int(age),
                    'Fault':                fault,
                    'PolicyType':           f'{vehicle_category} - Liability',
                    'VehicleCategory':      vehicle_category,
                    'VehiclePrice':         'more than 69000',
                    'RepNumber':            1,
                    'Deductible':           int(deductible),
                    'DriverRating':         int(driver_rating),
                    'Days_Policy_Accident': 'more than 30',
                    'Days_Policy_Claim':    'more than 30',
                    'PastNumberOfClaims':   past_claims,
                    'AgeOfVehicle':         '3 years',
                    'AgeOfPolicyHolder':    '26 to 30',
                    'PoliceReportFiled':    police_report,
                    'WitnessPresent':       witness,
                    'AgentType':            agent_type,
                    'NumberOfSuppliments':  'none',
                    'AddressChange_Claim':  '1 year',
                    'NumberOfCars':         '1 vehicle',
                    'Year':                 2024,
                    'BasePolicy':           base_policy,
                }

                # Step 4 — Fraud prediction
                fraud_result = predict_fraud(claim_dict, damage_pred=damage_result)

                # Step 5 — SHAP explanation
                try:
                    shap_result  = explain(claim_dict)
                    shap_factors = shap_result.get('top_factors', [])
                except Exception:
                    shap_factors = []

            except Exception as e:
                st.error(f'Analysis failed: {e}')
                st.stop()

        # ── Results ──────────────────────────────────────────────────────────
        prob      = fraud_result['fraud_probability']
        risk      = fraud_result['risk_level']
        prob_pct  = int(prob * 100)
        bar_color = '#ff4444' if risk == 'HIGH' else '#ffaa00' if risk == 'MEDIUM' else '#00cc66'

        st.markdown('<div class="section-label">Analysis Results</div>', unsafe_allow_html=True)

        # Fraud probability
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Fraud Probability</div>
            <div class="metric-value risk-{risk}">{prob_pct}%</div>
            <div class="prob-bar-container">
                <div class="prob-bar-fill" style="width:{prob_pct}%; background:{bar_color};"></div>
            </div>
            <div style="font-size:0.8rem; color:#4a7a9b;">
                Risk Level: <span class="risk-{risk}" style="font-weight:bold;">{risk}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Recommendation
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Recommendation</div>
            <div style="font-size:1rem; color:#c8d8e8; margin-top:0.3rem;">
                {fraud_result['recommendation']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Damage + NLP side by side
        mc1, mc2 = st.columns(2)
        with mc1:
            sev       = damage_result['severity']
            sev_color = '#ff4444' if sev == 'severe' else '#ffaa00' if sev == 'moderate' else '#00cc66'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Damage Severity</div>
                <div class="metric-value" style="color:{sev_color}; font-size:1.4rem;">{sev.upper()}</div>
                <div style="font-size:0.75rem; color:#4a7a9b;">Confidence: {int(damage_result['confidence']*100)}%</div>
            </div>
            """, unsafe_allow_html=True)

        with mc2:
            nlp_pct   = int((nlp_result.get('anomaly_score', 0) or 0) * 100)
            nlp_color = '#ff4444' if nlp_pct > 60 else '#ffaa00' if nlp_pct > 30 else '#00cc66'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">NLP Anomaly Score</div>
                <div class="metric-value" style="color:{nlp_color}; font-size:1.4rem;">{nlp_pct}%</div>
                <div style="font-size:0.75rem; color:#4a7a9b;">Description analysis</div>
            </div>
            """, unsafe_allow_html=True)

        # Triggered keywords
        keywords = nlp_result.get('triggered_keywords', [])
        if keywords:
            st.markdown('<div class="section-label">Flagged Keywords</div>', unsafe_allow_html=True)
            kw_html = ''.join([f'<span class="keyword-tag">⚠ {kw}</span>' for kw in keywords])
            st.markdown(f'<div style="padding:0.5rem 0">{kw_html}</div>', unsafe_allow_html=True)

        # SHAP factors
        if shap_factors:
            st.markdown('<div class="section-label">Top Risk Factors</div>', unsafe_allow_html=True)
            shap_html = '<div class="metric-card">'
            for factor in shap_factors:
                impact     = factor['impact']
                impact_cls = 'shap-positive' if impact > 0 else 'shap-negative'
                arrow      = '▲' if impact > 0 else '▼'
                shap_html += f'''
                <div class="shap-row">
                    <div style="color:#c8d8e8;">{factor['feature']}</div>
                    <div class="{impact_cls}">{arrow} {abs(impact):.4f}</div>
                </div>'''
            shap_html += '</div>'
            st.markdown(shap_html, unsafe_allow_html=True)

        # Status footer
        st.markdown("""
        <div style="margin-top:1rem; font-family:'Share Tech Mono',monospace;
                    font-size:0.7rem; color:#1e3a5f; text-align:right;">
            <span class="status-dot"></span>
            ANALYSIS COMPLETE · ALL MODULES ACTIVE
        </div>
        """, unsafe_allow_html=True)
