import streamlit as st
import requests
import json
from PIL import Image
import io

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

/* Base */
html, body, [class*="css"] {
    background-color: #080c14;
    color: #c8d8e8;
    font-family: 'Rajdhani', sans-serif;
}

/* Hide Streamlit branding */
#MainMenu, footer, header {visibility: hidden;}

/* Main container */
.block-container {
    padding: 2rem 3rem;
    max-width: 1400px;
}

/* Header */
.vericlaim-header {
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 1.2rem;
    margin-bottom: 2rem;
}
.vericlaim-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.4rem;
    color: #00d4ff;
    letter-spacing: 0.15em;
    margin: 0;
}
.vericlaim-subtitle {
    color: #4a7a9b;
    font-size: 0.95rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* Section labels */
.section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #00d4ff;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    border-left: 3px solid #00d4ff;
    padding-left: 0.6rem;
    margin-bottom: 1rem;
    margin-top: 1.5rem;
}

/* Cards */
.metric-card {
    background: #0d1824;
    border: 1px solid #1e3a5f;
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.metric-label {
    font-size: 0.75rem;
    color: #4a7a9b;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2rem;
    font-weight: bold;
}

/* Risk colors */
.risk-HIGH   { color: #ff4444; }
.risk-MEDIUM { color: #ffaa00; }
.risk-LOW    { color: #00cc66; }

/* Probability bar */
.prob-bar-container {
    background: #0d1824;
    border: 1px solid #1e3a5f;
    border-radius: 2px;
    height: 8px;
    width: 100%;
    margin: 0.5rem 0 1rem 0;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.8s ease;
}

/* Keywords */
.keyword-tag {
    display: inline-block;
    background: #1a0a0a;
    border: 1px solid #ff4444;
    color: #ff6666;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    margin: 0.2rem;
}

/* SHAP factors */
.shap-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid #0d1824;
    font-size: 0.9rem;
}
.shap-positive { color: #ff6644; }
.shap-negative { color: #00cc66; }

/* Analyse button */
.stButton > button {
    background: linear-gradient(135deg, #003d5c, #006494);
    color: #00d4ff;
    border: 1px solid #00d4ff;
    border-radius: 2px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1rem;
    letter-spacing: 0.15em;
    padding: 0.7rem 2.5rem;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #006494, #0088cc);
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
}

/* Inputs */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > div {
    background: #0d1824 !important;
    border: 1px solid #1e3a5f !important;
    color: #c8d8e8 !important;
    border-radius: 2px !important;
    font-family: 'Rajdhani', sans-serif !important;
}

/* Divider */
.scan-line {
    height: 1px;
    background: linear-gradient(90deg, transparent, #00d4ff, transparent);
    margin: 1.5rem 0;
    opacity: 0.4;
}

/* Status dot */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #00cc66;
    margin-right: 0.5rem;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}
</style>
""", unsafe_allow_html=True)

# ── Config ─────────────────────────────────────────────────────────────────────
API_URL = 'http://localhost:8000'

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="vericlaim-header">
    <div class="vericlaim-title">⬡ VERICLAIM</div>
    <div class="vericlaim-subtitle">AI-Powered Motor Insurance Fraud Detection System</div>
</div>
""", unsafe_allow_html=True)

# ── Layout ─────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap='large')

# ══ LEFT PANEL — Input ════════════════════════════════════════════════════════
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
        make = st.selectbox('Vehicle Make', [
            'Honda', 'Toyota', 'Ford', 'Chevrolet', 'BMW',
            'Mercedes', 'Nissan', 'Hyundai', 'Volkswagen', 'Other'
        ])
        accident_area = st.selectbox('Accident Area', ['Urban', 'Rural'])
        fault = st.selectbox('Fault', ['Policy Holder', 'Third Party'])
        vehicle_category = st.selectbox('Vehicle Category', [
            'Sport', 'Sedan', 'SUV', 'Utility', 'Hatchback'
        ])
        age = st.number_input('Driver Age', min_value=18, max_value=80, value=35)
        deductible = st.selectbox('Deductible', [300, 400, 500, 700])

    with col_b:
        police_report = st.selectbox('Police Report Filed', ['No', 'Yes'])
        witness = st.selectbox('Witness Present', ['No', 'Yes'])
        agent_type = st.selectbox('Agent Type', ['External', 'Internal'])
        base_policy = st.selectbox('Base Policy', [
            'Liability', 'Collision', 'All Perils'
        ])
        driver_rating = st.selectbox('Driver Rating', [1, 2, 3, 4])
        past_claims = st.selectbox('Past Claims', [
            'none', '1', '2', '3 to 5', 'more than 5'
        ])

    st.markdown('<div class="section-label">03 / Incident Description</div>', unsafe_allow_html=True)
    description = st.text_area(
        'Describe what happened',
        placeholder='e.g. Vehicle caught fire at 3am on a remote highway, no witnesses present...',
        height=100,
        label_visibility='collapsed'
    )

    st.markdown('<br>', unsafe_allow_html=True)
    analyse_btn = st.button('⬡  ANALYSE CLAIM', use_container_width=True)


# ══ RIGHT PANEL — Results ═════════════════════════════════════════════════════
with right_col:

    if not analyse_btn:
        st.markdown("""
        <div style="
            height: 500px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border: 1px dashed #1e3a5f;
            border-radius: 4px;
            color: #1e3a5f;
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.85rem;
            letter-spacing: 0.1em;
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.3;">⬡</div>
            <div>AWAITING CLAIM SUBMISSION</div>
            <div style="margin-top: 0.5rem; font-size: 0.7rem; opacity: 0.5;">
                UPLOAD IMAGE AND FILL FORM TO BEGIN ANALYSIS
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        if not uploaded_file:
            st.error('Please upload a vehicle damage image before analysing.')
        else:
            with st.spinner('Analysing claim...'):
                try:
                    claim_data = {
                        'Month':                str(st.session_state.get('month', 'Jan')),
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
                        'incident_description': description
                    }

                    uploaded_file.seek(0)
                    response = requests.post(
                        f'{API_URL}/api/v1/predict/fraud',
                        files={'image': ('image.jpg', uploaded_file.read(), 'image/jpeg')},
                        data={'claim_data': json.dumps(claim_data)},
                        timeout=30
                    )

                    if response.status_code != 200:
                        st.error(f'API Error {response.status_code}: {response.text}')
                        st.stop()

                    r = response.json()

                except requests.exceptions.ConnectionError:
                    st.error(
                        'Cannot connect to VeriClaim API. '
                        'Make sure uvicorn is running on port 8000.'
                    )
                    st.stop()
                except Exception as e:
                    st.error(f'Error: {e}')
                    st.stop()

            # ── Results ──────────────────────────────────────────────────────
            prob       = r['fraud_probability']
            risk       = r['risk_level']
            prob_pct   = int(prob * 100)
            bar_color  = '#ff4444' if risk == 'HIGH' else '#ffaa00' if risk == 'MEDIUM' else '#00cc66'

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
                    {r['recommendation']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Two column metrics
            mc1, mc2 = st.columns(2)
            with mc1:
                dmg_color = '#ff4444' if r['damage_severity'] == 'severe' else '#ffaa00' if r['damage_severity'] == 'moderate' else '#00cc66'
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Damage Severity</div>
                    <div class="metric-value" style="color:{dmg_color}; font-size:1.4rem;">
                        {r['damage_severity'].upper()}
                    </div>
                    <div style="font-size:0.75rem; color:#4a7a9b;">
                        Confidence: {int(r['damage_confidence']*100)}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with mc2:
                nlp_score = r.get('anomaly_score', 0) or 0
                nlp_pct   = int(nlp_score * 100)
                nlp_color = '#ff4444' if nlp_pct > 60 else '#ffaa00' if nlp_pct > 30 else '#00cc66'
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">NLP Anomaly Score</div>
                    <div class="metric-value" style="color:{nlp_color}; font-size:1.4rem;">
                        {nlp_pct}%
                    </div>
                    <div style="font-size:0.75rem; color:#4a7a9b;">
                        Description analysis
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Triggered keywords
            keywords = r.get('triggered_keywords', [])
            if keywords:
                st.markdown('<div class="section-label">Flagged Keywords</div>', unsafe_allow_html=True)
                kw_html = ''.join([f'<span class="keyword-tag">⚠ {kw}</span>' for kw in keywords])
                st.markdown(f'<div style="padding:0.5rem 0">{kw_html}</div>', unsafe_allow_html=True)

            # SHAP factors
            shap_factors = r.get('top_shap_factors', [])
            if shap_factors:
                st.markdown('<div class="section-label">Top Risk Factors</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                for i, factor in enumerate(shap_factors):
                    impact     = factor['impact']
                    impact_cls = 'shap-positive' if impact > 0 else 'shap-negative'
                    arrow      = '▲' if impact > 0 else '▼'
                    st.markdown(f"""
                    <div class="shap-row">
                        <div style="color:#c8d8e8;">{factor['feature']}</div>
                        <div class="{impact_cls}">{arrow} {abs(impact):.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Status footer
            st.markdown(f"""
            <div style="margin-top:1rem; font-family:'Share Tech Mono',monospace;
                        font-size:0.7rem; color:#1e3a5f; text-align:right;">
                <span class="status-dot"></span>
                ANALYSIS COMPLETE · ALL MODULES ACTIVE
            </div>
            """, unsafe_allow_html=True)