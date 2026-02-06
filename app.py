import streamlit as st
import pandas as pd
import joblib
import requests
from streamlit_lottie import st_lottie

# --- 1. PAGE CONFIG & STYLES ---
st.set_page_config(page_title="ChurnGuard AI", page_icon="🛡️", layout="wide")

# Custom CSS for animations and styling
st.markdown("""
    <style>
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    .main { animation: fadeIn 1.5s; }
    .stButton>button {
        width: 100%;
        border-radius: 25px;
        height: 3em;
        background-color: #6200ea;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        border: 2px solid #6200ea;
        background-color: white;
        color: #6200ea;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HELPER FUNCTIONS ---
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# Load animation
lottie_url = "https://assets10.lottiefiles.com/packages/lf20_m9p9iz6j.json"
lottie_anim = load_lottieurl(lottie_url)

# --- 3. HEADER SECTION ---
# We define col1 and col2 here so they are available globally
header_col1, header_col2 = st.columns([2, 1])

with header_col1:
    st.title("🛡️ ChurnGuard AI")
    st.markdown("### High-Precision Customer Loyalty Prediction")
    st.write("Our ML model analyzes behavior patterns to predict if a customer is likely to leave your service.")

with header_col2:
    if lottie_anim:
        st_lottie(lottie_anim, height=200, key="header_anim")
    else:
        st.markdown("# 📊")

st.divider()

# --- 4. LOAD MODEL & PREPROCESSORS ---
@st.cache_resource # Keeps model in memory for speed
def load_ml_assets():
    try:
        model = joblib.load('churn_model.pkl')
        scaler = joblib.load('scaler.pkl')
        encoders = joblib.load('encoders.pkl')
        return model, scaler, encoders
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

model, scaler, encoders = load_ml_assets()

# --- 5. INPUT FORM ---
if model:
    with st.container():
        st.subheader("📝 Enter Customer Attributes")
        
        # Create input grid
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        row2_col1, row2_col2, row2_col3 = st.columns(3)
        
        # Row 1 Inputs
        age = row1_col1.number_input("Age", 18, 100, 30)
        gender = row1_col2.selectbox("Gender", ["Male", "Female"])
        tenure = row1_col3.slider("Tenure (Months)", 1, 120, 12)
        
        # Row 2 Inputs
        usage = row2_col1.slider("Usage Frequency", 1, 30, 15)
        calls = row2_col2.number_input("Support Calls", 0, 20, 2)
        delay = row2_col3.number_input("Payment Delay (Days)", 0, 30, 0)
        
        # Additional Inputs in an expander for cleanliness
        with st.expander("More Subscription Details"):
            sub_col1, sub_col2, sub_col3 = st.columns(3)
            sub_type = sub_col1.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
            contract = sub_col2.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
            spend = sub_col3.number_input("Total Spend ($)", 0, 10000, 500)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- 6. PREDICTION LOGIC ---
    if st.button("🔮 ANALYZE RISK"):
        try:
            # Match encoding logic from training
            g_enc = encoders['Gender'].transform([gender])[0]
            s_enc = encoders['Subscription Type'].transform([sub_type])[0]
            c_enc = encoders['Contract Length'].transform([contract])[0]
            
            # Prepare feature array (Ensure order matches training exactly)
            # Order: Age, Gender, Tenure, Usage, Calls, Delay, SubType, Contract, Spend, Last Interaction (dummy)
            features = [[age, g_enc, tenure, usage, calls, delay, s_enc, c_enc, spend, 0]]
            features_scaled = scaler.transform(features)
            
            # Predict
            prediction = model.predict(features_scaled)
            prob = model.predict_proba(features_scaled)[0][1]

            # Display Results
            st.divider()
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                if prediction[0] == 1:
                    st.error("### 🚨 HIGH RISK")
                    st.write(f"This customer is highly likely to churn.")
                else:
                    st.success("### ✅ LOW RISK")
                    st.write(f"This customer is likely to remain loyal.")
                    st.balloons()
            
            with res_col2:
                st.metric(label="Churn Probability", value=f"{prob:.1%}")
                st.progress(prob)

        except Exception as e:
            st.error(f"Prediction Error: {e}")
else:
    st.info("Waiting for model files to be uploaded to the repository...")
