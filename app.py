import streamlit as st
import pandas as pd
import joblib
import requests
from streamlit_lottie import st_lottie


def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5) # Added a timeout
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

lottie_anim = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_m9p9iz6j.json")

# --- THEN UPDATE THE DISPLAY SECTION ---
with col2:
    if lottie_anim:
        st_lottie(lottie_anim, height=200, key="coding")
    else:
        st.write("📈") # Fallback to an emoji if the animation fails

# --- CONFIG & STYLING ---
st.set_page_config(page_title="ChurnGuard AI", page_icon="🛡️", layout="wide")

def local_css():
    st.markdown("""
        <style>
        /* Fade-in Animation */
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .main { animation: fadeIn 2s; }
        
        /* Floating Button Effect */
        div.stButton > button:first-child {
            background-color: #6200ea;
            color: white;
            border-radius: 20px;
            transition: all 0.3s ease-in-out;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        div.stButton > button:hover {
            transform: scale(1.05);
            background-color: #3700b3;
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

# --- LOAD ASSETS ---
def load_lottieurl(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

lottie_anim = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_m9p9iz6j.json")

# --- UI LAYOUT ---
with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("🛡️ ChurnGuard AI")
        st.subheader("Predicting customer loyalty with high-precision Machine Learning.")
        st.write("Fill in the customer details on the right to see the churn risk.")
    with col2:
        st_lottie(lottie_anim, height=200, key="coding")

st.divider()

# --- INPUT FORM ---
try:
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')

    with st.expander("📝 Customer Details Input", expanded=True):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 18, 100, 30)
        gender = c2.selectbox("Gender", ["Male", "Female"])
        tenure = c3.slider("Tenure (Months)", 1, 120, 12)
        
        usage = c1.slider("Usage Frequency", 1, 30, 15)
        calls = c2.number_input("Support Calls", 0, 20, 2)
        delay = c3.number_input("Payment Delay (Days)", 0, 30, 0)
        
        sub_type = c1.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
        contract = c2.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
        spend = c3.number_input("Total Spend", 0, 10000, 500)

    # --- PREDICTION LOGIC ---
    if st.button("🔮 Predict Churn Risk"):
        # Encode inputs
        g_enc = encoders['Gender'].transform([gender])[0]
        s_enc = encoders['Subscription Type'].transform([sub_type])[0]
        c_enc = encoders['Contract Length'].transform([contract])[0]
        
        features = [[age, g_enc, tenure, usage, calls, delay, s_enc, c_enc, spend, 0]] # '0' for Last Interaction dummy
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled)
        prob = model.predict_proba(features_scaled)[0][1]

        if prediction[0] == 1:
            st.error(f"⚠️ **High Risk!** The customer is likely to churn. (Probability: {prob:.2%})")
            st.toast("Churn Alert!", icon='🚨')
        else:
            st.success(f"✅ **Safe!** This customer is likely to stay. (Probability of churn: {prob:.2%})")
            st.balloons()

except FileNotFoundError:
    st.warning("Please run the training script first to generate the model files!")
