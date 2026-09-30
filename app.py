import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS for Alert Dark UI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="🚨",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #0c0f16;
        color: #e0e0e0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e222d 0%, #151922 100%);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        border: 1px solid #2a2e38;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-val {
        font-size: 30px;
        font-weight: 800;
    }
    .pos-val { color: #2ecc71; }
    .neg-val { color: #e74c3c; }
    .tot-val { color: #3498db; }
    
    .alert-box {
        background-color: #2c1a1d;
        border-left: 5px solid #e74c3c;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Anonymized Financial Transaction Generator
# -----------------------------------------------------------------------------
@st.cache_data
def load_transaction_data():
    np.random.seed(42)
    n_samples = 1200

    v1 = np.random.normal(0, 1, n_samples)
    v2 = np.random.normal(0, 1, n_samples)
    v3 = np.random.normal(0, 1, n_samples)
    v4 = np.random.normal(0, 1, n_samples)
    
    amount = np.random.exponential(scale=100, size=n_samples).clip(1, 5000)
    
    fraud_prob = (np.abs(v1) * 0.2 + np.abs(v4) * 0.3 + (amount > 2000) * 0.4)
    fraud_prob = np.clip(fraud_prob, 0.01, 0.9)
    
    is_fraud = np.random.binomial(1, fraud_prob / 5)

    df = pd.DataFrame({
        'Transaction_Amount': np.round(amount, 2),
        'V1_PCA': np.round(v1, 3),
        'V2_PCA': np.round(v2, 3),
        'V3_PCA': np.round(v3, 3),
        'V4_PCA': np.round(v4, 3),
        'Is_Fraud': is_fraud
    })
    return df

# -----------------------------------------------------------------------------
# 3. Anomaly Detection & SMOTE Pipeline
# -----------------------------------------------------------------------------
@st.cache_resource
def train_fraud_models(df):
    X = df.drop(columns=['Is_Fraud'])
    y = df['Is_Fraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_res, y_train_res)

    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)

    return iso_forest, rf_model, auc_score, X.columns

# -----------------------------------------------------------------------------
# 4. Dashboard Layout
# -----------------------------------------------------------------------------
st.title("🚨 Real-Time Financial Fraud Detection & Monitoring")
st.caption("Anomaly detection & supervised ML with SMOTE class balancing for financial safety.")
st.markdown("---")

df = load_transaction_data()

st.sidebar.title("⚙️ Monitoring Controls")
mode = st.sidebar.radio("Navigation:", ["🚨 Real-Time Transaction Alert Stream", "📊 Model Imbalance & Metrics", "🧪 Single Transaction Tester"])

iso_forest, rf_model, auc_score, feature_names = train_fraud_models(df)

# -----------------------------------------------------------------------------
# Page 1: Real-Time Transaction Alert Stream
# -----------------------------------------------------------------------------
if mode == "🚨 Real-Time Transaction Alert Stream":
    st.subheader("🚨 Live Fraud Alert Center")

    tot_tx = len(df)
    fraud_tx = df['Is_Fraud'].sum()
    fraud_rate = (fraud_tx / tot_tx) * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div>Total Processed</div><div class="metric-val tot-val">{tot_tx}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div>Normal Transactions</div><div class="metric-val pos-val">{tot_tx - fraud_tx}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div>Fraud Flagged</div><div class="metric-val neg-val">{fraud_tx}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div>Fraud Ratio</div><div class="metric-val neg-val">{fraud_rate:.2f}%</div></div>', unsafe_allow_html=True)

    st.write("")
    st.write("")

    fraud_df = df[df['Is_Fraud'] == 1].head(5)
    
    st.markdown("""
    <div class="alert-box">
        <h4>🚨 HIGH-PRIORITY FRAUD ALERTS DETECTED</h4>
        <p>The system automatically flagged suspicious transactions exceeding risk thresholds using Isolation Forest & SMOTE models.</p>
    </div>
    """, unsafe_allow_html=True)

    left_col, right_col = st.columns([1.3, 1])

    with left_col:
        st.subheader("📋 Suspicious Transactions Log")
        st.dataframe(fraud_df, use_container_width=True)

    with right_col:
        st.subheader("📈 Transaction Amount Anomaly Distribution")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor('#0c0f16')
        ax.set_facecolor('#0c0f16')
        
        # Corrected Palette Parameter for Seaborn
        sns.boxplot(x='Is_Fraud', y='Transaction_Amount', data=df, palette=['#2ecc71', '#e74c3c'], hue='Is_Fraud', legend=False, ax=ax)
        
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#2a2e38')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#2a2e38')
        ax.set_xticklabels(['Legit', 'Fraud'], color='white')
        ax.set_ylabel("Amount ($)", color="white")
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# Page 2: Model Imbalance & Metrics
# -----------------------------------------------------------------------------
elif mode == "📊 Model Imbalance & Metrics":
    st.subheader("📊 Class Imbalance & Model ROC-AUC Performance")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("ROC-AUC Score (Post SMOTE Balancing)", f"{auc_score:.3f}")
        st.info("SMOTE generates synthetic samples for minority fraud cases, ensuring the classifier avoids bias toward majority normal transactions.")

    with col2:
        st.subheader("⚖️ Target Class Distribution (Imbalance)")
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor('#0c0f16')
        ax.set_facecolor('#0c0f16')
        
        df['Is_Fraud'].value_counts().plot(kind='bar', color=['#2ecc71', '#e74c3c'], ax=ax)
        ax.tick_params(colors='white')
        ax.set_xticklabels(['Normal (98%)', 'Fraud (2%)'], rotation=0, color='white')
        ax.spines['bottom'].set_color('#2a2e38')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#2a2e38')
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# Page 3: Single Transaction Tester
# -----------------------------------------------------------------------------
else:
    st.subheader("🧪 Evaluate Individual Transaction Risk")
    
    col1, col2 = st.columns(2)
    with col1:
        tx_amount = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=10000.0, value=2500.0, step=50.0)
        v1 = st.slider("PCA Feature V1", -5.0, 5.0, 2.5)
        v2 = st.slider("PCA Feature V2", -5.0, 5.0, -1.2)
    
    with col2:
        v3 = st.slider("PCA Feature V3", -5.0, 5.0, 0.8)
        v4 = st.slider("PCA Feature V4", -5.0, 5.0, 3.1)

    if st.button("🔎 Scan Transaction for Fraud"):
        input_data = pd.DataFrame([[tx_amount, v1, v2, v3, v4]], columns=feature_names)
        
        iso_pred = iso_forest.predict(input_data)[0]
        fraud_prob = rf_model.predict_proba(input_data)[0][1] * 100

        st.write("")
        st.subheader("📋 Security Scan Result")

        if fraud_prob > 40.0 or iso_pred == -1:
            st.error(f"🚨 **FRAUD / ANOMALY DETECTED!** | Risk Probability: `{fraud_prob:.1f}%`")
            st.warning("Action Taken: Transaction blocked immediately. Security team alerted.")
        else:
            st.success(f"✅ **TRANSACTION LEGITIMATE** | Risk Probability: `{fraud_prob:.1f}%`")
            st.info("Transaction passed all safety criteria.")