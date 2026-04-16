import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklift.metrics import uplift_at_k, qini_auc_score
# Removing sklift.viz due to compatibility issues with sklearn 1.4+

# Page Config
st.set_page_config(
    page_title="Uplift & Experimentation Optimizer",
    page_icon="💸",
    layout="wide"
)

# Custom Styling (Emerald & Slate Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    [data-testid="stAppViewContainer"] {
        background-color: #f8fafc;
    }
    
    .main-header {
        background: linear-gradient(135deg, #064e3b 0%, #059669 100%);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(5, 150, 105, 0.2);
    }
    
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #064e3b;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .stButton>button {
        background-color: #059669 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "criteo_sample.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_uplift_model.pkl")

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3201/3201558.png", width=80)
st.sidebar.title("Uplift Optimizer v1.0")
st.sidebar.markdown("---")

# Load Data & Model
@st.cache_resource
def load_assets():
    if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):
        return None, None
    df = pd.read_pickle(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    return df, model

df, model = load_assets()

if df is None:
    st.error("Assets not found. Please ensure data ingestion and modeling scripts have been run.")
    st.stop()

# Manual Qini Plotting Function (Fixed for Compatibility)
def custom_plot_qini(y_true, uplift, treat, name='Model', ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # Sort by uplift
    order = np.argsort(uplift)[::-1]
    y_true = np.array(y_true)[order]
    treat = np.array(treat)[order]
    
    # Cumulative gain
    treat_mask = (treat == 1); control_mask = (treat == 0)
    y_t = np.cumsum(y_true * treat_mask); y_c = np.cumsum(y_true * control_mask)
    n_t = np.cumsum(treat_mask); n_c = np.cumsum(control_mask)
    
    # Avoid div by zero
    n_t[n_t == 0] = 1; n_c[n_c == 0] = 1
    
    # Qini formula: y_t - y_c * (n_t / n_c)
    qini = y_t - y_c * (n_t / n_c)
    qini = np.concatenate([[0], qini])
    pop = np.arange(len(qini))
    
    # Random model
    random_qini = pop * (qini[-1] / (pop[-1] if pop[-1] > 0 else 1))
    
    ax.plot(pop, qini, label=name, color='#059669', linewidth=2)
    ax.plot(pop, random_qini, label='Random', color='#64748b', linestyle='--')
    ax.fill_between(pop, random_qini, qini, alpha=0.1, color='#059669')
    ax.set_xlabel("Population")
    ax.set_ylabel("Incremental Gain")
    ax.legend(facecolor='white')
    return ax

# Header
st.markdown("""
<div class='main-header'>
    <h1>🎯 Causal Uplift & Experimentation Optimizer</h1>
    <p>Leveraging Causal Inference to Predict Incrementality and Optimize ROI</p>
</div>
""", unsafe_allow_html=True)

# Main Dashboard
tab1, tab2, tab3, tab4 = st.tabs(["🎯 ROI Simulator", "📈 Performance Metrics", "🤖 Prediction Console", "🧠 Causal Explainability"])

with tab1:
    st.markdown("### 💰 Marketing Budget Optimization")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        cost_per_ad = st.slider("Cost per Advertisement ($)", 1, 50, 10)
        revenue_per_conv = st.slider("Revenue per Conversion ($)", 50, 500, 200)
        targeting_percent = st.slider("Population Targeted (%)", 5, 100, 30)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Simulation Logic
    # Predict uplift for all sampled data
    X = df.drop(['visit', 'treatment', 'exposure', 'conversion'], axis=1, errors='ignore')
    uplift_scores = model.predict(X)
    
    # Sort by uplift
    df_results = df.copy()
    df_results['uplift'] = uplift_scores
    df_results = df_results.sort_values(by='uplift', ascending=False)
    
    # Slice targeted group
    n_target = int(len(df_results) * (targeting_percent / 100))
    df_target = df_results.head(n_target)
    
    # Estimated Incremental Conversions
    # Roughly Sum(Uplift) * n_samples (since uplift is probability delta)
    est_inc_conv = df_target['uplift'].sum()
    total_cost = n_target * cost_per_ad
    total_rev = est_inc_conv * revenue_per_conv
    net_roi = total_rev - total_cost
    
    with col2:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='card'><p style='color:#64748b;font-size:0.8rem'>INC. CONVERSIONS</p><p class='metric-value'>{int(est_inc_conv)}</p></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='card'><p style='color:#64748b;font-size:0.8rem'>TOTAL COST</p><p class='metric-value'>${total_cost:,.0f}</p></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='card'><p style='color:#64748b;font-size:0.8rem'>NET PROFIT</p><p class='metric-value' style='color:#059669'>${net_roi:,.0f}</p></div>", unsafe_allow_html=True)
            
        # ROI Chart
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=['Cost', 'Revenue', 'Profit'], y=[total_cost, total_rev, net_roi], palette='Greens_d', ax=ax)
        ax.set_title("Estimated Economic Impact")
        st.pyplot(fig)

with tab2:
    st.markdown("### 📈 Uplift Validation Metrics")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("#### Qini Curve (Incrementality)")
        # Custom Qini plot using our manual implementation
        fig_q, ax_q = plt.subplots(figsize=(6, 4))
        custom_plot_qini(df['visit'], uplift_scores, df['treatment'], name='CROP Engine', ax=ax_q)
        st.pyplot(fig_q)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_b:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("#### Cumulative Lift")
        # DIY Lift Curve
        df_results['cum_uplift'] = df_results['uplift'].cumsum()
        fig_l, ax_l = plt.subplots()
        ax_l.plot(range(len(df_results)), df_results['cum_uplift'], color='#059669', linewidth=2)
        ax_l.set_xlabel("Population Percentile")
        ax_l.set_ylabel("Cumulative Incremental Gain")
        st.pyplot(fig_l)
        st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("### 🤖 Treatment Recommendation Console")
    st.info("Input visitor features to receive an AI-driven targeting decision.")
    
    c1, c2, c3 = st.columns(3)
    features = {}
    for i in range(12):
        col_idx = (i % 3)
        with [c1, c2, c3][col_idx]:
            features[f"f{i}"] = st.number_input(f"Feature f{i}", value=0.0)
            
    if st.button("🚀 Run Causal Inference"):
        in_df = pd.DataFrame([features])
        pred_uplift = model.predict(in_df)[0]
        
        st.markdown("---")
        if pred_uplift > 0.01:
            st.success(f"**RECOMMENDATION: TARGET** (Score: {pred_uplift:.4f})")
            st.write("This user is highly persuadable. Directing ads to them will likely yield a positive conversion.")
        elif pred_uplift < -0.005:
            st.error(f"**RECOMMENDATION: DO NOT TREAT** (Score: {pred_uplift:.4f})")
            st.write("CAUTION: This user is a 'Sleeping Dog'. An ad might actually decrease their likelihood of conversion.")
        else:
            st.warning(f"**RECOMMENDATION: HOLD** (Score: {pred_uplift:.4f})")
            st.write("User is likely a 'Lost Cause' or 'Sure Thing'. Targeting is not cost-effective.")

with tab4:
    st.markdown("### 🧠 Causal Explainability (Global View)")
    st.info("This section uses SHAP values to explain which features drive the 'Uplift' across the entire population.")
    
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("#### Uplift Feature Importance")
        shap_img = os.path.join(MODEL_DIR, "shap_plot.png")
        if os.path.exists(shap_img):
            st.image(shap_img, use_column_width=True)
            st.caption("Features are ranked by their average impact on the Predicted Uplift.")
        else:
            st.warning("SHAP explanation assets not found. Visit the research folder to generate them.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_r:
        st.markdown("<div class='card' style='height:100%'>", unsafe_allow_html=True)
        st.write("#### 🛡️ Interpreting the Causal Signal")
        st.write("""
        Unlike standard feature importance, **Causal SHAP** tells us which features are responsible for the *difference* in behavior between being treated and not treated.
        
        - **High Ranking Features**: These variables are the strongest indicators of whether someone is 'Persuadable'.
        - **Decision Logic**: If a user has specific values for these top features, the model identifies them as a high-value target for marketing spend.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.info("This project utilizes **Causal Inference** to solve the 'Counterfactual' problem in marketing.")
