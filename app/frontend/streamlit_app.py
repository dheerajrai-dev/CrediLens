import streamlit as st

# Configure page layout and metadata
st.set_page_config(
    page_title="CrediLens — Credit Risk & Decision Support",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

from app.frontend.styles import GLASS_CSS
from app.frontend.api_client import ApiClient
from app.frontend.views.prediction_view import render_prediction_view
from app.frontend.views.evaluation_view import render_evaluation_view
from app.frontend.views.explainability_view import render_explainability_view
from app.frontend.views.batch_view import render_batch_view

# Inject Glassmorphism design styling
st.markdown(GLASS_CSS, unsafe_allow_html=True)

# Initialize API client
@st.cache_resource
def get_client():
    return ApiClient()

client = get_client()

# Sidebar Navigation & System Information
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 16px 0;">
            <div style="font-size: 1.7rem; font-weight: 800; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                CrediLens
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.3; margin-top: 4px;">
                Explainable Credit Risk Prediction & Decision Support System
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    nav_selection = st.radio(
        "Application Navigation",
        options=[
            "🎯 Risk Prediction",
            "📊 Model Evaluation",
            "🔍 Explainability & SHAP",
            "📁 Batch Prediction"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Live System Status
    health = client.get_health()
    api_online = client.is_api_reachable()

    st.markdown("##### System Infrastructure")
    st.markdown(
        f"""
        <div class="glass-container" style="padding:14px; font-size:0.83rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#94a3b8;">FastAPI Service:</span>
                <span style="color:{('#34d399' if api_online else '#fbbf24')}; font-weight:600;">
                    {(f'Online (Port {client.port})' if api_online else 'In-Process Standalone')}
                </span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#94a3b8;">Final Model:</span>
                <span style="color:#60a5fa; font-weight:600;">Calibrated XGBoost</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#94a3b8;">Operating Cutoff:</span>
                <span style="color:#f8fafc; font-weight:600;">{health.get('threshold', 0.7632):.4f}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#94a3b8;">Dataset:</span>
                <span style="color:#e2e8f0;">Credit Risk (32.5k)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="color:#64748b; font-size:0.75rem; text-align:center; margin-top:18px;">
            Academic ML Portfolio &middot; Final Delivery Scope
        </div>
        """,
        unsafe_allow_html=True
    )

# Route to selected view
if nav_selection == "🎯 Risk Prediction":
    render_prediction_view(client)
elif nav_selection == "📊 Model Evaluation":
    render_evaluation_view(client)
elif nav_selection == "🔍 Explainability & SHAP":
    render_explainability_view(client)
elif nav_selection == "📁 Batch Prediction":
    render_batch_view(client)
