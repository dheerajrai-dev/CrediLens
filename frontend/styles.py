# CrediLens Streamlit Glassmorphism Design System

GLASS_CSS = """
<style>
/* Global App Background & Base Typography */
.stApp {
    background: radial-gradient(circle at 10% 20%, #0d1527 0%, #0b0f19 50%, #070a12 100%) !important;
    color: #e2e8f0 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Glassmorphism Card Container */
.glass-container {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 22px 26px;
    margin-bottom: 22px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.35);
    transition: all 0.25s ease-in-out;
}
.glass-container:hover {
    border-color: rgba(255, 255, 255, 0.2);
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.45);
}

/* Glass Card Highlight (Final Model / Champion) */
.glass-highlight {
    background: linear-gradient(135deg, rgba(30, 58, 138, 0.35) 0%, rgba(17, 24, 39, 0.6) 100%);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(96, 165, 250, 0.45);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 8px 32px 0 rgba(37, 99, 235, 0.25);
    margin-bottom: 22px;
}

/* Badges */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-low {
    background: rgba(16, 185, 129, 0.18);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
}
.badge-mod {
    background: rgba(245, 158, 11, 0.18);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.35);
}
.badge-high {
    background: rgba(249, 115, 22, 0.18);
    color: #fb923c;
    border: 1px solid rgba(249, 115, 22, 0.35);
}
.badge-critical {
    background: rgba(239, 68, 68, 0.22);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.45);
}
.badge-champion {
    background: rgba(99, 102, 241, 0.22);
    color: #818cf8;
    border: 1px solid rgba(99, 102, 241, 0.45);
}

/* Metric Pill */
.metric-pill {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
}
.metric-pill .label {
    font-size: 0.78rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.metric-pill .value {
    font-size: 1.55rem;
    font-weight: 700;
    color: #f8fafc;
}

/* Confusion Matrix Tiles */
.cm-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}
.cm-tile {
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.cm-tn { background: rgba(16, 185, 129, 0.12); border-color: rgba(16, 185, 129, 0.25); }
.cm-fp { background: rgba(245, 158, 11, 0.12); border-color: rgba(245, 158, 11, 0.25); }
.cm-fn { background: rgba(239, 68, 68, 0.15); border-color: rgba(239, 68, 68, 0.3); }
.cm-tp { background: rgba(59, 130, 246, 0.14); border-color: rgba(59, 130, 246, 0.3); }

/* Sidebar Glass Styling */
section[data-testid="stSidebar"] {
    background: rgba(11, 15, 25, 0.85) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Custom Headers */
.gradient-title {
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 4px;
}
.gradient-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 24px;
}

/* Streamlit Native Overrides */
div[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    color: #f1f5f9 !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    padding: 6px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: rgba(59, 130, 246, 0.25) !important;
    color: #60a5fa !important;
    border: 1px solid rgba(96, 165, 250, 0.3) !important;
}
</style>
"""

def get_risk_badge(band: str) -> str:
    band_lower = band.lower()
    if "low" in band_lower:
        return f'<span class="badge badge-low">{band} Risk</span>'
    elif "mod" in band_lower:
        return f'<span class="badge badge-mod">{band} Risk</span>'
    elif "high" in band_lower:
        return f'<span class="badge badge-high">{band} Risk</span>'
    else:
        return f'<span class="badge badge-critical">{band} Risk</span>'
