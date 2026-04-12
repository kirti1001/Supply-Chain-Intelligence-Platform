"""
app.py — SKVision Supply Chain Intelligence Platform
Main entry point. Multi-page Streamlit app with full SCM pipeline.
"""

import streamlit as st
st.set_page_config(
    page_title="SKVision — Supply Chain Intelligence",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: Dark industrial theme matching SKVision ────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: #06080f; color: #c5d8f0; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: rgba(8,12,22,0.98) !important;
    border-right: 1px solid rgba(0,212,255,0.12);
  }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: rgba(12,16,32,0.9);
    border: 1px solid rgba(0,212,255,0.1);
    border-radius: 8px;
    padding: 12px 16px !important;
  }
  [data-testid="stMetricValue"] { color: #00d4ff; font-family: 'IBM Plex Mono'; }
  [data-testid="stMetricDelta"]  { font-size: 0.7rem; }

  /* Buttons */
  .stButton > button {
    background: rgba(0,212,255,0.1) !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    color: #00d4ff !important;
    font-family: 'IBM Plex Mono';
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    border-radius: 6px;
    transition: all 0.15s;
  }
  .stButton > button:hover { background: rgba(0,212,255,0.2) !important; }
  .stButton > button[kind="primary"] {
    background: rgba(0,212,255,0.85) !important;
    color: #06080f !important;
    font-weight: 700;
  }
  .stButton > button[kind="primary"]:hover {
    background: #00d4ff !important;
    box-shadow: 0 0 16px rgba(0,212,255,0.35);
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: rgba(12,16,32,0.8);
    border-radius: 8px;
    gap: 4px;
    padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono';
    font-size: 0.72rem;
    color: #3d5070;
    border-radius: 5px;
  }
  .stTabs [aria-selected="true"] {
    background: rgba(0,212,255,0.1) !important;
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff;
  }

  /* Dataframe */
  .stDataFrame { border: 1px solid rgba(0,212,255,0.1); border-radius: 6px; }

  /* Input */
  .stTextInput > div > input, .stTextArea > div > textarea,
  .stSelectbox > div > div { background: rgba(12,16,32,0.9) !important; color: #c5d8f0 !important; }

  /* Headers */
  h1, h2, h3 { color: #c5d8f0 !important; }
  h2 { border-bottom: 1px solid rgba(0,212,255,0.15); padding-bottom: 8px; }

  /* Alert / info */
  .stAlert { border-radius: 6px; }
  [data-testid="stExpanderDetails"] { background: rgba(8,12,22,0.6); }

  /* Nav items (styled radio as nav) */
  .nav-item { padding: 8px 14px; border-radius: 6px; cursor: pointer; margin: 2px 0; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.15); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Imports ──────────────────────────────────────────────────────────────────
from settings import render_settings_sidebar, get_credentials
from data_loader import (render_upload_section, auto_load_reference_files,
                         get_datasets, is_module_enabled, get_pipeline_state)


# ── Auto-load reference files on first run ───────────────────────────────────
if "datasets_loaded" not in st.session_state:
    auto_load_reference_files()
    st.session_state["datasets_loaded"] = True


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="padding: 16px 0 12px; border-bottom: 1px solid rgba(0,212,255,0.15); margin-bottom: 12px;">
      <div style="font-family: 'IBM Plex Mono'; font-size: 1.2rem; font-weight: 700; color: #fff;">
        SK<span style="color:#00d4ff">Vision</span>
      </div>
      <div style="font-size: 0.6rem; color: #3d5070; letter-spacing: 0.1em; text-transform: uppercase;">
        Agentic Supply Chain Intelligence
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    page = st.radio(
        "Navigation",
        options=[
            "🏠 Home & Upload",
            "📦 Demand Forecasting",
            "⚠️ Risk Assessment",
            "📊 Inventory Management",
            "🌊 Seasonality Analysis",
            "🚨 Stockout Prediction",
            "🏭 Supplier Risk",
            "🚛 Route Optimization",
            "📋 AI Report",
            "📚 Report Library",
            "🤖 AI Chatbot",
        ],
        label_visibility="collapsed"
    )

    st.divider()

    # Data + pipeline status
    ds = get_datasets()
    ps = get_pipeline_state()
    if ds:
        st.markdown("**📂 Loaded Datasets**")
        for dtype, df in ds.items():
            st.caption(f"✅ {dtype}: {len(df):,} rows")
        if ps:
            ena = ps.get("enabled_modules", [])
            blk = ps.get("blocked_modules", [])
            st.markdown("**🔬 Pipeline**")
            st.caption(f"✅ {len(ena)} modules ready")
            if blk:
                st.caption(f"⚠ {len(blk)} need more data")
    else:
        st.warning("⚠ No data loaded. Upload on Home page.")

    # Settings
    render_settings_sidebar()

    # Credentials quick status
    creds = get_credentials()
    st.divider()
    cols = st.columns(3)
    cols[0].metric("Groq",    "✅" if creds["groq_api_key"]  else "❌")
    cols[1].metric("MongoDB", "✅" if creds["mongo_uri"]      else "❌")
    cols[2].metric("Search",  "✅" if creds["tavily_api_key"] else "⚪")


# ── MAIN CONTENT ─────────────────────────────────────────────────────────────
if page == "🏠 Home & Upload":
    st.markdown("# 🔗 SKVision — Supply Chain Intelligence")
    st.caption("Multi-agent AI platform for demand forecasting, risk assessment, inventory optimization, and strategic reporting.")

    # Feature cards
    st.markdown("### 🧠 Platform Capabilities")
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.info("**📦 Demand Forecast**\nAI + statistical models with seasonality & event detection")
    r1c2.info("**⚠️ Risk Assessment**\nMulti-dimensional risk scoring across all supply chain layers")
    r1c3.info("**📊 Inventory Optim**\nEOQ, safety stock, reorder points using forecasted demand")
    r1c4.info("**🧊 3D Visualization**\nInteractive 3D Plotly charts for spatial supply chain analysis")

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.success("**🌊 Seasonality**\nDecompose demand into trend/seasonal/residual + web search")
    r2c2.success("**🚨 Stockout Pred**\nPredict stockout probability & optimal reorder timing")
    r2c3.success("**🏭 Supplier Risk**\nLead time, fill rate, concentration, quality risk scoring")
    r2c4.success("**🚛 Route Optim**\nDelay patterns, fuel efficiency & cost optimization")

    st.divider()
    render_upload_section()

    # Quick stats if data loaded
    if ds:
        st.divider()
        st.markdown("### 📊 Data Overview")
        for dtype, df in ds.items():
            with st.expander(f"📄 {dtype.upper()} Dataset ({len(df):,} rows × {df.shape[1]} cols)"):
                st.dataframe(df.head(5), use_container_width=True)
                num_cols = df.select_dtypes("number").columns.tolist()
                if num_cols:
                    st.caption(f"Numeric columns: {', '.join(num_cols[:8])}")

elif page == "📦 Demand Forecasting":
    if not is_module_enabled("demand_forecast"):
        st.warning("⚠️ **Demand Forecasting** requires `timestamp`, `product_name`, `units_sold`. "
                   "Run the pipeline on Home → Autonomous Upload to enable it.")
    else:
        from demand_forecasting import render_demand_forecasting
        render_demand_forecasting()

elif page == "⚠️ Risk Assessment":
    if not is_module_enabled("risk_assessment"):
        st.warning("⚠️ **Risk Assessment** requires `product_name`. Upload your dataset first.")
    else:
        from risk_assessment import render_risk_assessment
        render_risk_assessment()

elif page == "📊 Inventory Management":
    if not is_module_enabled("inventory_management"):
        st.warning("⚠️ **Inventory Management** requires `product_name` and `stock_units`.")
    else:
        from scm_modules import render_inventory_management
        render_inventory_management()

elif page == "🌊 Seasonality Analysis":
    if not is_module_enabled("seasonality"):
        st.warning("⚠️ **Seasonality** requires `timestamp`, `product_name`, `units_sold`.")
    else:
        from scm_modules import render_seasonality
        render_seasonality()

elif page == "🚨 Stockout Prediction":
    if not is_module_enabled("stockout_prediction"):
        st.warning("⚠️ **Stockout Prediction** requires `product_name` and `stock_units`.")
    else:
        from scm_modules import render_stockout
        render_stockout()

elif page == "🏭 Supplier Risk":
    if not is_module_enabled("supplier_risk"):
        st.warning("⚠️ **Supplier Risk** requires `supplier_name`. "
                   "Provide a supplier dataset or use Autonomous mode.")
    else:
        from scm_modules import render_supplier_risk
        render_supplier_risk()

elif page == "🚛 Route Optimization":
    if not is_module_enabled("route_optimization"):
        st.warning("⚠️ **Route Optimization** requires `route_type`. "
                   "Provide transport data or use Autonomous mode.")
    else:
        from scm_modules import render_route_optimization
        render_route_optimization()

elif page == "📋 AI Report":
    from reports_chat import render_reports
    render_reports()

elif page == "📚 Report Library":
    from reports_chat import render_report_library
    render_report_library()

elif page == "🤖 AI Chatbot":
    from reports_chat import render_chatbot
    render_chatbot()