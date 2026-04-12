"""
reports_chat.py — Report generation, report library, comparison, and chatbot agent.
"""

from __future__ import annotations
import json
import streamlit as st
from datetime import datetime
from data_loader import get_datasets

from db_ops import (save_report, load_all, delete_document, load_by_id,
                    COLL_REPORTS, COLL_FORECAST, COLL_RISK, COLL_INVENTORY,
                    COLL_SEASONAL, COLL_STOCKOUT, COLL_SUPPLIER, COLL_ROUTES)
from llm_agents import agent_generate_report, agent_compare_reports, agent_chat


# ════════════════════════════════════════════════════════════════════════════
#  REPORT GENERATION
# ════════════════════════════════════════════════════════════════════════════

def _collect_all_results() -> dict:
    """Gather latest results from every module for the report."""
    results = {}
    for coll, key in [(COLL_FORECAST, "demand_forecast"),
                      (COLL_RISK,     "risk_assessment"),
                      (COLL_INVENTORY,"inventory"),
                      (COLL_SEASONAL, "seasonality"),
                      (COLL_STOCKOUT, "stockout"),
                      (COLL_SUPPLIER, "supplier_risk"),
                      (COLL_ROUTES,   "route_optimization")]:
        docs = load_all(coll)
        if docs:
            results[key] = docs[:3]   # latest 3 per module
    return results


def render_reports():
    st.markdown("## 📋 AI Report Generation")
    st.caption("Generate comprehensive supply chain reports by synthesizing all module results.")

    from data_loader import get_datasets
    ds = get_datasets()

    with st.sidebar:
        st.markdown("### 📋 Report Config")
        org       = st.text_input("Organization",    placeholder="Your Company Ltd.")
        analyst   = st.text_input("Analyst Name",    placeholder="Your Name")
        period    = st.text_input("Reporting Period", placeholder="Q1 2024 – Q2 2024")
        rtype     = st.selectbox("Report Type", [
            "Comprehensive Supply Chain Intelligence",
            "Demand Forecast Focus",
            "Risk & Mitigation Report",
            "Inventory Optimization Report",
            "Supplier Performance Report",
            "Executive Dashboard Summary",
        ])
        title    = st.text_input("Report Title", value=f"SCM Report — {datetime.now().strftime('%b %Y')}")
        sections = st.multiselect("Include Sections", [
            "Executive Summary", "KPI Dashboard", "Demand Forecast",
            "Risk Analysis", "Inventory Status", "Supplier Performance",
            "Logistics & Routes", "Recommendations", "Action Plan"
        ], default=["Executive Summary","KPI Dashboard","Demand Forecast","Risk Analysis","Recommendations"])
        gen_btn  = st.button("🧠 Generate Report", type="primary", use_container_width=True)

    if not gen_btn:
        st.info("Configure report settings in the sidebar and click **🧠 Generate Report**.")
        _show_recent_reports()
        return

    with st.spinner("🔄 Collecting analysis results…"):
        all_results = _collect_all_results()

    if not all_results:
        st.warning("No analysis results found in the database. Run at least one analysis module first.")
        return

    with st.spinner("🧠 AI generating report… (this may take 20-30 seconds)"):
        report_md = agent_generate_report(title, rtype, all_results, org, period)

    # Save to DB
    doc_id = save_report(title, rtype, {
        "markdown": report_md,
        "org": org, "analyst": analyst, "period": period,
        "sections": sections,
        "source_results": {k: len(v) for k, v in all_results.items()},
    })

    # Display
    st.markdown(f"### {title}")
    st.caption(f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')} · {org} · {analyst}")

    # Download buttons
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇ Download Markdown", report_md, file_name=f"{title.replace(' ','_')}.md", mime="text/markdown")
    with c2:
        json_export = json.dumps(all_results, default=str, indent=2)
        st.download_button("⬇ Download JSON Data", json_export, file_name=f"{title.replace(' ','_')}_data.json", mime="application/json")

    st.divider()
    st.markdown(report_md)

    if doc_id:
        st.success(f"✅ Report saved to database (ID: `{doc_id[:8]}…`)")


# ════════════════════════════════════════════════════════════════════════════
#  REPORT LIBRARY
# ════════════════════════════════════════════════════════════════════════════

def render_report_library():
    st.markdown("## 📚 Report Library")
    st.caption("View, compare, and manage all saved reports.")

    reports = load_all(COLL_REPORTS)

    if not reports:
        st.info("No saved reports yet. Generate reports in the **AI Report** section.")
        return

    # ── Comparison mode ───────────────────────────────────────────────────────
    st.markdown("### 🔍 Compare Two Reports")
    with st.expander("Compare Reports", expanded=False):
        report_options = {f"{r.get('title','?')} ({r.get('created_at','')[:10]})": r["_id"] for r in reports}
        option_list = list(report_options.keys())

        if len(option_list) >= 2:
            c1, c2 = st.columns(2)
            sel_a = c1.selectbox("Report A", option_list, key="cmp_a")
            sel_b = c2.selectbox("Report B", option_list, index=1, key="cmp_b")
            cmp_query = st.text_area("Comparison Query",
                placeholder="e.g. How has supplier performance changed? What improved and what worsened?",
                height=80)
            if st.button("🧠 Compare & Analyze", type="primary") and cmp_query:
                r_a = load_by_id(COLL_REPORTS, report_options[sel_a])
                r_b = load_by_id(COLL_REPORTS, report_options[sel_b])
                if r_a and r_b:
                    with st.spinner("🧠 Comparing reports…"):
                        comparison = agent_compare_reports(r_a, r_b, cmp_query)
                    st.markdown("#### Comparison Analysis")
                    st.markdown(comparison)
        else:
            st.info("Generate at least 2 reports to enable comparison.")

    st.divider()
    st.markdown("### 📄 All Reports")

    for r in reports:
        with st.expander(f"📋 {r.get('title','Untitled')} — {r.get('report_type','?')} — {r.get('created_at','')[:10]}"):
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.caption(f"ID: `{r['_id'][:8]}…` | Type: {r.get('report_type','?')}")
            if c2.button("👁 View", key=f"view_{r['_id']}"):
                st.markdown(r.get("content", {}).get("markdown", "No content available."))
            if c3.button("🗑 Delete", key=f"del_{r['_id']}", type="secondary"):
                if delete_document(COLL_REPORTS, r["_id"]):
                    st.success("Deleted!")
                    st.rerun()


def _show_recent_reports():
    reports = load_all(COLL_REPORTS)
    if reports:
        st.markdown("#### Recent Reports")
        for r in reports[:5]:
            with st.expander(f"📋 {r.get('title','Untitled')} — {r.get('created_at','')[:10]}"):
                c1, c2 = st.columns([4, 1])
                c1.caption(f"Type: {r.get('report_type','?')} | ID: `{r['_id'][:8]}…`")
                if c2.button("🗑 Delete", key=f"home_del_{r['_id']}", type="secondary"):
                    if delete_document(COLL_REPORTS, r["_id"]):
                        st.rerun()
                md = r.get("content", {}).get("markdown", "")
                if md:
                    st.markdown(md[:1000] + ("…" if len(md) > 1000 else ""))


# ════════════════════════════════════════════════════════════════════════════
#  CHATBOT AGENT
# ════════════════════════════════════════════════════════════════════════════

def safe_json_load(response: str):
    try:
        return json.loads(response)
    except:
        # Try to extract JSON from text
        import re
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                return None
        return None

def render_ai_response(response: str):
    """Render AI response in clean UI format."""

    try:
        data = safe_json_load(response)

        if not data:
            st.markdown(response)
            return
    except:
        # fallback if it's plain text
        st.markdown(response)
        return

    # ── Main Answer ─────────────────────────
    if "response" in data:
        st.markdown("### 💡 Answer")
        st.markdown(data["response"])

    # ── Recommended Analysis ────────────────
    if "recommended_analysis" in data:
        rec = data["recommended_analysis"]

        st.markdown("### 📊 Recommended Analysis")

        if "module" in rec:
            st.markdown(f"**Module:** `{rec['module']}`")

        if "description" in rec:
            st.markdown(rec["description"])

        if "steps" in rec:
            st.markdown("#### 🔍 Steps")
            for i, step in enumerate(rec["steps"], 1):
                st.markdown(f"{i}. {step}")

        if "output_example" in rec:
            st.markdown("#### 📄 Example Output")
            st.json(rec["output_example"])

    # ── Next Steps ─────────────────────────
    if "next_steps" in data:
        st.markdown("### 🚀 Next Steps")
        st.info(data["next_steps"])


def render_chatbot():
    st.markdown("## 🤖 AI Supply Chain Chatbot")
    st.caption("Ask anything about your supply chain data. The agent has context from all loaded datasets and past analyses.")

    ds = get_datasets()

    # Build data context summary
    data_context = ""
    if ds:
        for dtype, df in ds.items():
            cols = list(df.columns[:6])
            data_context += f"\n{dtype}: {len(df)} rows, cols: {cols}"

    # DB context: latest results
    db_context = ""
    fc_docs = load_all(COLL_FORECAST)
    if fc_docs:
        db_context += f"\nLatest forecasts: {[d.get('product') for d in fc_docs[:3]]}"
    risk_docs = load_all(COLL_RISK)
    if risk_docs:
        db_context += f"\nRisk assessments: {len(risk_docs)} stored"

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content":
             "👋 Hello! I'm your **SKVision Supply Chain AI**.\n\n"
             "I have access to your uploaded datasets and all analysis results. Ask me anything:\n"
             "- *Which product has the highest stockout risk?*\n"
             "- *What's the demand forecast for Laptop next month?*\n"
             "- *Which supplier is least reliable?*\n"
             "- *What actions should I take to reduce inventory costs?*"}
        ]

    # Render chat messages
    chat_container = st.container(height=520)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
                if msg["role"] == "assistant":
                    render_ai_response(msg["content"])
                else:
                    st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Ask your supply chain question…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.spinner("🧠 Thinking…"):
            response = agent_chat(
                prompt,
                st.session_state.chat_history[:-1],
                data_context=data_context,
                db_context=db_context
            )

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    # Clear chat button
    if st.button("🗑 Clear Chat History", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()
