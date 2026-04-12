"""
risk_assessment.py — Multi-dimensional supply chain risk analysis.
Computes risk scores from demand, inventory, supplier, and transport data.
Saves structured JSON to MongoDB.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

from db_ops import save_risk, load_all, COLL_RISK
from llm_agents import agent_risk_analysis, web_search
from charts import (risk_gauge, risk_3d_scatter, heatmap_correlation,
                    bar_comparison, pie_distribution, demand_trend_line, anomaly_scatter)


# ─── Risk computation helpers ────────────────────────────────────────────────
def _demand_risk(df: pd.DataFrame) -> dict:
    """Compute demand volatility and trend risk."""
    value_col = "units_sold" if "units_sold" in df.columns else \
                [c for c in df.columns if "unit" in c or "sale" in c][0]
    date_col = "timestamp" if "timestamp" in df.columns else \
               df.select_dtypes("datetime").columns[0]

    risks = {}
    if "product_name" in df.columns:
        for prod, grp in df.groupby("product_name"):
            ts = pd.to_datetime(grp[date_col])
            vals = grp.set_index(ts)[value_col].resample("D").sum().fillna(method="ffill")
            cv = float(vals.std() / vals.mean() * 100) if vals.mean() > 0 else 0
            trend = float((vals.iloc[-1] - vals.iloc[0]) / (vals.iloc[0] + 1e-9) * 100) if len(vals) > 1 else 0
            # Zero-demand days
            zero_pct = float((vals == 0).sum() / len(vals) * 100)
            score = min(100, cv * 0.5 + zero_pct * 1.5 + max(0, -trend) * 0.5)
            risks[prod] = {"cv": round(cv, 1), "trend_pct": round(trend, 1),
                           "zero_pct": round(zero_pct, 1), "risk_score": round(score, 1)}
    return risks


def _inventory_risk(df: pd.DataFrame) -> dict:
    risks = {}
    for _, row in df.iterrows():
        prod = str(row.get("product_name", "?"))
        wh   = str(row.get("warehouse_id",  "?"))
        key  = f"{prod}@{wh}"
        stock   = float(row.get("stock_units",  0))
        reorder = float(row.get("reorder_level", 0))
        status  = str(row.get("inventory_status", "")).lower()
        score   = 0
        if stock <= reorder:         score += 40
        if stock <= reorder * 0.5:   score += 30
        if status == "critical":     score += 30
        elif status == "low":        score += 15
        risks[key] = {"stock": stock, "reorder": reorder,
                      "status": status, "risk_score": min(100, score)}
    return risks


def _supplier_risk(df: pd.DataFrame) -> dict:
    risks = {}
    for sup, grp in df.groupby("supplier_name" if "supplier_name" in df.columns else df.columns[0]):
        fr  = grp["fulfillment_rate"].mean() if "fulfillment_rate"  in grp.columns else 1.0
        rel = grp["supplier_reliability_score"].mean() if "supplier_reliability_score" in grp.columns else 1.0
        var = grp["supply_variation_days"].mean() if "supply_variation_days" in grp.columns else 0
        dmg_rate = (grp["cargo_condition_status"].str.lower() == "damaged").mean() if "cargo_condition_status" in grp.columns else 0

        score = 0
        if fr < 0.8:   score += 30
        if rel < 0.75: score += 25
        if var > 2:    score += 20
        if dmg_rate > 0.2: score += 25
        risks[str(sup)] = {
            "fill_rate": round(float(fr), 3),
            "reliability": round(float(rel), 3),
            "avg_delay":  round(float(var), 2),
            "damage_rate": round(float(dmg_rate), 3),
            "risk_score":  min(100, score)
        }
    return risks


def _transport_risk(df: pd.DataFrame) -> dict:
    risks = {}
    if "route_type" in df.columns:
        for rt, grp in df.groupby("route_type"):
            dp  = grp["delay_probability"].mean()   if "delay_probability"  in grp.columns else 0
            rl  = grp["route_risk_level"].mean()     if "route_risk_level"   in grp.columns else 0
            # dev = grp["delivery_time_deviation"].abs().mean() if "delivery_time_deviation" in grp.columns else 0
            dev = _safe_abs_mean(grp["delivery_time_deviation"]) \
            if "delivery_time_deviation" in grp.columns else 0
            fc  = grp["fuel_consumption_rate"].mean() if "fuel_consumption_rate" in grp.columns else 0
            score = min(100, dp * 40 + (rl / 10) * 30 + min(dev, 10) * 3)
            risks[str(rt)] = {
                "delay_prob": round(float(dp), 3),
                "risk_level": round(float(rl), 2),
                "avg_deviation_hrs": round(float(dev), 2),
                "fuel_rate":  round(float(fc), 2),
                "risk_score": round(score, 1)
            }
    return risks


def _overall_risk(d_risk, i_risk, s_risk, t_risk) -> float:
    scores = []
    if d_risk: scores += [v["risk_score"] for v in d_risk.values()]
    if i_risk: scores += [v["risk_score"] for v in i_risk.values()]
    if s_risk: scores += [v["risk_score"] for v in s_risk.values()]
    if t_risk: scores += [v["risk_score"] for v in t_risk.values()]
    return round(float(np.mean(scores)), 1) if scores else 50.0


# ─── Render ──────────────────────────────────────────────────────────────────
def render_risk_assessment():
    st.markdown("## ⚠️ Risk Assessment")
    st.caption("Comprehensive multi-dimensional supply chain risk scoring with LLM-enhanced analysis.")

    from data_loader import get_datasets
    ds = get_datasets()

    if not ds:
        st.info("Upload datasets first (Home page).")
        return

    # Controls
    with st.sidebar:
        st.markdown("### 🎛 Risk Controls")
        focus = st.selectbox("Risk Focus", ["All Dimensions", "Demand Volatility",
                                             "Inventory Risk", "Supplier Risk", "Logistics Risk"])
        nl_query = st.text_area("💬 Query", placeholder="e.g. Which product has highest stockout risk?", height=80)
        run_btn = st.button("▶ Assess Risk", type="primary", use_container_width=True)

    if not run_btn:
        st.info("Configure risk focus and click **▶ Assess Risk**.")
        return

    with st.spinner("🔄 Computing risk metrics…"):
        d_risk = _demand_risk(ds["demand"])   if "demand"    in ds else {}
        i_risk = _inventory_risk(ds["inventory"]) if "inventory" in ds else {}
        s_risk = _supplier_risk(ds["supplier"])   if "supplier"  in ds else {}
        t_risk = _transport_risk(ds["transport"]) if "transport" in ds else {}
        overall = _overall_risk(d_risk, i_risk, s_risk, t_risk)

    # ── KPI row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Overall Risk", f"{overall:.0f}/100",
              delta="⚠ High" if overall > 65 else "✅ Manageable")
    if d_risk:
        avg_d = np.mean([v["risk_score"] for v in d_risk.values()])
        c2.metric("Demand Risk", f"{avg_d:.0f}/100")
    if i_risk:
        avg_i = np.mean([v["risk_score"] for v in i_risk.values()])
        c3.metric("Inventory Risk", f"{avg_i:.0f}/100")
    if s_risk:
        avg_s = np.mean([v["risk_score"] for v in s_risk.values()])
        c4.metric("Supplier Risk", f"{avg_s:.0f}/100")
    if t_risk:
        avg_t = np.mean([v["risk_score"] for v in t_risk.values()])
        c5.metric("Transport Risk", f"{avg_t:.0f}/100")

    # ── Gauge row ─────────────────────────────────────────────────────────────
    g1, g2, g3, g4 = st.columns(4)
    for col, title, data in zip([g1, g2, g3, g4],
                                 ["Demand", "Inventory", "Supplier", "Transport"],
                                 [d_risk, i_risk, s_risk, t_risk]):
        if data:
            score = np.mean([v["risk_score"] for v in data.values()])
            col.plotly_chart(risk_gauge(round(score, 1), f"{title} Risk"), use_container_width=True)

    # ── Detail tabs ───────────────────────────────────────────────────────────
    tabs = st.tabs(["🔴 Demand", "📦 Inventory", "🏭 Supplier", "🚛 Transport", "🧊 3D Risk", "🧠 AI Analysis"])

    with tabs[0]:  # Demand risk
        if d_risk:
            df_d = pd.DataFrame(d_risk).T.reset_index().rename(columns={"index": "product"})
            df_d = df_d.sort_values("risk_score", ascending=False)
            fig = bar_comparison(df_d, "product", "risk_score", title="Demand Risk Score by Product")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_d.style.background_gradient(cmap="RdYlGn_r", subset=["risk_score"]),
                         use_container_width=True, hide_index=True)

    with tabs[1]:  # Inventory risk
        if i_risk:
            df_i = pd.DataFrame(i_risk).T.reset_index().rename(columns={"index": "product@warehouse"})
            df_i[["product", "warehouse"]] = df_i["product@warehouse"].str.split("@", expand=True)
            df_i = df_i.sort_values("risk_score", ascending=False)
            st.dataframe(df_i[["product","warehouse","stock","reorder","status","risk_score"]]
                         .style.background_gradient(cmap="RdYlGn_r", subset=["risk_score"]),
                         use_container_width=True, hide_index=True)
            if "inventory" in ds:
                from charts import inventory_3d_heatmap
                fig3d = inventory_3d_heatmap(ds["inventory"], "product_name", "warehouse_id",
                                             "stock_units", "3D Inventory Stock Distribution")
                st.plotly_chart(fig3d, use_container_width=True)

    with tabs[2]:  # Supplier risk
        if s_risk:
            df_s = pd.DataFrame(s_risk).T.reset_index().rename(columns={"index": "supplier"})
            df_s = df_s.sort_values("risk_score", ascending=False)
            fig_s = bar_comparison(df_s, "supplier", "risk_score", title="Supplier Risk Scores")
            st.plotly_chart(fig_s, use_container_width=True)
            if "supplier" in ds:
                num_cols = ds["supplier"].select_dtypes("number").columns.tolist()[:6]
                if len(num_cols) >= 2:
                    fig_h = heatmap_correlation(ds["supplier"], num_cols, "Supplier Metrics Correlation")
                    st.plotly_chart(fig_h, use_container_width=True)

    with tabs[3]:  # Transport risk
        if t_risk:
            df_t = pd.DataFrame(t_risk).T.reset_index().rename(columns={"index": "route_type"})
            fig_t = bar_comparison(df_t, "route_type", "risk_score", title="Transport Risk by Route Type")
            st.plotly_chart(fig_t, use_container_width=True)
            if "transport" in ds:
                from charts import route_3d_path
                fig_r3d = route_3d_path(ds["transport"], title="3D Route Risk Map")
                st.plotly_chart(fig_r3d, use_container_width=True)

    with tabs[4]:  # 3D Risk scatter
        all_risk_rows = []
        for name, score_dict in {**d_risk, **s_risk}.items():
            all_risk_rows.append({"entity": name,
                                  "risk_score": score_dict.get("risk_score", 0),
                                  "cv_or_fill": score_dict.get("cv", score_dict.get("fill_rate", 0.5)),
                                  "trend_or_delay": score_dict.get("trend_pct", score_dict.get("avg_delay", 0))})
        if len(all_risk_rows) >= 3:
            df_3d = pd.DataFrame(all_risk_rows)
            fig_3d = risk_3d_scatter(df_3d, "cv_or_fill", "trend_or_delay", "risk_score",
                                     color_col="risk_score", label_col="entity",
                                     title="3D Supply Chain Risk Map")
            st.plotly_chart(fig_3d, use_container_width=True)

    with tabs[5]:  # AI analysis
        with st.spinner("🧠 AI risk analysis in progress…"):
            risk_data = {
                "demand_risks":    d_risk,
                "inventory_risks": i_risk,
                "supplier_risks":  s_risk,
                "transport_risks": t_risk,
                "overall_score":   overall,
            }
            ai_result = agent_risk_analysis(risk_data, nl_query or focus)

        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.metric("AI Risk Level", ai_result.get("risk_level", "?"))
            st.metric("Overall Score (AI)", f"{ai_result.get('overall_risk_score',0)}/100")
            if ai_result.get("risks"):
                df_ai = pd.DataFrame(ai_result["risks"])
                st.dataframe(df_ai.style.background_gradient(cmap="RdYlGn_r", subset=["score"]
                             if "score" in df_ai.columns else []),
                             use_container_width=True, hide_index=True)
        with col_r:
            st.markdown("**Top Vulnerabilities**")
            for v in ai_result.get("top_vulnerabilities", []):
                st.warning(v)
            st.markdown("**Immediate Actions**")
            for a in ai_result.get("immediate_actions", []):
                st.error(a)
        st.markdown("---")
        st.markdown(ai_result.get("summary", ""))

        # Save to DB
        doc_id = save_risk(focus, {"computed": risk_data, "ai_analysis": ai_result})
        if doc_id:
            st.success(f"✅ Risk assessment saved (ID: `{doc_id[:8]}…`)")


#helping function
def _safe_abs_mean(series):
    import pandas as pd
    import numpy as np

    if series.empty:
        return 0

    # datetime → convert to timedelta from zero
    if np.issubdtype(series.dtype, np.datetime64):
        series = pd.to_datetime(series) - pd.Timestamp("1970-01-01")

    # timedelta → convert to numeric (hours)
    if np.issubdtype(series.dtype, np.timedelta64):
        series = series.dt.total_seconds() / 3600

    # object → try numeric
    if series.dtype == "object":
        series = pd.to_numeric(series, errors="coerce")

    return np.nanmean(np.abs(series))