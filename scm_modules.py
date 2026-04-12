"""
scm_modules.py — Inventory Management, Seasonality, Stockout Prediction,
                  Supplier Risk, and Route Optimization pages.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

from db_ops import (save_inventory, save_seasonal, save_stockout,
                    save_supplier_risk, save_route, get_latest_forecast,
                    load_all, COLL_FORECAST, COLL_INVENTORY)
from llm_agents import (agent_inventory_insights, agent_seasonality_analysis,
                        agent_stockout_prediction, agent_supplier_risk,
                        agent_route_optimization, web_search)
from charts import (forecast_band_chart, demand_trend_line, demand_seasonality_3d,
                    inventory_3d_heatmap, supplier_3d_bar, route_3d_path,
                    lead_time_3d, risk_gauge, bar_comparison, pie_distribution,
                    heatmap_correlation, sankey_flow, waterfall_chart)


# ════════════════════════════════════════════════════════════════════════════
#  INVENTORY MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════

def render_inventory_management():
    st.markdown("## 📊 Inventory Management")
    st.caption("Intelligent inventory optimization using demand forecasts, safety stock calculation, and EOQ.")

    from data_loader import get_datasets
    ds = get_datasets()
    if "inventory" not in ds:
        st.info("Upload the **inventory** dataset.")
        return

    inv_df = ds["inventory"].copy()

    with st.sidebar:
        st.markdown("### 🎛 Inventory Controls")
        products  = sorted(inv_df["product_name"].unique().tolist()) if "product_name" in inv_df.columns else []
        warehouses = sorted(inv_df["warehouse_id"].unique().tolist()) if "warehouse_id" in inv_df.columns else []
        sel_product   = st.selectbox("Product", ["All"] + products)
        sel_warehouse = st.selectbox("Warehouse", ["All"] + warehouses)
        holding_cost  = st.number_input("Annual Holding Cost per Unit (₹)", 10, 10000, 100)
        ordering_cost = st.number_input("Ordering Cost per Order (₹)", 100, 100000, 5000)
        nl_query = st.text_area("💬 Query", placeholder="Which products are at critical stock levels?", height=80)
        run_btn = st.button("▶ Optimize Inventory", type="primary", use_container_width=True)

    if not run_btn:
        _inventory_overview(inv_df)
        return

    # Filter
    filt = inv_df.copy()
    if sel_product != "All" and "product_name" in filt.columns:
        filt = filt[filt["product_name"] == sel_product]
    if sel_warehouse != "All" and "warehouse_id" in filt.columns:
        filt = filt[filt["warehouse_id"] == sel_warehouse]

    # ── KPI metrics ─────────────────────────────────────────────────────────
    total_stock   = int(filt["stock_units"].sum())   if "stock_units"   in filt.columns else 0
    below_reorder = int((filt["stock_units"] < filt["reorder_level"]).sum()) \
                    if "stock_units" in filt.columns and "reorder_level" in filt.columns else 0
    critical_cnt  = int((filt.get("inventory_status", pd.Series([])).str.lower() == "critical").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Stock Units",   f"{total_stock:,}")
    c2.metric("Below Reorder Level", below_reorder, delta="⚠ Action needed" if below_reorder > 0 else "✅ OK")
    c3.metric("Critical Items",      critical_cnt,  delta="🚨 Urgent" if critical_cnt > 0 else "✅ OK")
    c4.metric("Products Analyzed",   len(filt["product_name"].unique()) if "product_name" in filt.columns else len(filt))

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_status, tab_eoq, tab_3d, tab_ai = st.tabs(
        ["📋 Stock Status", "📐 EOQ & Safety Stock", "🧊 3D View", "🧠 AI Recommendations"])

    with tab_status:
        status_df = filt[["product_name", "warehouse_id", "warehouse_location",
                           "stock_units", "reorder_level", "inventory_status",
                           "physical_condition", "last_restock_date"]
                          if all(c in filt.columns for c in ["product_name","warehouse_id","stock_units"])
                          else filt.columns.tolist()].copy()

        def _color_status(val):
            if str(val).lower() == "critical": return "background-color: rgba(255,71,87,0.3)"
            if str(val).lower() == "low":      return "background-color: rgba(255,184,48,0.25)"
            return "background-color: rgba(0,229,160,0.15)"

        if "inventory_status" in status_df.columns:
            st.dataframe(status_df.style.applymap(_color_status, subset=["inventory_status"]),
                         use_container_width=True, hide_index=True)
        else:
            st.dataframe(status_df, use_container_width=True, hide_index=True)

        if "inventory_status" in filt.columns:
            status_counts = filt["inventory_status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig_pie = pie_distribution(status_counts, "Status", "Count", "Inventory Status Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

    with tab_eoq:
        # EOQ and Safety Stock for each product
        eoq_rows = []
        for prod in filt["product_name"].unique() if "product_name" in filt.columns else []:
            prod_inv = filt[filt["product_name"] == prod]
            avg_stock = float(prod_inv["stock_units"].mean()) if "stock_units" in prod_inv.columns else 100
            reorder   = float(prod_inv["reorder_level"].mean()) if "reorder_level" in prod_inv.columns else 50

            # Get demand from forecast if available
            fc_doc = get_latest_forecast(prod)
            if fc_doc:
                fc_vals = fc_doc.get("result", {}).get("forecast_values", [])
                avg_daily = float(np.mean(fc_vals)) if fc_vals else avg_stock / 30
            else:
                avg_daily = avg_stock / 30  # fallback

            annual_demand = avg_daily * 365
            eoq = np.sqrt(2 * annual_demand * ordering_cost / (holding_cost + 1e-9))
            safety_stock = 1.65 * avg_daily * 2   # 95% SL, 2-day lead time std
            rop = avg_daily * 7 + safety_stock     # 7-day avg + safety stock

            eoq_rows.append({
                "Product": prod,
                "Avg Stock": round(avg_stock, 0),
                "Reorder Level": round(reorder, 0),
                "Avg Daily Demand": round(avg_daily, 1),
                "EOQ (units)": round(eoq, 0),
                "Safety Stock": round(safety_stock, 0),
                "Reorder Point": round(rop, 0),
                "Current vs ROP": "🚨 Below" if avg_stock < rop else "✅ OK",
                "Suggested Order": round(max(0, rop - avg_stock + eoq), 0)
            })

        if eoq_rows:
            eoq_df = pd.DataFrame(eoq_rows)
            st.dataframe(eoq_df.style.applymap(
                lambda v: "background-color: rgba(255,71,87,0.3)" if v == "🚨 Below" else "",
                subset=["Current vs ROP"]), use_container_width=True, hide_index=True)

            # Waterfall for suggested orders
            if len(eoq_rows) > 0:
                fig_wf = waterfall_chart(eoq_df["Product"].tolist(),
                                         eoq_df["Suggested Order"].tolist(),
                                         "Suggested Order Quantities")
                st.plotly_chart(fig_wf, use_container_width=True)

    with tab_3d:
        if "product_name" in filt.columns and "warehouse_id" in filt.columns:
            fig3d = inventory_3d_heatmap(filt, "product_name", "warehouse_id",
                                          "stock_units", "3D Stock Distribution: Product × Warehouse")
            st.plotly_chart(fig3d, use_container_width=True)

    with tab_ai:
        fc_context = get_latest_forecast(sel_product if sel_product != "All" else "")
        with st.spinner("🧠 AI inventory optimization…"):
            inv_data = {
                "products": filt["product_name"].unique().tolist() if "product_name" in filt.columns else [],
                "avg_stock": float(filt["stock_units"].mean()) if "stock_units" in filt.columns else 0,
                "critical": int((filt.get("inventory_status", pd.Series([])).str.lower() == "critical").sum()),
                "below_reorder": int((filt["stock_units"] < filt["reorder_level"]).sum())
                if "stock_units" in filt.columns and "reorder_level" in filt.columns else 0,
                "total_rows": len(filt)
            }
            ai_result = agent_inventory_insights(inv_data, fc_context, nl_query)

        if ai_result.get("reorder_recommendations"):
            st.markdown("#### 🔄 Reorder Recommendations")
            rec_df = pd.DataFrame(ai_result["reorder_recommendations"])
            st.dataframe(rec_df, use_container_width=True, hide_index=True)

        if ai_result.get("overstock_alerts"):
            st.markdown("#### ⚠️ Overstock Alerts")
            for a in ai_result["overstock_alerts"]:
                st.warning(f"**{a.get('product')}** @ {a.get('warehouse')}: {a.get('excess_units')} excess units — {a.get('action')}")

        st.info(ai_result.get("summary", ""))

        doc_id = save_inventory(
            sel_product, sel_warehouse,
            {"inventory_data": inv_data, "eoq_analysis": eoq_rows, "ai_result": ai_result}
        )
        if doc_id:
            st.success(f"✅ Inventory analysis saved (ID: `{doc_id[:8]}…`)")


def _inventory_overview(df: pd.DataFrame):
    st.info("Configure filters and click **▶ Optimize Inventory**.")
    if "inventory_status" in df.columns:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Products", df["product_name"].nunique() if "product_name" in df.columns else len(df))
        c2.metric("Critical", int((df["inventory_status"].str.lower() == "critical").sum()))
        c3.metric("Warehouses", df["warehouse_id"].nunique() if "warehouse_id" in df.columns else 1)
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
#  SEASONALITY
# ════════════════════════════════════════════════════════════════════════════

def render_seasonality():
    st.markdown("## 🌊 Seasonality Analysis")
    st.caption("Decompose demand patterns by season, events, and day-of-week. Web search enriches event context.")

    from data_loader import require_dataset
    df = require_dataset("demand")
    if df is None:
        return

    date_col  = "timestamp" if "timestamp" in df.columns else df.select_dtypes("datetime").columns[0]
    value_col = "units_sold" if "units_sold" in df.columns else df.select_dtypes("number").columns[0]

    with st.sidebar:
        st.markdown("### 🎛 Controls")
        products  = sorted(df["product_name"].unique().tolist()) if "product_name" in df.columns else []
        sel_prod  = st.selectbox("Product", products)
        period    = st.selectbox("Seasonal Period", ["Weekly (7d)", "Monthly (30d)", "Quarterly (90d)"])
        run_btn   = st.button("▶ Analyze Seasonality", type="primary", use_container_width=True)

    if not run_btn:
        st.info("Select product and click **▶ Analyze Seasonality**.")
        return

    df["_dt"] = pd.to_datetime(df[date_col])
    prod_df   = df[df["product_name"] == sel_prod].copy() if "product_name" in df.columns else df.copy()
    prod_df   = prod_df.sort_values("_dt")
    ts        = prod_df.set_index("_dt")[value_col].resample("D").sum().fillna(method="ffill")

    # Build monthly + weekday aggregates
    ts_df = ts.reset_index()
    ts_df.columns = ["date", "value"]
    ts_df["month"]   = ts_df["date"].dt.month
    ts_df["dow"]     = ts_df["date"].dt.dayofweek
    ts_df["week"]    = ts_df["date"].dt.isocalendar().week
    ts_df["quarter"] = ts_df["date"].dt.quarter

    month_avg   = ts_df.groupby("month")["value"].mean()
    dow_avg     = ts_df.groupby("dow")["value"].mean()
    overall_avg = float(ts.mean())

    seasonal_index = {str(i): round(float(v / overall_avg), 3) for i, v in month_avg.items()}

    stats = {
        "peak_month": int(month_avg.idxmax()),
        "low_month":  int(month_avg.idxmin()),
        "peak_dow":   int(dow_avg.idxmax()),
        "low_dow":    int(dow_avg.idxmin()),
        "cv":         round(float(ts.std() / ts.mean() * 100), 1),
        "seasonal_index": seasonal_index,
    }

    # Event-based analysis
    event_stats = {}
    if "event_flag" in prod_df.columns:
        for ev, grp in prod_df.groupby("event_flag"):
            if str(ev).lower() not in ["none", "nan"]:
                ev_avg = grp[value_col].mean()
                event_stats[str(ev)] = {
                    "avg_demand": round(float(ev_avg), 1),
                    "lift_pct":   round((ev_avg / overall_avg - 1) * 100, 1)
                }

    tab_sea, tab_event, tab_3d, tab_ai = st.tabs(
        ["📊 Seasonal Patterns", "🎯 Event Impact", "🧊 3D Seasonality", "🧠 AI Analysis"])

    with tab_sea:
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        dow_names   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

        c1, c2 = st.columns(2)
        with c1:
            import plotly.graph_objects as go
            from charts import DARK_LAYOUT, _apply_dark
            fig_m = go.Figure(go.Bar(
                x=[month_names[i-1] for i in month_avg.index],
                y=month_avg.values,
                marker_color=["#ff4757" if i == month_avg.idxmax() else "#00e5a0" if i == month_avg.idxmin() else "#00d4ff"
                               for i in month_avg.index]
            ))
            _apply_dark(fig_m, f"{sel_prod} — Monthly Avg Demand")
            st.plotly_chart(fig_m, use_container_width=True)
        with c2:
            fig_d = go.Figure(go.Bar(
                x=[dow_names[i] for i in dow_avg.index],
                y=dow_avg.values,
                marker_color=["#ff4757" if i == dow_avg.idxmax() else "#00e5a0" if i == dow_avg.idxmin() else "#9b8cff"
                               for i in dow_avg.index]
            ))
            _apply_dark(fig_d, f"{sel_prod} — Day-of-Week Avg Demand")
            st.plotly_chart(fig_d, use_container_width=True)

        # Seasonal index table
        si_df = pd.DataFrame({"Month": month_names,
                               "Avg Demand": [round(float(month_avg.get(i+1, 0)), 1) for i in range(12)],
                               "Seasonal Index": [round(float(month_avg.get(i+1, overall_avg)) / (overall_avg+1e-9), 3) for i in range(12)]})
        st.dataframe(si_df.style.background_gradient(cmap="RdYlGn", subset=["Seasonal Index"]),
                     use_container_width=True, hide_index=True)

    with tab_event:
        if event_stats:
            ev_df = pd.DataFrame(event_stats).T.reset_index().rename(columns={"index": "event"})
            c1, c2 = st.columns(2)
            with c1:
                fig_ev = bar_comparison(ev_df, "event", "lift_pct", title="Event Demand Lift (%)")
                st.plotly_chart(fig_ev, use_container_width=True)
            with c2:
                for ev, info in event_stats.items():
                    color = "🟢" if info["lift_pct"] > 20 else "🟡" if info["lift_pct"] > 5 else "🔴"
                    st.metric(f"{color} {ev}", f"{info['avg_demand']:.0f} units",
                              delta=f"{info['lift_pct']:+.1f}% vs baseline")
        else:
            if "season" in prod_df.columns:
                sea_stats = prod_df.groupby("season")[value_col].mean().reset_index()
                fig_sea = bar_comparison(sea_stats, "season", value_col, title="Demand by Season")
                st.plotly_chart(fig_sea, use_container_width=True)

    with tab_3d:
        fig3d = demand_seasonality_3d(prod_df, "_dt", "product_name" if "product_name" in prod_df.columns else "_dt",
                                      value_col, f"{sel_prod} — 3D Seasonality (Month × Day of Week)")
        st.plotly_chart(fig3d, use_container_width=True)

    with tab_ai:
        with st.spinner("🔍 Searching for seasonal events…"):
            search_q = f"{sel_prod} seasonal demand India festivals events {datetime.now().year}"
            search_ctx = web_search(search_q)
        with st.spinner("🧠 AI seasonality analysis…"):
            ai_result = agent_seasonality_analysis(sel_prod, {**stats, "event_impact": event_stats}, search_ctx)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Peak Months:**", )
            for m in ai_result.get("peak_months", []):
                st.success(f"📈 {m}")
            st.markdown("**Key Events:**")
            for ev in ai_result.get("peak_events", []):
                st.info(f"🎯 **{ev.get('event','')}** — {ev.get('typical_lift','?')} lift, {ev.get('preparation_lead_days','?')}d prep needed")
        with col2:
            si = ai_result.get("seasonal_index", {})
            if si:
                si_df2 = pd.DataFrame(list(si.items()), columns=["Month", "Index"])
                import plotly.graph_objects as go
                from charts import _apply_dark
                fig_si = go.Figure(go.Scatterpolar(
                    r=[float(v) for v in si.values()],
                    theta=list(si.keys()),
                    fill="toself",
                    line_color="#00d4ff"
                ))
                _apply_dark(fig_si, "Seasonal Index Radar")
                st.plotly_chart(fig_si, use_container_width=True)

        st.markdown("**AI Recommendations:**")
        for r in ai_result.get("recommendations", []):
            st.info(r)

        if search_ctx and "not available" not in search_ctx.lower():
            with st.expander("🌐 Web Research"):
                st.text(search_ctx)

        doc_id = save_seasonal(sel_prod, {"stats": stats, "event_impact": event_stats, "ai_result": ai_result})
        if doc_id:
            st.success(f"✅ Seasonality saved (ID: `{doc_id[:8]}…`)")


# ════════════════════════════════════════════════════════════════════════════
#  STOCKOUT PREDICTION
# ════════════════════════════════════════════════════════════════════════════

def render_stockout():
    st.markdown("## 🚨 Stockout Prediction")
    st.caption("AI-driven stockout probability using demand forecasts, current inventory, and lead time analysis.")

    from data_loader import get_datasets
    ds = get_datasets()
    if "inventory" not in ds:
        st.info("Upload inventory dataset.")
        return

    inv_df = ds["inventory"]

    with st.sidebar:
        st.markdown("### 🎛 Controls")
        products  = sorted(inv_df["product_name"].unique().tolist()) if "product_name" in inv_df.columns else []
        sel_prod  = st.selectbox("Product", products)
        run_btn   = st.button("▶ Predict Stockout", type="primary", use_container_width=True)

    if not run_btn:
        _stockout_overview(inv_df)
        return

    prod_inv  = inv_df[inv_df["product_name"] == sel_prod].copy() if "product_name" in inv_df.columns else inv_df.copy()
    avg_stock = float(prod_inv["stock_units"].mean()) if "stock_units" in prod_inv.columns else 100
    reorder   = float(prod_inv["reorder_level"].mean()) if "reorder_level" in prod_inv.columns else 50
    status    = str(prod_inv["inventory_status"].mode()[0]) if "inventory_status" in prod_inv.columns else "unknown"

    # Get forecast
    fc_doc = get_latest_forecast(sel_prod)
    fc_vals = fc_doc.get("result", {}).get("forecast_values", []) if fc_doc else []
    avg_daily = float(np.mean(fc_vals)) if fc_vals else avg_stock / 14

    # Compute days to stockout
    days_to_stockout = int(avg_stock / avg_daily) if avg_daily > 0 else 999
    prob_7d  = max(0, min(1, 1 - (avg_stock / (avg_daily * 7 + 1e-9))))
    prob_30d = max(0, min(1, 1 - (avg_stock / (avg_daily * 30 + 1e-9))))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Stock",        f"{avg_stock:.0f} units")
    c2.metric("Avg Daily Demand",     f"{avg_daily:.1f} units/day")
    c3.metric("Est. Days to Stockout", days_to_stockout,
              delta="🚨 Critical!" if days_to_stockout < 7 else "✅ OK")
    c4.metric("30-day Stockout Risk", f"{prob_30d*100:.1f}%")

    # Depletion curve
    import plotly.graph_objects as go
    from charts import _apply_dark
    days = list(range(60))
    depletion = [max(0, avg_stock - avg_daily * d) for d in days]
    fig_dep = go.Figure()
    fig_dep.add_trace(go.Scatter(x=days, y=depletion, mode="lines", name="Stock Level",
                                  line=dict(color="#00d4ff", width=2),
                                  fill="tozeroy", fillcolor="rgba(0,212,255,0.06)"))
    fig_dep.add_hline(y=reorder, line_dash="dot", line_color="#ffb830",
                      annotation_text=f"Reorder Point ({reorder:.0f})")
    fig_dep.add_hline(y=0, line_dash="dash", line_color="#ff4757", annotation_text="Stockout")
    _apply_dark(fig_dep, f"{sel_prod} — Stock Depletion Forecast")
    st.plotly_chart(fig_dep, use_container_width=True)

    # AI prediction
    with st.spinner("🧠 AI stockout prediction…"):
        stats  = {"avg_daily_demand": avg_daily, "cv": 30, "lead_time_days": 7}
        inv_st = {"stock_units": avg_stock, "reorder_level": reorder, "status": status}
        ai_pred = agent_stockout_prediction(sel_prod, stats, inv_st)

    c_l, c_r = st.columns(2)
    with c_l:
        st.metric("AI Risk Level",    ai_pred.get("risk_level", "?"))
        st.metric("AI 7d Probability", f"{float(ai_pred.get('stockout_probability_7d',prob_7d))*100:.1f}%")
        st.metric("AI 30d Probability", f"{float(ai_pred.get('stockout_probability_30d',prob_30d))*100:.1f}%")
        st.metric("Recommended Order", f"{ai_pred.get('recommended_order_qty',0)} units")
    with c_r:
        st.markdown("**Contributing Factors:**")
        for f in ai_pred.get("contributing_factors", []):
            st.warning(f)
        if ai_pred.get("recommended_order_by"):
            st.error(f"⏰ Order by: **{ai_pred['recommended_order_by']}**")

    st.info(ai_pred.get("summary", ""))

    doc_id = save_stockout(sel_prod, {"computed": {"days_to_stockout": days_to_stockout,
                                                    "prob_7d": prob_7d, "prob_30d": prob_30d},
                                      "ai_prediction": ai_pred})
    if doc_id:
        st.success(f"✅ Stockout prediction saved (ID: `{doc_id[:8]}…`)")


def _stockout_overview(df: pd.DataFrame):
    st.info("Select a product and click **▶ Predict Stockout**.")
    if "inventory_status" in df.columns:
        critical = df[df["inventory_status"].str.lower().isin(["critical", "low"])]
        if not critical.empty:
            st.warning(f"⚠ **{len(critical)} items** at low/critical stock levels:")
            st.dataframe(critical[["product_name","warehouse_id","stock_units","reorder_level","inventory_status"]].head(10),
                         use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
#  SUPPLIER RISK
# ════════════════════════════════════════════════════════════════════════════

def render_supplier_risk():
    st.markdown("## 🏭 Supplier Risk & Performance")
    st.caption("Deep supplier risk scoring, lead time analysis, fulfillment tracking, and concentration risk.")

    from data_loader import require_dataset
    df = require_dataset("supplier")
    if df is None:
        return

    with st.sidebar:
        st.markdown("### 🎛 Controls")
        suppliers = sorted(df["supplier_name"].unique().tolist()) if "supplier_name" in df.columns else []
        sel_sup   = st.selectbox("Supplier", ["All"] + suppliers)
        nl_query  = st.text_area("💬 Query", placeholder="Which supplier is riskiest?", height=80)
        run_btn   = st.button("▶ Assess Supplier Risk", type="primary", use_container_width=True)

    if not run_btn:
        _supplier_overview(df)
        return

    filt = df[df["supplier_name"] == sel_sup].copy() if sel_sup != "All" and "supplier_name" in df.columns else df.copy()

    # ── Metrics ────────────────────────────────────────────────────────────────
    avg_fill = float(filt["fulfillment_rate"].mean()) if "fulfillment_rate" in filt.columns else 0
    avg_rel  = float(filt["supplier_reliability_score"].mean()) if "supplier_reliability_score" in filt.columns else 0
    avg_var  = float(filt["supply_variation_days"].mean()) if "supply_variation_days" in filt.columns else 0
    dmg_rate = float((filt["cargo_condition_status"].str.lower() == "damaged").mean()) if "cargo_condition_status" in filt.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Fill Rate",        f"{avg_fill*100:.1f}%", delta="✅" if avg_fill > 0.9 else "⚠")
    c2.metric("Avg Reliability",      f"{avg_rel*100:.1f}%",  delta="✅" if avg_rel > 0.85 else "⚠")
    c3.metric("Avg Delay (days)",     f"{avg_var:.1f}",       delta="⚠ High" if avg_var > 1 else "✅")
    c4.metric("Damaged Cargo Rate",   f"{dmg_rate*100:.1f}%", delta="🚨" if dmg_rate > 0.15 else "✅")

    tab_perf, tab_3d, tab_lead, tab_ai = st.tabs(
        ["📊 Performance", "🧊 3D View", "⏱ Lead Time", "🧠 AI Risk Report"])

    with tab_perf:
        if "supplier_name" in filt.columns:
            grp = filt.groupby("supplier_name").agg({
                "fulfillment_rate": "mean",
                "supplier_reliability_score": "mean",
                "supply_variation_days": "mean",
            }).reset_index()
            grp.columns = ["Supplier", "Fill Rate", "Reliability", "Avg Delay"]
            st.dataframe(grp.style.background_gradient(cmap="RdYlGn", subset=["Fill Rate", "Reliability"]),
                         use_container_width=True, hide_index=True)
            fig_b = bar_comparison(grp, "Supplier", "Fill Rate", title="Fulfillment Rate by Supplier")
            st.plotly_chart(fig_b, use_container_width=True)

    with tab_3d:
        if "supplier_name" in filt.columns and "product_name" in filt.columns:
            fig3d = supplier_3d_bar(filt, "supplier_name", "product_name",
                                    "fulfillment_rate", "3D Supplier Fulfillment Matrix")
            st.plotly_chart(fig3d, use_container_width=True)

    with tab_lead:
        if "supplier_name" in filt.columns:
            fig_lt = lead_time_3d(filt, "supplier_name", "product_name",
                                  title="Lead Time: Ideal vs Actual vs Reliability")
            st.plotly_chart(fig_lt, use_container_width=True)

    with tab_ai:
        metrics = {"fill_rate": avg_fill, "reliability": avg_rel,
                   "avg_delay": avg_var, "damage_rate": dmg_rate,
                   "products": filt["product_name"].unique().tolist() if "product_name" in filt.columns else []}
        with st.spinner("🧠 AI supplier risk analysis…"):
            ai_result = agent_supplier_risk(sel_sup, metrics, nl_query)

        c_l, c_r = st.columns([2, 1])
        with c_l:
            st.metric("AI Risk Score", f"{ai_result.get('overall_risk_score',0)}/100")
            if ai_result.get("risk_categories"):
                cats = ai_result["risk_categories"]
                cat_df = pd.DataFrame({"Category": list(cats.keys()), "Score": list(cats.values())})
                from charts import _apply_dark
                import plotly.graph_objects as go
                fig_cat = go.Figure(go.Bar(x=cat_df["Category"], y=cat_df["Score"],
                                           marker_color=["#ff4757" if v > 65 else "#ffb830" if v > 40 else "#00e5a0"
                                                          for v in cat_df["Score"]]))
                _apply_dark(fig_cat, "Risk Category Breakdown")
                st.plotly_chart(fig_cat, use_container_width=True)
        with c_r:
            st.markdown("**🚩 Red Flags:**")
            for rf in ai_result.get("red_flags", []):
                st.error(rf)
            st.markdown("**💪 Strengths:**")
            for s in ai_result.get("strengths", []):
                st.success(s)

        st.markdown("**Recommended Actions:**")
        for a in ai_result.get("recommended_actions", []):
            st.info(f"[{a.get('priority','?')}] {a.get('action','?')} — {a.get('timeline','?')}")
        st.markdown(ai_result.get("summary", ""))

        doc_id = save_supplier_risk(sel_sup, {"metrics": metrics, "ai_result": ai_result})
        if doc_id:
            st.success(f"✅ Supplier risk saved (ID: `{doc_id[:8]}…`)")


def _supplier_overview(df: pd.DataFrame):
    st.info("Select a supplier and click **▶ Assess Supplier Risk**.")
    if "supplier_name" in df.columns and "fulfillment_rate" in df.columns:
        ov = df.groupby("supplier_name")["fulfillment_rate"].mean().reset_index()
        ov.columns = ["Supplier", "Avg Fill Rate"]
        st.dataframe(ov.style.background_gradient(cmap="RdYlGn", subset=["Avg Fill Rate"]),
                     use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
#  ROUTE OPTIMIZATION
# ════════════════════════════════════════════════════════════════════════════

#mean function for all dtype

def safe_mean(series: pd.Series, default=0.0) -> float:
    """Compute mean safely regardless of dtype."""
    
    if series is None or len(series) == 0:
        return default

    # Case 1: timedelta → convert to days
    if pd.api.types.is_timedelta64_dtype(series):
        val = series.dt.total_seconds() / 86400  # convert to days
        mean_val = val.mean()

    # Case 2: datetime → not meaningful → return default
    elif pd.api.types.is_datetime64_any_dtype(series):
        return default

    # Case 3: numeric
    elif pd.api.types.is_numeric_dtype(series):
        mean_val = series.mean()

    # Case 4: object / mixed → try convert
    else:
        val = pd.to_numeric(series, errors="coerce")
        mean_val = val.mean()

    return float(mean_val) if pd.notna(mean_val) else default



def render_route_optimization():
    st.markdown("## 🚛 Route Optimization")
    st.caption("Analyze transport routes, delay patterns, fuel efficiency, and cost optimization paths.")

    from data_loader import require_dataset
    df = require_dataset("transport")
    if df is None:
        return

    with st.sidebar:
        st.markdown("### 🎛 Controls")
        nl_query = st.text_area("💬 Query", placeholder="Which route has highest delay risk?", height=80)
        run_btn  = st.button("▶ Optimize Routes", type="primary", use_container_width=True)

    if not run_btn:
        st.info("Click **▶ Optimize Routes** to analyze.")
        if "route_type" in df.columns:
            st.dataframe(df[["product_name","route_type","risk_classification",
                              "shipping_costs","lead_time_days","delay_probability"]].head(10),
                         use_container_width=True, hide_index=True)
        return

    tab_map, tab_cost, tab_delay, tab_ai = st.tabs(
        ["🗺️ Route Map 3D", "💰 Cost Analysis", "⏰ Delay Patterns", "🧠 AI Optimization"])

    with tab_map:
        if "vehicle_gps_latitude" in df.columns:
            fig3d = route_3d_path(df, title="3D Route Risk Visualization")
            st.plotly_chart(fig3d, use_container_width=True)

    with tab_cost:
        if "route_type" in df.columns and "shipping_costs" in df.columns:
            cost_df = df.groupby("route_type")["shipping_costs"].agg(["mean","sum","count"]).reset_index()
            cost_df.columns = ["Route", "Avg Cost", "Total Cost", "Shipments"]
            c1, c2 = st.columns(2)
            with c1:
                fig_c = bar_comparison(cost_df, "Route", "Avg Cost", title="Avg Shipping Cost by Route")
                st.plotly_chart(fig_c, use_container_width=True)
            with c2:
                fig_p = pie_distribution(cost_df, "Route", "Total Cost", "Cost Distribution by Route")
                st.plotly_chart(fig_p, use_container_width=True)

    with tab_delay:
        if "delay_probability" in df.columns:
            grp = df.groupby("route_type").agg({
                "delay_probability": "mean",
                "eta_variation_hours": "mean" if "eta_variation_hours" in df.columns else "count",
                "delivery_time_deviation": "mean" if "delivery_time_deviation" in df.columns else "count",
            }).reset_index()
            st.dataframe(grp.style.background_gradient(cmap="RdYlGn_r", subset=["delay_probability"]),
                         use_container_width=True, hide_index=True)

            if "risk_classification" in df.columns:
                risk_counts = df["risk_classification"].value_counts().reset_index()
                risk_counts.columns = ["Risk Level", "Count"]
                fig_pie = pie_distribution(risk_counts, "Risk Level", "Count", "Risk Classification Distribution")
                st.plotly_chart(fig_pie, use_container_width=True)

    with tab_ai:
        num_cols = df.select_dtypes("number").columns.tolist()
        route_data = {
            "route_types": df["route_type"].unique().tolist() if "route_type" in df.columns else [],
            "avg_delay_prob": float(df["delay_probability"].mean()) if "delay_probability" in df.columns else 0,
            "avg_cost":       float(df["shipping_costs"].mean()) if "shipping_costs" in df.columns else 0,
            "high_risk_pct":  float((df["risk_classification"].str.lower() == "high").mean() * 100) if "risk_classification" in df.columns else 0,
            "avg_lead_time": safe_mean(df.get("lead_time_days")) if "lead_time_days" in df.columns else 0,
        }
        with st.spinner("🧠 AI route optimization…"):
            ai_result = agent_route_optimization(route_data, nl_query)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Logistics Score", f"{ai_result.get('overall_logistics_score',0)}/100")
            if ai_result.get("high_risk_routes"):
                st.markdown("**⚠️ High Risk Routes:**")
                for r in ai_result["high_risk_routes"]:
                    st.error(f"**{r.get('route')}** — Risk {r.get('risk_score')}/100 | Avg delay {r.get('avg_delay_hrs')}h | {r.get('reason')}")
        with col2:
            if ai_result.get("cost_optimization"):
                st.markdown("**💰 Cost Savings Opportunities:**")
                for c in ai_result["cost_optimization"]:
                    st.success(f"Switch to **{c.get('switch_to')}** → Save ₹{c.get('potential_saving',0):,}/shipment")

        st.markdown("**Recommendations:**")
        for r in ai_result.get("recommendations", []):
            st.info(r)
        st.info(ai_result.get("summary", ""))

        doc_id = save_route({"route_data": route_data, "ai_result": ai_result})
        if doc_id:
            st.success(f"✅ Route optimization saved (ID: `{doc_id[:8]}…`)")
