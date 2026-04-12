"""
demand_forecasting.py — Full demand forecasting pipeline.
Strategy: Data prep → Feature engineering → Statistical model →
          LLM insight enrichment → Web search for events → Save to DB
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

from db_ops import save_forecast, get_latest_forecast, load_all, COLL_FORECAST
from llm_agents import (agent_generate_forecast_insights, agent_interpret_forecast_query,
                        web_search)
from charts import (forecast_band_chart, demand_trend_line, demand_3d_surface,
                    seasonal_decomposition_chart, demand_seasonality_3d, anomaly_scatter)


# ─── Core forecasting engine ─────────────────────────────────────────────────
def _compute_stats(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {}
    return {
        "mean":       round(float(s.mean()), 2),
        "std":        round(float(s.std()), 2),
        "min":        round(float(s.min()), 2),
        "max":        round(float(s.max()), 2),
        "cv":         round(float(s.std() / s.mean() * 100) if s.mean() else 0, 1),
        "q1":         round(float(s.quantile(0.25)), 2),
        "q3":         round(float(s.quantile(0.75)), 2),
        "total":      round(float(s.sum()), 2),
        "trend_pct":  round((s.iloc[-1] - s.iloc[0]) / abs(s.iloc[0] + 1e-9) * 100, 1) if len(s) > 1 else 0,
        "growth_rate":round((s.iloc[-int(len(s)*0.1):].mean() - s.iloc[:int(len(s)*0.1)].mean())
                            / (s.iloc[:int(len(s)*0.1)].mean() + 1e-9) * 100, 1) if len(s) > 10 else 0,
    }


def _detect_seasonality(series: pd.Series, period: int = 7) -> np.ndarray:
    """Simple seasonal index via period averaging."""
    s = series.values.astype(float)
    n = len(s)
    seasonal = np.zeros(period)
    counts   = np.zeros(period)
    for i, v in enumerate(s):
        if not np.isnan(v):
            seasonal[i % period] += v
            counts[i % period]   += 1
    seasonal = np.where(counts > 0, seasonal / counts, 1.0)
    mean_s   = seasonal.mean()
    seasonal = seasonal / (mean_s + 1e-9)
    return seasonal


def _trend_forecast(series: pd.Series, horizon: int, method: str = "ets") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (forecast, upper_bound, lower_bound) for `horizon` days.
    Methods: ets (Holt-Winters-like) | linear | ensemble
    """
    s = series.dropna().values.astype(float)
    if len(s) < 2:
        base = s[-1] if len(s) else 100.0
        fc = np.full(horizon, base)
        return fc, fc * 1.1, fc * 0.9

    # ── ETS (simple exponential + trend) ──────────────────────────────
    alpha, beta = 0.3, 0.1
    level  = s[0]
    trend  = s[1] - s[0]
    for v in s[1:]:
        prev_level = level
        level = alpha * v + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend

    # Weekly seasonality
    seasonal = _detect_seasonality(series, period=7)
    ets_fc = np.array([(level + trend * (i + 1)) * seasonal[(len(s) + i) % 7]
                        for i in range(horizon)])
    ets_fc = np.maximum(ets_fc, 0)

    # ── Linear regression ─────────────────────────────────────────────
    x = np.arange(len(s), dtype=float)
    coeffs = np.polyfit(x, s, 1)
    lin_fc = np.array([coeffs[0] * (len(s) + i) + coeffs[1] for i in range(horizon)])
    lin_fc = np.maximum(lin_fc, 0)

    if method == "linear":
        fc = lin_fc
    elif method == "ets":
        fc = ets_fc
    else:  # ensemble
        fc = 0.6 * ets_fc + 0.4 * lin_fc

    # Confidence interval: ±1.65 std of residuals
    residuals = s - (np.polyval(coeffs, x))
    std_r = residuals.std()
    upper = fc + 1.65 * std_r
    lower = np.maximum(fc - 1.65 * std_r, 0)
    return np.round(fc, 1), np.round(upper, 1), np.round(lower, 1)


def _seasonal_decompose(series: pd.Series, period: int = 7):
    """Manual additive decomposition → (trend, seasonal, residual)."""
    s = series.values.astype(float)
    n = len(s)
    # Trend: centered moving average
    half = period // 2
    trend = np.full(n, np.nan)
    for i in range(half, n - half):
        trend[i] = np.nanmean(s[i - half: i + half + 1])
    # Fill edges
    trend[:half]  = trend[half]
    trend[-half:] = trend[-half - 1]

    detrended  = s - trend
    seasonal   = np.zeros(n)
    idx_season = np.zeros(period)
    counts     = np.zeros(period)
    for i in range(n):
        idx_season[i % period] += detrended[i] if not np.isnan(detrended[i]) else 0
        counts[i % period]     += 1
    idx_season /= (counts + 1e-9)
    for i in range(n):
        seasonal[i] = idx_season[i % period]

    residual = s - trend - seasonal
    return trend, seasonal, residual


def _detect_anomalies(series: pd.Series, threshold_z: float = 2.5) -> pd.Series:
    z = (series - series.rolling(14, min_periods=3).mean()) / \
        (series.rolling(14, min_periods=3).std() + 1e-9)
    return z.abs() > threshold_z


# ─── Main Streamlit page renderer ────────────────────────────────────────────
def render_demand_forecasting():
    st.markdown("## 📦 Demand Forecasting")
    st.caption("AI-powered demand forecasting with seasonality, event detection, and natural language query support.")

    from data_loader import require_dataset
    df = require_dataset("demand")
    if df is None:
        return

    # ── Sidebar controls ────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🎛 Forecast Controls")
        products = sorted(df["product_name"].unique().tolist()) if "product_name" in df.columns else []
        selected_products = st.multiselect("Select Products", products,
                                            default=products[:2] if len(products) >= 2 else products)

        date_col = "timestamp" if "timestamp" in df.columns else df.select_dtypes("datetime").columns[0]
        min_date = pd.to_datetime(df[date_col]).min().date()
        max_date = pd.to_datetime(df[date_col]).max().date()

        st.markdown("**Analysis Window**")
        analysis_start = st.date_input("From", value=min_date, min_value=min_date, max_value=max_date)
        analysis_end   = st.date_input("To",   value=max_date, min_value=min_date, max_value=max_date)

        horizon = st.slider("Forecast Horizon (days)", 7, 180, 30)
        method  = st.selectbox("Forecast Method", ["ensemble", "ets", "linear"])
        conf    = st.select_slider("Confidence Level", [0.80, 0.90, 0.95, 0.99], value=0.95)

        st.markdown("---")
        nl_query = st.text_area("💬 Natural Language Query",
                                placeholder="e.g. Forecast Laptop demand for next 60 days focusing on festive season…",
                                height=100)
        run_btn = st.button("▶ Run Forecast", type="primary", use_container_width=True)

    # ── Parse NL query ───────────────────────────────────────────────────────
    if nl_query and run_btn:
        with st.spinner("🤖 Interpreting query…"):
            parsed = agent_interpret_forecast_query(
                nl_query,
                available_products=products,
                available_cols=list(df.columns)
            )
        if parsed.get("products"):
            selected_products = [p for p in parsed["products"] if p in products] or selected_products
        if parsed.get("horizon_days"):
            horizon = int(parsed["horizon_days"])
        if parsed.get("analysis_window_days"):
            days_back = int(parsed["analysis_window_days"])
            analysis_start = (pd.to_datetime(max_date) - timedelta(days=days_back)).date()
        with st.info(f"**AI Interpretation:** {parsed.get('explanation', '')}"):
            pass

    if not run_btn and not nl_query:
        _show_overview(df, date_col)
        return

    if not selected_products:
        st.warning("Select at least one product.")
        return

    # ── Filter data ──────────────────────────────────────────────────────────
    df["_dt"] = pd.to_datetime(df[date_col])
    mask = (df["_dt"].dt.date >= analysis_start) & (df["_dt"].dt.date <= analysis_end)
    filtered = df[mask].copy()

    value_col = "units_sold" if "units_sold" in df.columns else \
                [c for c in df.columns if "unit" in c or "demand" in c or "sale" in c][0]

    for product in selected_products:
        st.markdown(f"---\n### 📊 {product}")
        prod_df = filtered[filtered["product_name"] == product].copy() if "product_name" in filtered.columns else filtered.copy()

        if prod_df.empty:
            st.warning(f"No data for {product} in selected range.")
            continue

        prod_df = prod_df.sort_values("_dt")
        ts = prod_df.set_index("_dt")[value_col].resample("D").sum().fillna(method="ffill")
        if ts.empty:
            st.warning(f"No time series data for {product}")
            continue


        stats = _compute_stats(ts)
        fc_vals, upper, lower = _trend_forecast(ts, horizon, method)
        fc_dates = pd.date_range(ts.index[-1] + timedelta(days=1), periods=horizon, freq="D")

        # ── Tabs ─────────────────────────────────────────────────────────────
        tab_fc, tab_decomp, tab_3d, tab_anomaly, tab_insights = st.tabs(
            ["📈 Forecast", "🔬 Decomposition", "🧊 3D View", "⚡ Anomalies", "🧠 AI Insights"])

        with tab_fc:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean Demand",   f"{stats.get('mean',0):,.0f}")
            c2.metric("CV (Volatility)", f"{stats.get('cv',0):.1f}%")
            c3.metric("Forecast Avg",  f"{fc_vals.mean():.0f}")
            c4.metric("Trend",         f"{stats.get('trend_pct',0):+.1f}%")
            fig = forecast_band_chart(ts.index, ts.values, fc_dates, fc_vals,
                                      upper, lower, product, conf)
            st.plotly_chart(fig, use_container_width=True,key=f"forecast_{product}")

        with tab_decomp:
            trend_d, seasonal_d, residual_d = _seasonal_decompose(ts)
            fig2 = seasonal_decomposition_chart(ts.index, trend_d, seasonal_d, residual_d,
                                                f"{product} — Seasonal Decomposition")
            st.plotly_chart(fig2, use_container_width=True, key=f"decomp_{product}")

        with tab_3d:
            c3d1, c3d2 = st.columns(2)
            with c3d1:
                if "product_name" in filtered.columns:
                    fig3d = demand_3d_surface(filtered, "_dt", "product_name", value_col,
                                              "3D Monthly Demand Surface (All Products)")
                    st.plotly_chart(fig3d, use_container_width=True,key=f"3d_surface_{product}")
            with c3d2:
                fig3d2 = demand_seasonality_3d(prod_df, "_dt", "product_name" if "product_name" in prod_df.columns else "_dt",
                                               value_col, f"{product} — Seasonality Pattern 3D")
                st.plotly_chart(fig3d2, use_container_width=True,key=f"3d_season_{product}")

        with tab_anomaly:
            anomaly_mask = _detect_anomalies(ts)
            fig_an = anomaly_scatter(pd.DataFrame({"date": ts.index, "value": ts.values}),
                                     "date", "value", pd.Series(anomaly_mask.values),
                                     f"{product} — Anomaly Detection")
            st.plotly_chart(fig_an, use_container_width=True,key=f"anomaly_{product}")
            n_anom = int(anomaly_mask.sum())
            st.info(f"Detected **{n_anom}** anomalies ({n_anom/len(ts)*100:.1f}% of data points)")

        with tab_insights:

            # -------- Web Search ----------
            with st.spinner("🔍 Searching for seasonal events…"):
                search_q = f"{product} demand seasonality India {datetime.now().year} events festivals"
                search_ctx = web_search(search_q)

            # -------- AI Insights ----------
            with st.spinner("🧠 Generating AI forecast insights…"):
                insight = agent_generate_forecast_insights(
                    product, stats, fc_vals.tolist(), search_ctx
                )

            # if string -> convert to dict
            if isinstance(insight, str):
                import json
                try:
                    insight = json.loads(insight)
                except:
                    st.markdown(insight)
                    insight = None

            # -------- Beautiful UI ----------
            if insight:

                st.markdown("## 📈 Demand Forecast Insights")

                # ---- Trend ----
                st.markdown("### 🔎 Trend Interpretation")
                st.info(insight.get("trend_interpretation", ""))

                # ---- Drivers + Inventory tabs ----
                tab1, tab2, tab3 = st.tabs([
                    "🚀 Demand Drivers",
                    "📦 Inventory",
                    "🎯 Actions"
                ])

                # ---------- Drivers ----------
                with tab1:
                    # for driver in insight.get("key_demand_drivers", []):
                    #     st.markdown(f"""
                    #     **{driver['driver']}**  
                    #     {driver['impact']}
                    #     """)
                    drivers = insight.get("key_demand_drivers", [])

                    # Normalize drivers
                    normalized_drivers = []

                    if isinstance(drivers, list):
                        for d in drivers:
                            if isinstance(d, dict):
                                normalized_drivers.append({
                                    "driver": d.get("driver", "Unknown"),
                                    "impact": d.get("impact", "")
                                })
                            elif isinstance(d, str):
                                normalized_drivers.append({
                                    "driver": d,
                                    "impact": ""
                                })
                            else:
                                normalized_drivers.append({
                                    "driver": str(d),
                                    "impact": ""
                                })

                    elif isinstance(drivers, str):
                        normalized_drivers = [{"driver": drivers, "impact": ""}]

                    # Render safely
                    for driver in normalized_drivers:
                        st.markdown(f"""
                    **{driver['driver']}**  
                    {driver['impact']}
                    """)
                    
                    st.markdown("### 📅 Seasonal Impact")
                    st.success(insight.get("seasonal_event_impacts", ""))

                # ---------- Inventory ----------
                with tab2:
                    st.warning(insight.get("inventory_implications", ""))

                # ---------- Actions ----------
                with tab3:
                    # for act in insight.get("action_recommendations", []):
                    #     st.markdown(f"• {act}")
                    actions = insight.get("action_recommendations", [])

                    if isinstance(actions, str):
                        actions = [actions]

                    for act in actions:
                        st.markdown(f"• {str(act)}")

            # -------- Web Context ----------
            if search_ctx and "not available" not in search_ctx.lower():
                with st.expander("🌐 Web Research Context"):
                    st.text(search_ctx)

        # ── Save to DB ────────────────────────────────────────────────────────
        doc_id = save_forecast(product, horizon, {
            "stats":          stats,
            "forecast_values": fc_vals.tolist(),
            "upper_bound":    upper.tolist(),
            "lower_bound":    lower.tolist(),
            "forecast_dates": [str(d.date()) for d in fc_dates],
            "hist_start":     str(analysis_start),
            "hist_end":       str(analysis_end),
            "method":         method,
            "confidence":     conf,
            "insight":        insight if insight else "",
        })
        if doc_id:
            st.success(f"✅ Forecast saved to database (ID: `{doc_id[:8]}…`)")


def _show_overview(df: pd.DataFrame, date_col: str):
    """Quick overview when no product is selected yet."""
    st.info("👆 Select products and click **▶ Run Forecast** in the sidebar, or type a natural language query.")
    value_col = "units_sold" if "units_sold" in df.columns else df.select_dtypes("number").columns[0]
    if "product_name" in df.columns:
        agg = df.groupby("product_name")[value_col].sum().reset_index()
        agg.columns = ["Product", "Total Units Sold"]
        agg = agg.sort_values("Total Units Sold", ascending=False)
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(agg, use_container_width=True, hide_index=True)
        with c2:
            if "timestamp" in df.columns or date_col in df.columns:
                df["_dt"] = pd.to_datetime(df[date_col])
                fig = demand_trend_line(df, "_dt", value_col, "product_name",
                                        "Overall Demand Trend by Product")
                st.plotly_chart(fig, use_container_width=True)
