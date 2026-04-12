"""
charts.py — All visualization functions including 3D Plotly charts.
Each function returns a plotly Figure ready for st.plotly_chart().
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Color palette (matches SKVision dark theme) ────────────────────────────
CYAN    = "#00d4ff"
AMBER   = "#ffb830"
GREEN   = "#00e5a0"
RED     = "#ff4757"
PURPLE  = "#9b8cff"
PINK    = "#ff6eb4"
PALETTE = [CYAN, AMBER, GREEN, RED, PURPLE, PINK, "#ffd700", "#7fffd4"]

DARK_LAYOUT = dict(
    paper_bgcolor="rgba(8,12,22,0.0)",
    plot_bgcolor="rgba(8,12,22,0.0)",
    font=dict(color="#c5d8f0", family="IBM Plex Mono, monospace", size=11),
    legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
    margin=dict(l=50, r=30, t=50, b=50),
    colorway=PALETTE,
)


def _apply_dark(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(**DARK_LAYOUT, title=dict(text=title, font=dict(color=CYAN, size=14)))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.08)")
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  TIME-SERIES CHARTS
# ════════════════════════════════════════════════════════════════════════════

def demand_trend_line(df: pd.DataFrame, date_col: str, value_col: str,
                      group_col: str | None = None, title: str = "Demand Trend") -> go.Figure:
    """Multi-series line chart with rolling-average overlay."""
    fig = go.Figure()
    groups = df[group_col].unique() if group_col else [None]
    for i, grp in enumerate(groups):
        sub = df[df[group_col] == grp] if group_col else df
        sub = sub.sort_values(date_col)
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(
            x=sub[date_col], y=sub[value_col],
            mode="lines", name=str(grp) if grp else value_col,
            line=dict(color=color, width=1.8), opacity=0.85
        ))
        # 7-day rolling average
        ra = sub[value_col].rolling(7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=sub[date_col], y=ra,
            mode="lines", name=f"{grp} 7d MA" if grp else "7d MA",
            line=dict(color=color, width=2.5, dash="dot"), opacity=0.6
        ))
    return _apply_dark(fig, title)

def _safe_vline(fig, x, color="rgba(255,184,48,0.4)", text="Forecast →"):
    """Generic safe vline that works with datetime, int, float, string."""
    import pandas as pd
    import numpy as np
    from datetime import datetime

    # normalize value
    if isinstance(x, (pd.Timestamp, np.datetime64, datetime)):
        x_val = pd.to_datetime(x).to_pydatetime()
    elif isinstance(x, (int, float, np.number)):
        x_val = float(x)
    else:
        # fallback → string axis
        x_val = str(x)

    # use shape instead of add_vline (avoids Plotly bug)
    fig.add_shape(
        type="line",
        x0=x_val,
        x1=x_val,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(
            color=color,
            width=1,
            dash="dot"
        )
    )

    fig.add_annotation(
        x=x_val,
        y=1,
        yref="paper",
        text=text,
        showarrow=False,
        yshift=10,
        font=dict(color="#FFB830", size=11)
    )

def forecast_band_chart(hist_dates, hist_vals, fc_dates, fc_vals,
                        upper=None, lower=None, product="", conf=0.95) -> go.Figure:
    """Historical line + forecast band with confidence interval."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_vals, name="Historical",
        mode="lines", line=dict(color=CYAN, width=2),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.05)"
    ))

    fig.add_trace(go.Scatter(
        x=fc_dates, y=fc_vals,
        name=f"Forecast ({int(conf*100)}% CI)",
        mode="lines",
        line=dict(color=GREEN, width=2, dash="dash")
    ))

    if upper is not None and lower is not None:
        fig.add_trace(go.Scatter(
            x=list(fc_dates) + list(fc_dates)[::-1],
            y=list(upper) + list(lower)[::-1],
            fill="toself",
            fillcolor="rgba(0,229,160,0.08)",
            line=dict(color="rgba(0,229,160,0.2)"),
            name="Confidence Band"
        ))

    # Vertical separator (FIXED)
    if len(hist_dates):
        x_val = hist_dates[-1]
        if len(hist_dates):
            _safe_vline(fig, hist_dates[-1])

    return _apply_dark(fig, f"Demand Forecast — {product}")

def anomaly_scatter(df: pd.DataFrame, date_col: str, value_col: str,
                    anomaly_mask: pd.Series, title="Anomaly Detection") -> go.Figure:
    normal = df[~anomaly_mask]
    outlier = df[anomaly_mask]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal[date_col], y=normal[value_col],
        mode="lines+markers", name="Normal",
        marker=dict(size=3, color=CYAN), line=dict(color=CYAN, width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=outlier[date_col], y=outlier[value_col],
        mode="markers", name="Anomaly",
        marker=dict(size=9, color=RED, symbol="x-open", line=dict(width=2, color=RED))
    ))
    return _apply_dark(fig, title)


def seasonal_decomposition_chart(dates, trend, seasonal, residual, title="Seasonal Decomposition") -> go.Figure:
    fig = make_subplots(rows=3, cols=1, subplot_titles=["Trend", "Seasonal", "Residual"],
                        shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=dates, y=trend, name="Trend", line=dict(color=CYAN, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=seasonal, name="Seasonal",
                             fill="tozeroy", fillcolor="rgba(155,140,255,0.1)",
                             line=dict(color=PURPLE, width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=residual, name="Residual",
                             mode="markers", marker=dict(color=AMBER, size=3)), row=3, col=1)
    fig.update_layout(**DARK_LAYOUT, title=dict(text=title, font=dict(color=CYAN, size=14)), height=520)
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  3D CHARTS
# ════════════════════════════════════════════════════════════════════════════

def demand_3d_surface(df: pd.DataFrame, date_col: str, product_col: str, value_col: str,
                      title="3D Demand Surface") -> go.Figure:
    """3D surface: x=time buckets, y=product, z=demand value."""
    df = df.copy()
    df["_month"] = pd.to_datetime(df[date_col]).dt.to_period("M").astype(str)
    pivot = df.pivot_table(index=product_col, columns="_month", values=value_col, aggfunc="sum").fillna(0)
    z = pivot.values
    x = list(range(z.shape[1]))
    y = list(range(z.shape[0]))
    fig = go.Figure(go.Surface(
        z=z, x=x, y=y,
        colorscale=[[0, "#06080f"], [0.33, PURPLE], [0.66, CYAN], [1, GREEN]],
        opacity=0.85,
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor=AMBER, project=dict(z=True)))
    ))
    fig.update_layout(
        **DARK_LAYOUT, title=dict(text=title, font=dict(color=CYAN, size=14)), height=520,
        scene=dict(
            xaxis=dict(title="Time Period", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Product", tickvals=y, ticktext=list(pivot.index),
                       gridcolor="rgba(255,255,255,0.05)"),
            zaxis=dict(title=value_col.replace("_", " ").title(), gridcolor="rgba(255,255,255,0.05)"),
            bgcolor="rgba(6,8,15,0.0)",
        )
    )
    return fig


def risk_3d_scatter(df: pd.DataFrame, x_col: str, y_col: str, z_col: str,
                    color_col: str | None = None, label_col: str | None = None,
                    title="3D Risk Scatter") -> go.Figure:
    """3D scatter for multi-dimensional risk mapping."""
    color_vals = df[color_col] if color_col and color_col in df.columns else None
    text_vals  = df[label_col].astype(str) if label_col and label_col in df.columns else None
    fig = go.Figure(go.Scatter3d(
        x=df[x_col], y=df[y_col], z=df[z_col],
        mode="markers+text",
        text=text_vals,
        marker=dict(
            size=7,
            color=color_vals if color_vals is not None else CYAN,
            colorscale=[[0, GREEN], [0.5, AMBER], [1, RED]],
            showscale=True if color_vals is not None else False,
            colorbar=dict(title=color_col, tickfont=dict(color="#c5d8f0")),
            opacity=0.85,
            line=dict(width=0.5, color="rgba(255,255,255,0.15)")
        ),
        textfont=dict(color="#c5d8f0", size=9)
    ))
    fig.update_layout(
        **DARK_LAYOUT, title=dict(text=title, font=dict(color=CYAN)), height=520,
        scene=dict(
            xaxis=dict(title=x_col.replace("_", " ").title(), gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title=y_col.replace("_", " ").title(), gridcolor="rgba(255,255,255,0.05)"),
            zaxis=dict(title=z_col.replace("_", " ").title(), gridcolor="rgba(255,255,255,0.05)"),
            bgcolor="rgba(6,8,15,0.0)",
        )
    )
    return fig


import plotly.graph_objects as go
import pandas as pd

def supplier_3d_bar(df: pd.DataFrame, supplier_col: str, product_col: str,
                   value_col: str, title="3D Supplier Performance") -> go.Figure:
    
    pivot = df.pivot_table(
        index=supplier_col,
        columns=product_col,
        values=value_col,
        aggfunc="mean"
    ).fillna(0)

    suppliers = list(pivot.index)
    products = list(pivot.columns)

    x, y, z, text = [], [], [], []

    for i, sup in enumerate(suppliers):
        for j, prod in enumerate(products):
            val = pivot.loc[sup, prod]
            x.append(i)
            y.append(j)
            z.append(val)
            text.append(f"{sup}<br>{prod}<br>{val:.2f}")

    fig = go.Figure(data=[
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=10,
                color=z,
                colorscale=[[0, "red"], [0.5, "orange"], [1, "green"]],
                opacity=0.9,
                colorbar=dict(title=value_col)
            ),
            text=text,
            hoverinfo='text'
        )
    ])

    fig.update_layout(
        title=title,
        height=550,
        scene=dict(
            xaxis=dict(
                title="Supplier",
                tickvals=list(range(len(suppliers))),
                ticktext=suppliers
            ),
            yaxis=dict(
                title="Product",
                tickvals=list(range(len(products))),
                ticktext=products
            ),
            zaxis=dict(
                title=value_col.replace("_", " ").title()
            )
        )
    )

    return fig


def route_3d_path(df: pd.DataFrame, lat_col="vehicle_gps_latitude",
                  lon_col="vehicle_gps_longitude", z_col="route_risk_level",
                  color_col="risk_classification", title="3D Route Map") -> go.Figure:
    """3D path of vehicle GPS routes colored by risk."""
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    color_map = {"Low": GREEN, "Medium": AMBER, "High": RED}
    colors = df[color_col].map(color_map).fillna(CYAN) if color_col in df.columns else CYAN
    fig = go.Figure(go.Scatter3d(
        x=df[lon_col], y=df[lat_col], z=df[z_col] if z_col in df.columns else [0]*len(df),
        mode="markers+lines",
        marker=dict(size=4, color=list(colors), opacity=0.8,
                    line=dict(width=0.3, color="rgba(255,255,255,0.1)")),
        line=dict(color="rgba(0,212,255,0.3)", width=1),
    ))
    fig.update_layout(
        **DARK_LAYOUT, title=dict(text=title, font=dict(color=CYAN)), height=520,
        scene=dict(
            xaxis=dict(title="Longitude", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Latitude",  gridcolor="rgba(255,255,255,0.05)"),
            zaxis=dict(title="Risk Level", gridcolor="rgba(255,255,255,0.05)"),
            bgcolor="rgba(6,8,15,0.0)",
        )
    )
    return fig


def inventory_3d_heatmap(df: pd.DataFrame, product_col: str, warehouse_col: str,
                         value_col: str, title="Inventory Heat Map 3D") -> go.Figure:
    """3D heatmap surface for product × warehouse stock levels."""
    pivot = df.pivot_table(index=product_col, columns=warehouse_col,
                           values=value_col, aggfunc="mean").fillna(0)
    fig = go.Figure(go.Surface(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale=[[0, RED], [0.4, AMBER], [1, GREEN]],
        opacity=0.9,
        showscale=True,
        colorbar=dict(title=value_col, tickfont=dict(color="#c5d8f0"))
    ))
    fig.update_layout(
        **DARK_LAYOUT, title=dict(text=title, font=dict(color=CYAN)), height=520,
        scene=dict(
            xaxis=dict(title="Warehouse", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Product",   gridcolor="rgba(255,255,255,0.05)"),
            zaxis=dict(title=value_col.replace("_", " ").title(), gridcolor="rgba(255,255,255,0.05)"),
            bgcolor="rgba(6,8,15,0.0)",
        )
    )
    return fig


def lead_time_3d(df: pd.DataFrame, supplier_col: str, product_col: str,
                 x_col="ideal_supply_time_days", y_col="actual_supply_time_days",
                 z_col="supplier_reliability_score", title="Lead Time 3D Analysis") -> go.Figure:
    """3D bubble: ideal vs actual supply time vs reliability."""
    color_vals = df[z_col] if z_col in df.columns else None
    text_vals  = (df[supplier_col] + " / " + df[product_col]).astype(str)
    fig = go.Figure(go.Scatter3d(
        x=df[x_col], y=df[y_col],
        z=df[z_col] if z_col in df.columns else [1]*len(df),
        mode="markers",
        text=text_vals,
        marker=dict(
            size=8,
            color=color_vals if color_vals is not None else CYAN,
            colorscale=[[0, RED], [0.5, AMBER], [1, GREEN]],
            showscale=True,
            colorbar=dict(title="Reliability", tickfont=dict(color="#c5d8f0")),
            opacity=0.85
        )
    ))
    # Diagonal parity line
    mx = max(df[x_col].max(), df[y_col].max())
    fig.add_trace(go.Scatter3d(
        x=[0, mx], y=[0, mx], z=[0.5, 0.5], mode="lines",
        line=dict(color=AMBER, width=3, dash="dash"), name="On-time parity"
    ))
    fig.update_layout(
        **DARK_LAYOUT, title=dict(text=title, font=dict(color=CYAN)), height=520,
        scene=dict(
            xaxis=dict(title="Ideal Days", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Actual Days", gridcolor="rgba(255,255,255,0.05)"),
            zaxis=dict(title="Reliability", gridcolor="rgba(255,255,255,0.05)"),
            bgcolor="rgba(6,8,15,0.0)",
        )
    )
    return fig


def demand_seasonality_3d(df: pd.DataFrame, date_col: str, product_col: str,
                           value_col: str, title="Seasonality 3D") -> go.Figure:
    """3D surface: month × hour-of-week (or day-of-week) × avg demand."""
    df = df.copy()
    df["_dt"]    = pd.to_datetime(df[date_col])
    df["_month"] = df["_dt"].dt.month
    df["_dow"]   = df["_dt"].dt.dayofweek
    pivot = df.pivot_table(index="_month", columns="_dow", values=value_col, aggfunc="mean").fillna(0)
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    day_names   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    fig = go.Figure(go.Surface(
        z=pivot.values,
        x=[day_names[i] for i in pivot.columns],
        y=[month_names[m-1] for m in pivot.index],
        colorscale=[[0,"#06080f"],[0.3,PURPLE],[0.65,CYAN],[1,AMBER]],
        opacity=0.88, showscale=True,
        colorbar=dict(title="Avg Demand", tickfont=dict(color="#c5d8f0"))
    ))
    fig.update_layout(
        **DARK_LAYOUT, title=dict(text=title, font=dict(color=CYAN)), height=520,
        scene=dict(
            xaxis=dict(title="Day of Week", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Month",       gridcolor="rgba(255,255,255,0.05)"),
            zaxis=dict(title="Avg Units",   gridcolor="rgba(255,255,255,0.05)"),
            bgcolor="rgba(6,8,15,0.0)",
        )
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  2D UTILITY CHARTS
# ════════════════════════════════════════════════════════════════════════════

def risk_gauge(score: float, title="Risk Score") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"color": CYAN, "size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#7a94b8"},
            "bar":  {"color": RED if score > 65 else AMBER if score > 40 else GREEN},
            "steps": [
                {"range": [0, 40],  "color": "rgba(0,229,160,0.1)"},
                {"range": [40, 65], "color": "rgba(255,184,48,0.1)"},
                {"range": [65,100], "color": "rgba(255,71,87,0.1)"},
            ],
            "threshold": {"line": {"color": AMBER, "width": 3}, "thickness": 0.75, "value": score},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(255,255,255,0.08)",
        },
        number={"font": {"color": "#c5d8f0"}},
    ))
    fig.update_layout(**DARK_LAYOUT, height=260)
    return fig


def heatmap_correlation(df: pd.DataFrame, cols: list[str], title="Correlation Matrix") -> go.Figure:
    corr = df[cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0, RED], [0.5, "rgba(20,26,46,1)"], [1, GREEN]],
        text=corr.round(2).values, texttemplate="%{text}",
        colorbar=dict(tickfont=dict(color="#c5d8f0"))
    ))
    return _apply_dark(fig, title)


def bar_comparison(df: pd.DataFrame, x_col: str, y_col: str,
                   color_col: str | None = None, title="Comparison") -> go.Figure:
    fig = px.bar(df, x=x_col, y=y_col, color=color_col,
                 color_discrete_sequence=PALETTE)
    return _apply_dark(fig, title)


def pie_distribution(df: pd.DataFrame, names_col: str, values_col: str,
                     title="Distribution") -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=df[names_col], values=df[values_col],
        marker=dict(colors=PALETTE, line=dict(color="rgba(0,0,0,0.3)", width=1)),
        textfont=dict(color="#c5d8f0"), hole=0.4
    ))
    return _apply_dark(fig, title)


def waterfall_chart(labels: list, values: list, title="Waterfall") -> go.Figure:
    fig = go.Figure(go.Waterfall(
        name="", orientation="v",
        x=labels, y=values,
        textposition="outside",
        connector={"line": {"color": "rgba(255,255,255,0.1)"}},
        increasing={"marker": {"color": GREEN}},
        decreasing={"marker": {"color": RED}},
        totals={"marker": {"color": CYAN}},
    ))
    return _apply_dark(fig, title)


def sankey_flow(labels: list, source: list, target: list, value: list,
                title="Supply Flow") -> go.Figure:
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=18,
            line=dict(color="rgba(255,255,255,0.1)", width=0.5),
            label=labels,
            color=[PALETTE[i % len(PALETTE)] for i in range(len(labels))],
        ),
        link=dict(source=source, target=target, value=value,
                  color="rgba(0,212,255,0.12)")
    ))
    return _apply_dark(fig, title)


def candlestick_demand(df: pd.DataFrame, date_col: str, open_col: str,
                       high_col: str, low_col: str, close_col: str,
                       title="Demand Candlestick") -> go.Figure:
    fig = go.Figure(go.Candlestick(
        x=df[date_col], open=df[open_col], high=df[high_col],
        low=df[low_col], close=df[close_col],
        increasing_line_color=GREEN, decreasing_line_color=RED
    ))
    return _apply_dark(fig, title)
