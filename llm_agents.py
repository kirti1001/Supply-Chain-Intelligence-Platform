"""
llm_agents.py — LangChain + Groq multi-agent orchestration.
Agents: Orchestrator → Preprocess → Chart → Forecast → Risk → Inventory
        → Seasonal → Stockout → Supplier → Report
"""

from __future__ import annotations
import json
import re
import streamlit as st
from typing import Any
from settings import get_groq_client, get_credentials, GROQ_CHAT_MODEL, GROQ_VISION_MODEL


# ─── Low-level Groq call ─────────────────────────────────────────────────────
def llm_call(messages: list[dict], model: str = GROQ_CHAT_MODEL,
             temperature: float = 0.1, max_tokens: int = 3000,
             json_mode: bool = False) -> str:
    client = get_groq_client()
    if not client:
        return "⚠ No Groq API key configured. Go to ⚙️ Settings."
    kwargs: dict[str, Any] = dict(model=model, messages=messages,
                                  temperature=temperature, max_tokens=max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"⚠ LLM Error: {e}"


def parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response robustly."""
    text = text.strip()
    # Remove markdown fences
    text = re.sub(r"```json?", "", text).replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        # Try finding JSON block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {"raw": text}


# ─── Web search (Tavily) ────────────────────────────────────────────────────
def web_search(query: str, max_results: int = 5) -> str:
    """Search web for seasonal events, market context. Falls back gracefully."""
    creds = get_credentials()
    key = creds.get("tavily_api_key", "")
    if not key:
        return "Web search not available (no Tavily key). Using offline knowledge."
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=key)
        results = client.search(query, max_results=max_results)
        snippets = []
        for r in results.get("results", [])[:max_results]:
            snippets.append(f"• {r.get('title','')} — {r.get('content','')[:200]}")
        return "\n".join(snippets) if snippets else "No results found."
    except Exception as e:
        return f"Search error: {e}"


# ─── System prompt builder ───────────────────────────────────────────────────
def _build_system(role: str, extra: str = "") -> str:
    return f"""You are **{role}**, an expert supply chain intelligence agent.
You analyze real supply chain datasets (demand, inventory, supplier, transport).
Always be data-driven, precise, and actionable. Use numbers from context.
Respond in structured JSON unless asked otherwise. No hallucinated data.
{extra}"""


# ════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR AGENT — routes NL query to correct sub-agent
# ════════════════════════════════════════════════════════════════════════════

MODULES = ["demand_forecast", "risk_assessment", "inventory_management",
           "seasonality", "stockout_prediction", "supplier_risk",
           "route_optimization", "report", "chat"]

def orchestrate(user_query: str, context: str = "") -> dict:
    """Parse NL query and decide which module/agent to invoke."""
    prompt = f"""Given this user query, decide which SCM module to invoke.
Available modules: {MODULES}
User query: "{user_query}"
Context: {context}

Respond ONLY with JSON:
{{
  "module": "<module_name>",
  "intent": "<one-line description>",
  "params": {{"key": "value", ...}},
  "confidence": 0.0-1.0
}}"""
    raw = llm_call([{"role": "user", "content": prompt}], json_mode=True)
    result = parse_json_response(raw)
    result.setdefault("module", "chat")
    return result


# ════════════════════════════════════════════════════════════════════════════
#  DEMAND FORECAST AGENT
# ════════════════════════════════════════════════════════════════════════════

def agent_interpret_forecast_query(user_query: str, available_products: list,
                                   available_cols: list) -> dict:
    """Parse NL forecast query into structured parameters."""
    prompt = f"""Parse this demand forecasting query into parameters.
Query: "{user_query}"
Available products: {available_products}
Available columns: {available_cols}

Respond in JSON:
{{
  "products": ["list of product names from available_products"],
  "horizon_days": <integer, default 30>,
  "analysis_window_days": <integer, default 90>,
  "method": "prophet|arima|ets|ensemble",
  "confidence_level": 0.95,
  "include_events": true,
  "include_seasonality": true,
  "group_by": "region|category|null",
  "explanation": "what you understood"
}}"""
    raw = llm_call([{"role": "user", "content": prompt}], json_mode=True)
    return parse_json_response(raw)


def agent_generate_forecast_insights(product: str, stats: dict,
                                     forecast_values: list, search_context: str = "") -> str:
    """Generate LLM narrative for a demand forecast result."""
    prompt = f"""Generate a professional demand forecast analysis report section.
Product: {product}
Historical stats: {json.dumps(stats, default=str)[:800]}
Forecast (next {len(forecast_values)} days): mean={round(sum(forecast_values)/len(forecast_values),1)}, max={max(forecast_values)}, min={min(forecast_values)}
External context: {search_context[:500]}

Provide:
1. Trend interpretation
2. Key demand drivers identified
3. Seasonal/event impacts
4. Inventory implications
5. 3 specific action recommendations

Be precise with numbers. Max 350 words."""
    return llm_call([
        {"role": "system", "content": _build_system("Demand Forecast Analyst")},
        {"role": "user", "content": prompt}
    ], temperature=0.2)


# ════════════════════════════════════════════════════════════════════════════
#  RISK ASSESSMENT AGENT
# ════════════════════════════════════════════════════════════════════════════

def agent_risk_analysis(risk_data: dict, user_query: str = "") -> dict:
    """Generate structured risk analysis from computed metrics."""
    prompt = f"""Analyze this supply chain risk data and generate a comprehensive risk report.
Data: {json.dumps(risk_data, default=str)[:1500]}
User focus: {user_query or "comprehensive risk overview"}

Respond in JSON:
{{
  "overall_risk_score": <0-100>,
  "risk_level": "Low|Medium|High|Critical",
  "risks": [
    {{"name": "...", "score": 0-100, "category": "supplier|logistics|demand|inventory",
      "description": "...", "impact": "...", "mitigation": "...", "urgency": "immediate|1-week|1-month"}}
  ],
  "top_vulnerabilities": ["...", "..."],
  "immediate_actions": ["...", "..."],
  "risk_trend": "improving|stable|deteriorating",
  "summary": "3-sentence executive summary"
}}"""
    raw = llm_call([
        {"role": "system", "content": _build_system("Supply Chain Risk Manager")},
        {"role": "user", "content": prompt}
    ], json_mode=True)
    return parse_json_response(raw)


# ════════════════════════════════════════════════════════════════════════════
#  INVENTORY AGENT
# ════════════════════════════════════════════════════════════════════════════

def agent_inventory_insights(inventory_data: dict, forecast_data: dict | None,
                              user_query: str = "") -> dict:
    """Generate inventory optimization recommendations using forecast."""
    fc_info = ""
    if forecast_data:
        fv = forecast_data.get("result", {}).get("forecast_values", [])
        if fv:
            fc_info = f"Recent demand forecast (next {len(fv)} days): mean={round(sum(fv)/len(fv),1)}, max={max(fv)}"

    prompt = f"""Optimize inventory based on this data.
Inventory data: {json.dumps(inventory_data, default=str)[:1000]}
Forecast context: {fc_info}
Query: {user_query or "optimize inventory levels"}

Respond in JSON:
{{
  "reorder_recommendations": [
    {{"product": "...", "warehouse": "...", "current_stock": 0, "reorder_point": 0,
      "suggested_order_qty": 0, "urgency": "immediate|soon|normal", "reason": "..."}}
  ],
  "safety_stock": {{"product": 0, "formula": "...", "rationale": "..."}},
  "eoq": {{"product": 0, "eoq_units": 0, "cycle_days": 0}},
  "overstock_alerts": [{{"product": "...", "warehouse": "...", "excess_units": 0, "action": "..."}}],
  "kpis": {{"inventory_turnover": 0, "days_of_supply": 0, "fill_rate": 0}},
  "summary": "2-sentence summary"
}}"""
    raw = llm_call([
        {"role": "system", "content": _build_system("Inventory Optimization Specialist")},
        {"role": "user", "content": prompt}
    ], json_mode=True)
    return parse_json_response(raw)


# ════════════════════════════════════════════════════════════════════════════
#  SEASONALITY & STOCKOUT AGENT
# ════════════════════════════════════════════════════════════════════════════

def agent_seasonality_analysis(product: str, seasonal_stats: dict,
                                search_context: str = "") -> dict:
    prompt = f"""Analyze seasonality and external events for {product}.
Statistical patterns: {json.dumps(seasonal_stats, default=str)[:800]}
External context (web search): {search_context[:600]}

Respond in JSON:
{{
  "peak_months": ["..."],
  "low_months": ["..."],
  "peak_events": [{{"event": "...", "typical_lift": "X%", "duration_days": 0, "preparation_lead_days": 0}}],
  "seasonal_index": {{"Jan": 1.0, "Feb": 1.0, ...}},
  "yoy_growth": "...",
  "recommendations": ["...", "..."],
  "summary": "..."
}}"""
    raw = llm_call([
        {"role": "system", "content": _build_system("Seasonality & Demand Pattern Analyst")},
        {"role": "user", "content": prompt}
    ], json_mode=True)
    return parse_json_response(raw)


def agent_stockout_prediction(product: str, stats: dict, inventory_status: dict) -> dict:
    prompt = f"""Predict stockout probability and timeline for {product}.
Demand stats: {json.dumps(stats, default=str)[:600]}
Current inventory: {json.dumps(inventory_status, default=str)[:600]}

Respond in JSON:
{{
  "stockout_probability_7d": 0.0-1.0,
  "stockout_probability_30d": 0.0-1.0,
  "estimated_days_to_stockout": 0,
  "risk_level": "Low|Medium|High|Critical",
  "contributing_factors": ["..."],
  "recommended_order_qty": 0,
  "recommended_order_by": "YYYY-MM-DD",
  "confidence": 0.0-1.0,
  "summary": "..."
}}"""
    raw = llm_call([
        {"role": "system", "content": _build_system("Stockout Prevention Specialist")},
        {"role": "user", "content": prompt}
    ], json_mode=True)
    return parse_json_response(raw)


# ════════════════════════════════════════════════════════════════════════════
#  SUPPLIER RISK AGENT
# ════════════════════════════════════════════════════════════════════════════

def agent_supplier_risk(supplier_name: str, metrics: dict, user_query: str = "") -> dict:
    prompt = f"""Assess supplier risk for {supplier_name}.
Performance metrics: {json.dumps(metrics, default=str)[:1000]}
Focus: {user_query or "comprehensive risk assessment"}

Respond in JSON:
{{
  "overall_risk_score": 0-100,
  "risk_categories": {{
    "delivery_reliability": 0-100,
    "quality_risk": 0-100,
    "concentration_risk": 0-100,
    "financial_risk": 0-100,
    "lead_time_risk": 0-100
  }},
  "performance_summary": {{"on_time_rate": 0.0, "fill_rate": 0.0, "avg_delay_days": 0.0, "damage_rate": 0.0}},
  "red_flags": ["..."],
  "strengths": ["..."],
  "recommended_actions": [{{"action": "...", "priority": "HIGH|MED|LOW", "timeline": "..."}}],
  "alternative_supplier_needed": true/false,
  "contract_recommendation": "...",
  "summary": "3-sentence summary"
}}"""
    raw = llm_call([
        {"role": "system", "content": _build_system("Supplier Risk & Procurement Analyst")},
        {"role": "user", "content": prompt}
    ], json_mode=True)
    return parse_json_response(raw)


# ════════════════════════════════════════════════════════════════════════════
#  ROUTE OPTIMIZATION AGENT
# ════════════════════════════════════════════════════════════════════════════

def agent_route_optimization(route_data: dict, user_query: str = "") -> dict:
    prompt = f"""Analyze transport routes and suggest optimizations.
Route metrics: {json.dumps(route_data, default=str)[:1200]}
Query: {user_query or "optimize routes to minimize cost and delay"}

Respond in JSON:
{{
  "high_risk_routes": [{{"route": "...", "risk_score": 0-100, "avg_delay_hrs": 0, "reason": "..."}}],
  "cost_optimization": [{{"route": "...", "current_cost": 0, "potential_saving": 0, "switch_to": "Road|Air|Sea"}}],
  "preferred_routes": [{{"from": "...", "to": "...", "mode": "...", "why": "..."}}],
  "delay_patterns": [{{"route_type": "...", "peak_delay_period": "...", "avg_delay_hrs": 0}}],
  "fuel_efficiency": {{"avg_consumption": 0, "high_consumption_routes": ["..."]}},
  "recommendations": ["...", "..."],
  "overall_logistics_score": 0-100,
  "summary": "..."
}}"""
    raw = llm_call([
        {"role": "system", "content": _build_system("Logistics & Route Optimization Analyst")},
        {"role": "user", "content": prompt}
    ], json_mode=True)
    return parse_json_response(raw)


# ════════════════════════════════════════════════════════════════════════════
#  REPORT GENERATION AGENT
# ════════════════════════════════════════════════════════════════════════════

def agent_generate_report(title: str, report_type: str, all_results: dict,
                           org: str = "", period: str = "") -> str:
    """Generate a beautiful Markdown report from all collected results."""
    context = json.dumps(all_results, default=str)[:3000]
    prompt = f"""Generate a comprehensive, professional supply chain intelligence report.
Organization: {org or "N/A"}
Period: {period or "Recent Analysis"}
Report Type: {report_type}
Title: {title}
All analysis results: {context}

Write a detailed Markdown report with:
1. Executive Summary (3-4 paragraphs with specific numbers)
2. Key Performance Indicators table
3. Demand Forecast Analysis
4. Risk Assessment (with severity ratings)
5. Inventory Status & Recommendations
6. Supplier Performance Summary
7. Logistics & Route Analysis
8. Strategic Recommendations (prioritized action plan)
9. Conclusion

Use **bold** for key metrics. Use tables where appropriate. Be specific with numbers."""
    return llm_call([
        {"role": "system", "content": _build_system("Chief Supply Chain Intelligence Officer",
            "Generate beautiful, detailed Markdown reports that executives can act on immediately.")},
        {"role": "user", "content": prompt}
    ], temperature=0.25, max_tokens=4000)


def agent_compare_reports(report_a: dict, report_b: dict, user_query: str) -> str:
    """Compare two reports and answer a specific query."""
    prompt = f"""Compare these two supply chain analysis reports and answer the query.
Report A (from {report_a.get('created_at','?')}): {json.dumps(report_a.get('content',{}), default=str)[:1500]}
Report B (from {report_b.get('created_at','?')}): {json.dumps(report_b.get('content',{}), default=str)[:1500]}
Query: {user_query}

Provide:
1. Direct answer to the query
2. Key differences between the two periods
3. Trend direction (improving/deteriorating)
4. Specific metric changes with % or absolute differences
5. Recommended focus areas based on the comparison"""
    return llm_call([
        {"role": "system", "content": _build_system("Senior Supply Chain Analytics Reviewer")},
        {"role": "user", "content": prompt}
    ], temperature=0.2)


# ════════════════════════════════════════════════════════════════════════════
#  GENERAL CHATBOT AGENT
# ════════════════════════════════════════════════════════════════════════════

def agent_chat(user_message: str, conversation_history: list[dict],
               data_context: str = "", db_context: str = "") -> str:
    system = _build_system(
        "SKVision Supply Chain AI Assistant",
        f"""You have access to supply chain data context:
{data_context[:800]}
Recent analysis results in database:
{db_context[:600]}
Answer questions about the supply chain data. If asked to perform analysis,
guide the user to the appropriate module. Be conversational but data-driven."""
    )
    messages = [{"role": "system", "content": system}] + conversation_history[-10:] + \
               [{"role": "user", "content": user_message}]
    return llm_call(messages, temperature=0.3)
