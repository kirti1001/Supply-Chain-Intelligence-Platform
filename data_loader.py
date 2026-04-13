"""
data_loader.py — SKVision Supply Chain Intelligence Platform
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REDESIGNED PIPELINE  v3.0  (Production-Ready)

NEW UPLOAD FLOW:
  Step 0  User uploads CSV(s) + optional TXT/JSON description file
  Step 1  ColumnProfilerAgent   — analyze column names & stats ONLY (no full scan)
  Step 2  DescriptionFusionAgent — fuse user description + column profiles via LLM
  Step 3  SmartCombinerAgent     — best-strategy multi-file merge using LLM plan
  Step 4  MissingFieldsAgent     — identify gaps, classify as critical/optional
  Step 5  UI GATE: show gaps → user can upload supplement CSV or skip
  Step 6  RequirementValidatorAgent — revalidate after any user supplement
  Step 7  FeatureEngineeringAgent   — safe auto-derivation of remaining gaps
  Step 8  DataAssistantChatbot      — conversational resolution of final gaps
  Step 9  SplitterAgent             — map master → 4 canonical datasets
  Step 10 FinalSave                 — disk + MongoDB (only finalized datasets)

DESIGN PRINCIPLES:
  • LLM receives ONLY column names + tiny sample (≤ 10 rows) — never full CSV
  • LLM calls are sequential, batched per stage — never overlapping
  • Each stage result is cached in session_state; re-running only touched stages
  • 4 canonical datasets are saved separately: demand / inventory / supplier / transport
  • Auto-restore from disk/DB on startup; no re-processing of unchanged datasets
  • Classic upload mode retained for backward compatibility

SAFETY:
  • All transforms via whitelisted safe ops only (no exec/eval)
  • LLM JSON responses validated before use
"""

from __future__ import annotations

import io
import os
import re
import json
import logging
import datetime
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from db_ops import save_result, load_all, COLL_UPLOADS

logger = logging.getLogger("data_loader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALL_REQUIRED_COLS: list[str] = [
    "timestamp", "product_name", "units_sold",
    "warehouse_id", "stock_units", "reorder_level", "inventory_status",
    "supplier_name", "supplier_id", "fulfillment_rate",
    "supplier_reliability_score", "supply_variation_days", "cargo_condition_status",
    "route_type", "delay_probability", "route_risk_level",
    "delivery_time_deviation", "fuel_consumption_rate",
    "shipping_costs", "lead_time_days", "risk_classification",
]

DERIVED_COLS: set[str] = {
    "demand_risk_scores", "inventory_risk_scores",
    "supplier_risk_scores", "transport_risk_scores",
}

# Per-module minimum columns
MODULE_REQUIREMENTS: dict[str, list[str]] = {
    "demand_forecast":      ["timestamp", "product_name", "units_sold"],
    "risk_assessment":      ["product_name"],
    "inventory_management": ["product_name", "stock_units"],
    "seasonality":          ["timestamp", "product_name", "units_sold"],
    "stockout_prediction":  ["product_name", "stock_units"],
    "supplier_risk":        ["supplier_name"],
    "route_optimization":   ["route_type"],
    "report":               ["product_name"],
}

# CRITICAL = blocks core modules; OPTIONAL = nice-to-have
CRITICAL_COLS: set[str] = {
    "timestamp", "product_name", "units_sold", "stock_units",
    "supplier_name", "route_type",
}

CANONICAL_DOMAINS: list[str] = ["demand", "inventory", "supplier", "transport"]

# Signature columns per domain (for classifier + saver)
SCHEMA_HINTS: dict[str, list[str]] = {
    "demand":    ["timestamp", "product_name", "units_sold"],
    "inventory": ["warehouse_id", "product_name", "stock_units", "reorder_level"],
    "supplier":  ["supplier_id", "product_name", "fulfillment_rate", "supplier_reliability_score"],
    "transport": ["timestamp", "product_name", "supplier_id", "warehouse_id", "route_type"],
}

PREPARED_DIR   = Path("dataset/prepared")
CANONICAL_DIR  = Path("dataset/canonical")

_JOIN_CANDIDATES: list[str] = [
    "product_name", "product_id", "sku", "item_name",
    "supplier_id", "supplier_name", "warehouse_id", "order_id",
    "timestamp", "date",
]

_SAFE_OPS: set[str] = {
    "rename", "fill_constant", "fill_from_col", "compute_ratio",
    "compute_diff", "compute_product", "clip", "to_lower",
    "map_values", "derive_status", "log1p", "rolling_mean", "date_extract",
}

_HEURISTICS: dict[str, list[str]] = {
    "timestamp":                  ["timestamp","date","time","dt","order_date","sale_date","trans_date","created_at"],
    "product_name":               ["product_name","product","item","sku","item_name","product_desc","article"],
    "units_sold":                 ["units_sold","quantity","qty","sales","demand","volume","sold","qty_sold","sales_qty"],
    "warehouse_id":               ["warehouse_id","warehouse","wh_id","depot","location_id","wh_code","store_id"],
    "stock_units":                ["stock_units","stock","inventory","on_hand","available","qty_on_hand","closing_stock","balance_qty"],
    "reorder_level":              ["reorder_level","reorder","min_stock","min_qty","safety_stock","reorder_point","rop"],
    "inventory_status":           ["inventory_status","stock_status","status","stock_level","inv_status"],
    "supplier_name":              ["supplier_name","supplier","vendor","vendor_name","supplier_company"],
    "supplier_id":                ["supplier_id","supplier_code","vendor_id","sup_id"],
    "fulfillment_rate":           ["fulfillment_rate","fill_rate","order_fill","service_level","fulfilment_rate"],
    "supplier_reliability_score": ["supplier_reliability_score","reliability","reliability_score","perf_score","vendor_score"],
    "supply_variation_days":      ["supply_variation_days","lead_variance","delay_days","supply_delay","variation_days"],
    "cargo_condition_status":     ["cargo_condition_status","cargo_condition","condition","quality_status","damage_status"],
    "route_type":                 ["route_type","mode","transport_mode","shipment_mode","carrier_type","delivery_mode"],
    "delay_probability":          ["delay_probability","delay_prob","on_time_risk","delay_risk","prob_delay"],
    "route_risk_level":           ["route_risk_level","risk_level","route_risk","risk_score","route_score"],
    "delivery_time_deviation":    ["delivery_time_deviation","delay","time_deviation","schedule_deviation","late_days"],
    "fuel_consumption_rate":      ["fuel_consumption_rate","fuel_rate","fuel","fuel_consumption","fuel_used"],
    "shipping_costs":             ["shipping_costs","freight","shipping","transport_cost","logistics_cost","freight_cost"],
    "lead_time_days":             ["lead_time_days","lead_time","delivery_days","transit_days","procurement_days"],
    "risk_classification":        ["risk_classification","risk_class","risk_category","risk_tier","risk_label"],
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SAFE LLM WRAPPER  (sequential, batched, never blocking)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _llm(messages: list[dict], json_mode: bool = False, max_tokens: int = 1500) -> str:
    try:
        from llm_agents import llm_call
        return llm_call(messages, json_mode=json_mode, max_tokens=max_tokens)
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return "{}"


def _parse(text: str) -> dict:
    try:
        from llm_agents import parse_json_response
        return parse_json_response(text)
    except Exception:
        pass
    text = re.sub(r"```json?", "", text).replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {}


def _ts() -> str:
    return datetime.datetime.utcnow().isoformat()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 0 — Basic cleaning
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if any(kw in col.lower() for kw in ("date", "time", "stamp", "_dt", "created", "updated")):
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
            except Exception:
                pass
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]", "_", regex=True)
    )
    df = _parse_dates(df)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


@st.cache_data(show_spinner=False)
def load_csv_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    return _clean(df)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 1 — ColumnProfilerAgent
#  LLM receives ONLY: column names + dtypes + tiny sample (≤ 10 rows)
#  NO full-dataset statistics sent to LLM — keeps token usage minimal
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ColumnProfilerAgent(df: pd.DataFrame, filename: str = "", user_description: str = "") -> dict:
    """
    LLM-efficient column profiling. Sends only column metadata (not full data).
    Returns structured profile with domain classification and column mappings.
    """
    logger.info("[ColumnProfiler] %s — %d cols", filename, df.shape[1])

    # Lightweight column metadata — NO full stats
    col_meta: list[dict] = []
    for col in df.columns:
        s = df[col]
        null_pct = round(float(s.isnull().mean() * 100), 1)
        sample = [str(v) for v in s.dropna().head(5).tolist()]  # max 5 samples

        if pd.api.types.is_numeric_dtype(s):
            kind = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(s):
            kind = "datetime"
        else:
            kind = "categorical"

        col_meta.append({
            "col": col,
            "type": kind,
            "null_pct": null_pct,
            "sample": sample,
        })

    # Single LLM call per file — compact prompt
    prompt = f"""Supply chain data analyst. Analyze these columns and classify this dataset.

File: "{filename}"
User description: "{user_description or 'none provided'}"
Rows: {len(df)}, Columns: {df.shape[1]}

Column metadata (name, type, null%, sample values):
{json.dumps(col_meta, default=str)[:2500]}

Return ONLY valid JSON:
{{
  "domain": "demand|inventory|supplier|transport|mixed",
  "org_type": "retail|manufacturing|pharma|logistics|ecommerce|generic",
  "summary": "one line description",
  "has_timeseries": true,
  "likely_join_keys": ["col_name"],
  "mappings": [
    {{"col": "original_col", "maps_to": "required_col_or_null", "confidence": 0.0}}
  ]
}}

Required columns to map to: {ALL_REQUIRED_COLS}"""

    raw = _parse(_llm([{"role": "user", "content": prompt}], json_mode=True, max_tokens=1200))

    return {
        "filename": filename,
        "rows": len(df),
        "cols": df.shape[1],
        "columns": df.columns.tolist(),
        "col_meta": col_meta,
        "llm": raw,
        "profiled_at": _ts(),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 2 — SchemaMapperAgent
#  Heuristic + LLM fused column → required mapping
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def SchemaMapperAgent(profile: dict) -> dict:
    """
    Map original column names → required column names.
    Pass 1: LLM suggestions ≥ 0.65 confidence
    Pass 2: Heuristic keyword match
    Pass 3: Exact name match (highest priority)
    """
    orig_cols: list[str] = profile.get("columns", [])
    llm_maps:  list[dict] = profile.get("llm", {}).get("mappings", [])

    mapping:    dict[str, str | None] = {c: None for c in ALL_REQUIRED_COLS}
    confidence: dict[str, float]      = {c: 0.0  for c in ALL_REQUIRED_COLS}

    lower_orig = {c.lower(): c for c in orig_cols}

    # Pass 1 — LLM suggestions
    for m in llm_maps:
        orig = m.get("col", "")
        req  = m.get("maps_to")
        conf = float(m.get("confidence", 0.0))
        if req and req in mapping and conf >= 0.65 and orig in orig_cols:
            if conf > confidence[req]:
                mapping[req]    = orig
                confidence[req] = conf

    # Pass 2 — Heuristic fuzzy
    for req_col, keywords in _HEURISTICS.items():
        if mapping[req_col] is not None:
            continue
        for kw in keywords:
            if kw in lower_orig:
                mapping[req_col]    = lower_orig[kw]
                confidence[req_col] = 0.70
                break
            for lo, orig_c in lower_orig.items():
                if kw in lo or lo in kw:
                    mapping[req_col]    = orig_c
                    confidence[req_col] = 0.60
                    break
            if mapping[req_col]:
                break

    # Pass 3 — Exact match (override)
    for req_col in ALL_REQUIRED_COLS:
        if req_col in lower_orig:
            mapping[req_col]    = lower_orig[req_col]
            confidence[req_col] = 1.0

    mapped = sum(1 for v in mapping.values() if v is not None)
    return {
        "mapping":       mapping,
        "confidence":    confidence,
        "mapped_count":  mapped,
        "total_required": len(ALL_REQUIRED_COLS),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 3 — DescriptionFusionAgent
#  Parse optional user-uploaded description file (TXT/JSON)
#  and fuse with column profiles for richer context
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def DescriptionFusionAgent(
    profiles: list[dict],
    user_text_description: str = "",
    dataset_description_json: str = "",
) -> dict:
    """
    ONE LLM call to fuse all file profiles + user description into a
    global context document. Used to guide downstream agents.
    Sends only column names and metadata — no row data.
    """
    if not user_text_description and not dataset_description_json:
        return {"context": "", "org_type": "generic", "domain_map": {}}

    summary_per_file = []
    for p in profiles:
        summary_per_file.append({
            "file": p.get("filename"),
            "cols": p.get("columns", []),
            "domain": p.get("llm", {}).get("domain", "unknown"),
            "summary": p.get("llm", {}).get("summary", ""),
        })

    desc = user_text_description or ""
    if dataset_description_json:
        try:
            parsed_desc = json.loads(dataset_description_json)
            # Extract dataset feature names to aid mapping
            for ds_item in parsed_desc.get("datasets", []):
                desc += f"\nDataset '{ds_item.get('dataset_name','')}': "
                desc += ", ".join(f.get("name","") for f in ds_item.get("features", []))
        except Exception:
            desc += "\n" + dataset_description_json[:1500]

    prompt = f"""You are a supply chain data integration expert.

User-provided description:
{desc[:2000]}

Files uploaded (column names only):
{json.dumps(summary_per_file, default=str)[:2000]}

Based on the description and column names, return ONLY valid JSON:
{{
  "org_type": "retail|manufacturing|pharma|logistics|ecommerce|generic",
  "global_context": "2-3 sentence summary of what this data covers",
  "domain_map": {{"filename": "demand|inventory|supplier|transport|mixed"}},
  "key_entities": ["product", "supplier", "warehouse"],
  "join_strategy": "how files should be joined",
  "missing_hints": ["columns that seem absent but are described in the docs"]
}}"""

    raw = _parse(_llm([{"role": "user", "content": prompt}], json_mode=True, max_tokens=800))
    return {
        "context":    raw.get("global_context", ""),
        "org_type":   raw.get("org_type", "generic"),
        "domain_map": raw.get("domain_map", {}),
        "join_strategy": raw.get("join_strategy", ""),
        "missing_hints": raw.get("missing_hints", []),
        "key_entities": raw.get("key_entities", []),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 4 — SmartCombinerAgent
#  LLM plans the merge strategy; Python executes it safely
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _find_shared_keys(dfs: dict[str, pd.DataFrame]) -> list[str]:
    from collections import Counter
    col_counter: Counter = Counter()
    for df in dfs.values():
        col_counter.update(set(df.columns.str.lower()))
    shared = {c for c, cnt in col_counter.items() if cnt >= 2}
    priority = [k for k in _JOIN_CANDIDATES if k in shared]
    others   = sorted(shared - set(priority))
    return priority + others


def _safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    join_keys: list[str],
    how: str = "outer",
) -> pd.DataFrame:
    left_cols  = set(left.columns.str.lower())
    right_cols = set(right.columns.str.lower())
    usable = [k for k in join_keys if k in left_cols and k in right_cols]

    if usable:
        key = usable[0]
        try:
            merged = pd.merge(left, right, on=key, how=how, suffixes=("", "_dup"))
            dup_cols = [c for c in merged.columns if c.endswith("_dup")]
            for dup in dup_cols:
                base = dup[:-4]
                if base in merged.columns:
                    merged[base] = merged[base].combine_first(merged[dup])
                merged.drop(columns=[dup], inplace=True)
            logger.info("[Merge] on '%s': %d rows", key, len(merged))
            return merged
        except Exception as e:
            logger.warning("[Merge] failed on '%s': %s — falling back to concat", key, e)

    # Fallback: concat (align columns)
    merged = pd.concat([left, right], axis=0, ignore_index=True)
    logger.info("[Merge] concat fallback: %d rows", len(merged))
    return merged


def SmartCombinerAgent(
    file_entries: list[tuple[str, pd.DataFrame, dict]],  # (name, df, profile)
    fusion_context: dict,
) -> tuple[pd.DataFrame, dict]:
    """
    ONE LLM call to determine merge plan from column-level info only.
    Python then executes the merge safely.
    """
    if len(file_entries) == 1:
        fname, df, profile = file_entries[0]
        return df, {"strategy": "single_file", "file": fname, "shape": list(df.shape)}

    # Build a compact file summary for LLM (column names only)
    files_summary = []
    for fname, df, profile in file_entries:
        domain = fusion_context.get("domain_map", {}).get(fname) \
                 or profile.get("llm", {}).get("domain", "unknown")
        files_summary.append({
            "file": fname,
            "domain": domain,
            "cols": df.columns.tolist(),
            "rows": len(df),
        })

    prompt = f"""Supply chain dataset merger. Plan how to combine these files.

Context: {fusion_context.get('context', '')[:500]}
Join strategy hint: {fusion_context.get('join_strategy', '')}

Files (column names only):
{json.dumps(files_summary, default=str)[:2000]}

Return ONLY valid JSON:
{{
  "merge_order": ["file1.csv", "file2.csv"],
  "join_key": "column_name_or_null",
  "merge_how": "outer|inner|left",
  "domain_assignments": {{"file1.csv": "demand"}},
  "notes": "brief merge rationale"
}}"""

    plan = _parse(_llm([{"role": "user", "content": prompt}], json_mode=True, max_tokens=600))

    # Execute the merge according to plan
    domain_assignments = plan.get("domain_assignments", {})
    merge_order = plan.get("merge_order", [f[0] for f in file_entries])
    suggested_key = plan.get("join_key")

    # Build ordered dict of dfs
    file_map = {fname: (df, profile) for fname, df, profile in file_entries}
    ordered  = [(f, file_map[f][0]) for f in merge_order if f in file_map]
    # Add any files not in merge_order
    ordered += [(f, df) for f, df, _ in file_entries if f not in [o[0] for o in ordered]]

    # Determine join keys
    ordered_dfs = {f: df for f, df in ordered}
    shared_keys = _find_shared_keys(ordered_dfs)
    if suggested_key and suggested_key in shared_keys:
        shared_keys = [suggested_key] + [k for k in shared_keys if k != suggested_key]

    merge_log: list[str] = []
    master = ordered[0][1]
    for i in range(1, len(ordered)):
        right_fname, right_df = ordered[i]
        n_before = len(master)
        master = _safe_merge(master, right_df, shared_keys, how=plan.get("merge_how", "outer"))
        master.drop_duplicates(inplace=True)
        master.reset_index(drop=True, inplace=True)
        merge_log.append(
            f"✅ Merged '{ordered[i-1][0]}' + '{right_fname}' → "
            f"{len(master):,} rows × {master.shape[1]} cols (was {n_before:,})"
        )

    report = {
        "strategy":           "multi_file_smart_merge",
        "files":              [{"name": f, "domain": domain_assignments.get(f, "?"), "rows": len(df)} for f, df in ordered],
        "join_keys_used":     shared_keys[:3],
        "merge_log":          merge_log,
        "llm_plan":           plan,
        "final_shape":        list(master.shape),
    }

    logger.info("[Combiner] Master: %d rows × %d cols", *master.shape)
    return master, report


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 5 — MissingFieldsAgent
#  Classify missing columns as critical / optional; suggest strategies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def MissingFieldsAgent(
    available_cols: list[str],
    missing_cols:   list[str],
    fusion_context: dict,
) -> dict:
    """
    ONE LLM call to classify each missing column and suggest resolution.
    Returns structured dict for UI gate and user dialog.
    """
    if not missing_cols:
        return {"critical": [], "optional": [], "strategies": {}, "questions": []}

    critical = [c for c in missing_cols if c in CRITICAL_COLS]
    optional = [c for c in missing_cols if c not in CRITICAL_COLS]

    prompt = f"""Supply chain data gap analysis.

Available columns: {available_cols[:40]}
Organization type: {fusion_context.get('org_type', 'generic')}
Context: {fusion_context.get('context', '')[:300]}
Missing columns: {missing_cols}
Description hints: {fusion_context.get('missing_hints', [])}

For each missing column, provide the best resolution strategy.
Return ONLY valid JSON:
{{
  "analysis": [
    {{
      "col": "column_name",
      "priority": "critical|optional",
      "can_derive": true,
      "derive_from": ["existing_col"],
      "strategy": "rename|derive|ask_user|upload_more|fill_default",
      "user_question": "plain English question to ask the user",
      "default_value": null
    }}
  ],
  "overall_advice": "one sentence advice to user",
  "modules_at_risk": ["module_name"]
}}"""

    raw = _parse(_llm([{"role": "user", "content": prompt}], json_mode=True, max_tokens=1000))

    analysis = raw.get("analysis", [])
    strategies = {a["col"]: a for a in analysis if "col" in a}

    # Fill in any cols the LLM missed
    for col in missing_cols:
        if col not in strategies:
            strategies[col] = {
                "col": col,
                "priority": "critical" if col in CRITICAL_COLS else "optional",
                "can_derive": False,
                "strategy": "ask_user",
                "user_question": f"How should we fill in `{col}`? (column name, formula, or type 'skip')",
                "default_value": None,
            }

    return {
        "critical":       critical,
        "optional":       optional,
        "strategies":     strategies,
        "questions":      [strategies[c]["user_question"] for c in missing_cols if c in strategies],
        "overall_advice": raw.get("overall_advice", ""),
        "modules_at_risk": raw.get("modules_at_risk", []),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 6 — RequirementValidatorAgent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def RequirementValidatorAgent(schema_result: dict) -> dict:
    mapping   = schema_result["mapping"]
    available = {req for req, orig in mapping.items() if orig is not None}

    module_status: dict[str, dict] = {}
    for module, required in MODULE_REQUIREMENTS.items():
        missing = [c for c in required if c not in available]
        module_status[module] = {
            "ready":    len(missing) == 0,
            "missing":  missing,
            "required": required,
        }

    all_missing = sorted(
        {c for s in module_status.values() for c in s["missing"]} - DERIVED_COLS
    )
    enabled = [m for m, s in module_status.items() if s["ready"]]
    blocked = [m for m, s in module_status.items() if not s["ready"]]

    return {
        "module_status":    module_status,
        "enabled_modules":  enabled,
        "blocked_modules":  blocked,
        "all_missing_cols": all_missing,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 7 — FeatureEngineeringAgent
#  Auto-derive remaining missing cols using safe pandas operations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def FeatureEngineeringAgent(
    df: pd.DataFrame,
    missing_cols: list[str],
    profile: dict,
    schema_result: dict,
    missing_analysis: dict,
) -> dict:
    if not missing_cols:
        return {"derivable": [], "not_derivable": [], "steps": []}

    logger.info("[FeatureEng] Deriving: %s", missing_cols)
    avail   = list(df.columns)
    mapping = schema_result["mapping"]

    # Use MissingFieldsAgent analysis to pre-filter derivable cols
    strategies = missing_analysis.get("strategies", {})
    auto_derivable = [
        c for c in missing_cols
        if strategies.get(c, {}).get("can_derive") and
           strategies.get(c, {}).get("derive_from")
    ]

    prompt = f"""Supply chain feature engineering.
Available columns: {avail}
Missing columns to derive: {missing_cols}
Pre-analysis (derivable): {json.dumps({c: strategies.get(c, {}) for c in auto_derivable}, default=str)[:800]}
Column mapping (required→original): {json.dumps({k:v for k,v in mapping.items() if v}, default=str)[:600]}
Allowed operations ONLY: {sorted(_SAFE_OPS)}

Return ONLY valid JSON:
{{
  "derivable": ["col1"],
  "not_derivable": ["col2"],
  "steps": [
    {{
      "target_col": "missing_col",
      "operation": "safe_op",
      "source_cols": ["existing_col"],
      "params": {{}},
      "explanation": "reason"
    }}
  ]
}}
Rules:
- source_cols MUST exist in: {avail}
- supply_variation_days = actual_days - ideal_days
- fulfillment_rate = fulfilled_qty / order_qty
- inventory_status via derive_status: stock_col vs reorder_col
- delay_probability: fill_constant 0.3 if no data available
- Use fill_constant for truly unknown numeric cols"""

    raw = _parse(_llm([{"role": "user", "content": prompt}], json_mode=True, max_tokens=1200))

    # Safety: validate each step
    validated: list[dict] = []
    for step in raw.get("steps", []):
        op  = step.get("operation", "")
        src = step.get("source_cols", []) or []
        if op not in _SAFE_OPS:
            logger.warning("[FeatureEng] Rejected unsafe op: %s", op)
            continue
        if op in ("fill_constant", "date_extract"):
            validated.append(step)
        elif all(c in avail for c in src):
            validated.append(step)
        else:
            logger.warning("[FeatureEng] Rejected step — src not in df: %s", src)

    raw["steps"] = validated
    logger.info("[FeatureEng] %d validated steps", len(validated))
    return raw


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 8 — TransformationEngine  (safe pandas ops only)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def TransformationEngine(
    df: pd.DataFrame,
    steps: list[dict],
    mapping: dict,
) -> tuple[pd.DataFrame, list[str]]:
    df  = df.copy()
    log: list[str] = []

    # Alias mapped columns first
    for req_col, orig_col in mapping.items():
        if orig_col and orig_col in df.columns and req_col not in df.columns:
            df[req_col] = df[orig_col]
            log.append(f"✅ Alias '{orig_col}' → '{req_col}'")

    for step in steps:
        target = str(step.get("target_col", ""))
        op     = str(step.get("operation",  ""))
        src    = step.get("source_cols", []) or []
        params = step.get("params", {}) or {}

        if op not in _SAFE_OPS:
            log.append(f"⚠ Blocked unsafe op '{op}' for '{target}'")
            continue

        try:
            if op == "fill_constant":
                val = params.get("value", 0)
                df[target] = val
                log.append(f"✅ fill_constant: '{target}' = {val}")

            elif op in ("fill_from_col", "rename"):
                col = src[0] if src else None
                if col and col in df.columns:
                    df[target] = df[col]
                    log.append(f"✅ {op}: '{target}' ← '{col}'")

            elif op == "compute_ratio":
                if len(src) >= 2 and all(c in df.columns for c in src[:2]):
                    denom = pd.to_numeric(df[src[1]], errors="coerce").replace(0, np.nan)
                    df[target] = (pd.to_numeric(df[src[0]], errors="coerce") / denom).fillna(0).clip(0, 1)
                    log.append(f"✅ ratio: '{target}' = {src[0]} / {src[1]}")

            elif op == "compute_diff":
                if len(src) >= 2 and all(c in df.columns for c in src[:2]):
                    df[target] = (
                        pd.to_numeric(df[src[0]], errors="coerce") -
                        pd.to_numeric(df[src[1]], errors="coerce")
                    )
                    log.append(f"✅ diff: '{target}' = {src[0]} - {src[1]}")

            elif op == "compute_product":
                if len(src) >= 2 and all(c in df.columns for c in src[:2]):
                    df[target] = (
                        pd.to_numeric(df[src[0]], errors="coerce") *
                        pd.to_numeric(df[src[1]], errors="coerce")
                    )
                    log.append(f"✅ product: '{target}' = {src[0]} × {src[1]}")

            elif op == "clip":
                col = src[0] if src else None
                if col and col in df.columns:
                    lo = float(params.get("min", pd.to_numeric(df[col], errors="coerce").min()))
                    hi = float(params.get("max", pd.to_numeric(df[col], errors="coerce").max()))
                    df[target] = pd.to_numeric(df[col], errors="coerce").clip(lo, hi)
                    log.append(f"✅ clip: '{target}' [{lo},{hi}]")

            elif op == "to_lower":
                col = src[0] if src else None
                if col and col in df.columns:
                    df[target] = df[col].astype(str).str.lower()
                    log.append(f"✅ to_lower: '{target}' ← '{col}'")

            elif op == "map_values":
                col = src[0] if src else None
                if col and col in df.columns:
                    val_map = params.get("map", {})
                    default = params.get("default", "Unknown")
                    df[target] = df[col].astype(str).map(val_map).fillna(default)
                    log.append(f"✅ map_values: '{target}' from '{col}'")

            elif op == "derive_status":
                if len(src) >= 2 and all(c in df.columns for c in src[:2]):
                    stock_s   = pd.to_numeric(df[src[0]], errors="coerce").fillna(0)
                    reorder_s = pd.to_numeric(df[src[1]], errors="coerce").fillna(0)
                    df[target] = np.where(
                        stock_s > reorder_s * 1.5, "Normal",
                        np.where(stock_s > reorder_s * 0.5, "Low", "Critical"),
                    )
                    log.append(f"✅ derive_status: '{target}' from {src}")

            elif op == "log1p":
                col = src[0] if src else None
                if col and col in df.columns:
                    df[target] = np.log1p(pd.to_numeric(df[col], errors="coerce").fillna(0))
                    log.append(f"✅ log1p: '{target}' ← log1p({col})")

            elif op == "rolling_mean":
                col = src[0] if src else None
                if col and col in df.columns:
                    w = int(params.get("window", 7))
                    df[target] = (
                        pd.to_numeric(df[col], errors="coerce")
                        .fillna(method="ffill")
                        .rolling(w, min_periods=1)
                        .mean()
                    )
                    log.append(f"✅ rolling_mean({w}): '{target}' ← '{col}'")

            elif op == "date_extract":
                col  = src[0] if src else None
                part = params.get("part", "month")
                if col and col in df.columns:
                    dts = pd.to_datetime(df[col], errors="coerce")
                    part_map = {
                        "month": dts.dt.month, "year": dts.dt.year,
                        "dayofweek": dts.dt.dayofweek, "quarter": dts.dt.quarter,
                    }
                    if part in part_map:
                        df[target] = part_map[part]
                        log.append(f"✅ date_extract({part}): '{target}' ← '{col}'")

        except Exception as exc:
            log.append(f"❌ Error op='{op}' target='{target}': {exc}")
            logger.error("[Transform] %s", exc)

    return df, log


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 9 — SplitterAgent
#  Map master DataFrame → 4 canonical domain DataFrames
#  Save each separately for efficiency; only non-empty domains saved
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def SplitterAgent(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Split master DataFrame into canonical domain datasets.
    Each domain gets only the columns relevant to it (+ any shared cols).
    Returns dict: domain → DataFrame (only populated domains).
    """
    cols = set(df.columns)
    result: dict[str, pd.DataFrame] = {}

    domain_col_map = {
        "demand":    ["timestamp", "product_name", "units_sold", "price_per_unit",
                      "discount_percent", "customer_region", "demand_trend_index",
                      "season", "event_flag", "product_category"],
        "inventory": ["warehouse_id", "warehouse_location", "product_name", "stock_units",
                      "reorder_level", "physical_condition", "last_restock_date", "inventory_status"],
        "supplier":  ["supplier_id", "supplier_name", "product_name", "supplier_location",
                      "ideal_supply_time_days", "actual_supply_time_days", "supply_variation_days",
                      "supplier_reliability_score", "order_quantity", "fulfilled_quantity",
                      "fulfillment_rate", "cargo_condition_status"],
        "transport": ["timestamp", "product_name", "supplier_id", "warehouse_id", "route_type",
                      "vehicle_gps_latitude", "vehicle_gps_longitude", "fuel_consumption_rate",
                      "traffic_congestion_level", "weather_condition_severity", "shipping_costs",
                      "lead_time_days", "eta_variation_hours", "route_risk_level",
                      "delay_probability", "risk_classification", "delivery_time_deviation"],
    }

    for domain, domain_cols in domain_col_map.items():
        present = [c for c in domain_cols if c in cols]
        # Also require at least one signature column
        signature = SCHEMA_HINTS.get(domain, [])
        has_signature = any(s in cols for s in signature)
        if present and has_signature:
            # Include all present domain cols + any extra cols not in any domain
            result[domain] = df[present].copy().dropna(how="all")
            logger.info("[Splitter] Domain '%s': %d rows × %d cols", domain, len(result[domain]), len(present))

    # Fallback: if no domain matched, use master as demand
    if not result:
        result["demand"] = df.copy()

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 10 — FinalSave
#  Save canonical datasets to disk + MongoDB; keep updated on re-run
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_canonical_datasets(
    datasets: dict[str, pd.DataFrame],
    source_name: str = "dataset",
    session_hash: str = "",
) -> dict[str, str]:
    """
    Save each canonical domain dataset to disk.
    Uses fixed filenames (overwrite) so only the latest version is kept.
    Also saves master record to MongoDB.
    Returns dict: domain → filepath.
    """
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, str] = {}

    for domain, df in datasets.items():
        path = CANONICAL_DIR / f"{domain}_canonical.csv"
        df.to_csv(path, index=False)
        saved_paths[domain] = str(path)
        logger.info("[Save] canonical/%s.csv (%d rows)", domain, len(df))

    # Write metadata manifest
    manifest = {
        "source": source_name,
        "hash": session_hash,
        "saved_at": _ts(),
        "domains": {d: {"rows": len(df), "cols": df.shape[1]} for d, df in datasets.items()},
        "paths": saved_paths,
    }
    manifest_path = CANONICAL_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Save to MongoDB
    try:
        save_result(COLL_UPLOADS, {
            "source": source_name,
            "session_hash": session_hash,
            "domains": list(datasets.keys()),
            "row_counts": {d: len(df) for d, df in datasets.items()},
            "paths": saved_paths,
            "type": "canonical_save",
        })
    except Exception as exc:
        logger.warning("[Save] MongoDB log failed: %s", exc)

    return saved_paths


def load_canonical_datasets() -> dict[str, pd.DataFrame]:
    """
    Load previously saved canonical datasets from disk.
    Returns empty dict if not found.
    """
    if not CANONICAL_DIR.exists():
        return {}

    datasets: dict[str, pd.DataFrame] = {}
    for domain in CANONICAL_DOMAINS:
        path = CANONICAL_DIR / f"{domain}_canonical.csv"
        if path.exists():
            try:
                df = pd.read_csv(path)
                df = _parse_dates(df)
                datasets[domain] = df
                logger.info("[Load] canonical/%s.csv (%d rows)", domain, len(df))
            except Exception as exc:
                logger.warning("[Load] Failed to load %s: %s", path, exc)

    return datasets


def _get_canonical_manifest() -> dict:
    path = CANONICAL_DIR / "manifest.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 11 — FinalValidator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def FinalValidator(df: pd.DataFrame, _prev: dict = None) -> dict:
    avail = set(df.columns)
    module_status: dict[str, dict] = {}
    for module, required in MODULE_REQUIREMENTS.items():
        missing = [c for c in required if c not in avail]
        module_status[module] = {
            "ready":    not missing,
            "missing":  missing,
            "required": required,
        }
    enabled  = [m for m, s in module_status.items() if s["ready"]]
    blocked  = [m for m, s in module_status.items() if not s["ready"]]
    all_miss = sorted({c for s in module_status.values() for c in s["missing"]} - DERIVED_COLS)
    return {
        "module_status":    module_status,
        "enabled_modules":  enabled,
        "blocked_modules":  blocked,
        "all_missing_cols": all_miss,
        "overall_ready":    len(enabled) >= 2,
        "columns_in_df":    list(avail),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UI COMPONENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _render_gap_ui(
    ps: dict,
    master_df: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    """
    UI Gate (Step 5): Show missing fields with AI-generated questions.
    User can: upload supplemental CSV, answer questions, or skip.
    Returns updated df and whether pipeline should re-validate.
    """
    missing_analysis = ps.get("missing_analysis", {})
    still_missing    = ps.get("still_missing", [])
    if not still_missing:
        return master_df, False

    st.markdown("#### 🔍 Data Gap Resolution")
    advice = missing_analysis.get("overall_advice", "")
    if advice:
        st.info(f"💡 {advice}")

    critical = missing_analysis.get("critical", [])
    optional = missing_analysis.get("optional", [])

    if critical:
        st.error(f"**Critical missing columns** (block key modules): `{'`, `'.join(critical)}`")
    if optional:
        st.warning(f"**Optional missing columns** (reduce accuracy): `{'`, `'.join(optional)}`")

    needs_rerun = False

    # Option A: Upload supplemental CSV
    with st.expander("📎 Upload supplemental data to fill gaps", expanded=bool(critical)):
        st.caption("Upload a CSV that contains the missing columns. It will be merged with your existing data.")
        sup_file = st.file_uploader("Supplemental CSV", type=["csv"], key="supplemental_upload")
        if sup_file:
            sup_bytes = sup_file.read()
            sup_df    = load_csv_bytes(sup_bytes, sup_file.name)
            new_cols  = set(sup_df.columns) - set(master_df.columns)
            overlap   = set(sup_df.columns) & set(master_df.columns)
            st.caption(f"New columns found: {list(new_cols)} | Merge keys available: {list(overlap)[:5]}")
            if st.button("✅ Merge supplemental data", key="btn_merge_sup"):
                # Find best merge key
                shared_keys = [k for k in _JOIN_CANDIDATES if k in overlap]
                if shared_keys:
                    master_df = _safe_merge(master_df, sup_df, shared_keys, how="left")
                else:
                    master_df = pd.concat([master_df, sup_df], axis=0, ignore_index=True)
                master_df.drop_duplicates(inplace=True)
                st.success(f"✅ Merged! New shape: {master_df.shape}")
                needs_rerun = True

    # Option B: Conversational resolution
    st.markdown("#### 🤖 Data Assistant")
    master_df, chat_rerun = render_data_assistant(master_df, still_missing, ps)
    if chat_rerun:
        needs_rerun = True

    return master_df, needs_rerun


def render_data_assistant(
    df: pd.DataFrame,
    missing_cols: list[str],
    pipeline_state: dict,
) -> tuple[pd.DataFrame, bool]:
    """Conversational assistant for resolving remaining missing columns."""
    if not missing_cols:
        return df, False

    KEY = "assistant_chat"
    missing_analysis = pipeline_state.get("missing_analysis", {})
    strategies = missing_analysis.get("strategies", {})

    if KEY not in st.session_state:
        st.session_state[KEY] = []
        first_col = missing_cols[0]
        first_q = strategies.get(first_col, {}).get(
            "user_question",
            f"How should I fill in `{first_col}`? (column name, formula, `skip`, or `fill 0`)"
        )
        st.session_state[KEY].append({
            "role": "assistant",
            "content": (
                f"👋 I need a little help to complete your dataset.\n\n"
                f"**Still missing:** {', '.join(f'`{c}`' for c in missing_cols)}\n\n"
                f"**{first_col}:** {first_q}"
            )
        })

    chat_box = st.container(height=280)
    with chat_box:
        for msg in st.session_state[KEY]:
            av = "🤖" if msg["role"] == "assistant" else "👤"
            with st.chat_message(msg["role"], avatar=av):
                st.markdown(msg["content"])

    user_input = st.chat_input("Answer or provide column formula…", key="da_input")
    needs_rerun = False

    if user_input:
        st.session_state[KEY].append({"role": "user", "content": user_input})
        resolved = st.session_state.get("assistant_resolved", set())
        still    = [c for c in missing_cols if c not in resolved]

        if not still:
            st.session_state[KEY].append({"role": "assistant", "content": "✅ All columns addressed!"})
            return df, True

        target = still[0]
        txt    = user_input.strip()

        if txt.lower() in ("skip", "s"):
            resolved.add(target)
            st.session_state["assistant_resolved"] = resolved
            nxt = [c for c in still[1:]]
            if nxt:
                nxt_q = strategies.get(nxt[0], {}).get("user_question", f"How to fill `{nxt[0]}`?")
                reply = f"⏭ Skipped `{target}`.\n\n**{nxt[0]}:** {nxt_q}"
            else:
                reply = "✅ All columns addressed! Refreshing…"
                needs_rerun = True
            st.session_state[KEY].append({"role": "assistant", "content": reply})

        elif txt.lower().startswith("fill "):
            try:
                val_str = txt[5:].strip()
                val = float(val_str) if val_str.replace(".", "").replace("-", "").isdigit() else val_str
                df[target] = val
                pipeline_state["mapping"][target] = target
                resolved.add(target)
                st.session_state["assistant_resolved"] = resolved
                nxt = [c for c in still[1:]]
                if nxt:
                    nxt_q = strategies.get(nxt[0], {}).get("user_question", f"How to fill `{nxt[0]}`?")
                    reply = f"✅ Set `{target}` = {val}.\n\n**{nxt[0]}:** {nxt_q}"
                else:
                    reply = "🎉 All done! Refreshing…"
                    needs_rerun = True
                st.session_state[KEY].append({"role": "assistant", "content": reply})
            except Exception as e:
                st.session_state[KEY].append({"role": "assistant", "content": f"⚠ Could not parse value: {e}"})

        elif txt in df.columns:
            # User typed an existing column name
            df[target] = df[txt]
            pipeline_state["mapping"][target] = txt
            resolved.add(target)
            st.session_state["assistant_resolved"] = resolved
            nxt = [c for c in still[1:]]
            if nxt:
                nxt_q = strategies.get(nxt[0], {}).get("user_question", f"How to fill `{nxt[0]}`?")
                reply = f"✅ Mapped `{target}` ← `{txt}`.\n\n**{nxt[0]}:** {nxt_q}"
            else:
                reply = "🎉 All done! Refreshing…"
                needs_rerun = True
            st.session_state[KEY].append({"role": "assistant", "content": reply})

        else:
            # Try LLM formula parsing
            step = _parse_user_formula(txt, target, list(df.columns))
            if step:
                new_df, op_log = TransformationEngine(df, [step], pipeline_state.get("mapping", {}))
                if target in new_df.columns:
                    df = new_df
                    pipeline_state["mapping"][target] = target
                    resolved.add(target)
                    st.session_state["assistant_resolved"] = resolved
                    nxt = [c for c in still[1:]]
                    if nxt:
                        nxt_q = strategies.get(nxt[0], {}).get("user_question", f"How to fill `{nxt[0]}`?")
                        reply = f"✅ Derived `{target}` ({op_log[-1] if op_log else ''}).\n\n**{nxt[0]}:** {nxt_q}"
                    else:
                        reply = "🎉 All done! Refreshing…"
                        needs_rerun = True
                    st.session_state[KEY].append({"role": "assistant", "content": reply})
                else:
                    st.session_state[KEY].append({"role": "assistant",
                        "content": f"⚠ Couldn't derive `{target}`. Try a column name directly or `fill 0`."})
            else:
                st.session_state[KEY].append({"role": "assistant",
                    "content": f"❓ Didn't understand that for `{target}`.\nTry: column name · formula · `skip` · `fill 0`"})

    return df, needs_rerun


def _parse_user_formula(user_msg: str, target_col: str, df_cols: list[str]) -> dict | None:
    prompt = f"""Parse this user instruction into one safe transform.
Target column: '{target_col}'
Available columns: {df_cols}
Instruction: "{user_msg}"
Allowed operations: {sorted(_SAFE_OPS)}
Return ONLY JSON or null:
{{"target_col": "{target_col}", "operation": "op_name", "source_cols": ["col"], "params": {{}}, "explanation": "..."}}"""

    raw = _parse(_llm([{"role": "user", "content": prompt}], json_mode=True, max_tokens=300))
    if not raw or "operation" not in raw:
        return None
    if raw.get("operation") not in _SAFE_OPS:
        return None
    return raw


def render_pipeline_status(ps: dict) -> None:
    """Render pipeline status panel with module readiness grid."""
    if not ps:
        return

    fv    = ps.get("final_validation", {})
    ready = fv.get("overall_ready", False)
    ena   = fv.get("enabled_modules", [])
    blk   = fv.get("blocked_modules", [])
    miss  = fv.get("all_missing_cols", [])

    if ready:
        st.success(f"✅ **Dataset Ready** — {len(ena)} modules enabled, 4 canonical datasets saved")
    else:
        st.warning(f"⚠️ **Partial** — {len(ena)} modules ready, {len(blk)} need more data")

    fc = ps.get("file_count", 1)
    if fc > 1:
        st.info(f"🔗 **{fc} files combined** → {', '.join(ps.get('filenames', []))}")

    # Canonical datasets status
    saved_domains = ps.get("saved_domains", {})
    if saved_domains:
        st.markdown("**📁 Saved Canonical Datasets:**")
        dcols = st.columns(4)
        for i, domain in enumerate(CANONICAL_DOMAINS):
            if domain in saved_domains:
                row_count = ps.get("canonical_row_counts", {}).get(domain, 0)
                dcols[i].success(f"✅ **{domain.title()}**\n{row_count:,} rows")
            else:
                dcols[i].error(f"❌ **{domain.title()}**\nnot available")

    c1, c2, c3, c4, c5 = st.columns(5)
    profile = ps.get("master_profile", {})
    schema  = ps.get("master_schema", {})
    c1.metric("Rows",        f"{profile.get('rows', 0):,}")
    c2.metric("Columns",     f"{profile.get('cols', 0)}")
    c3.metric("Cols Mapped", f"{schema.get('mapped_count', 0)}/{schema.get('total_required', 0)}")
    c4.metric("Modules On",  f"{len(ena)}/8")
    c5.metric("Missing",     f"{len(miss)}")

    # Combination report
    combo_report = ps.get("combination_report", {})
    if combo_report and combo_report.get("strategy") != "single_file":
        with st.expander("🔗 Combination Report", expanded=False):
            for line in combo_report.get("merge_log", []):
                if line.startswith("✅"):
                    st.success(line)
                elif line.startswith("⚠"):
                    st.warning(line)
                else:
                    st.caption(line)

    # Module grid
    st.markdown("#### 📊 Module Readiness")
    icons = {
        "demand_forecast": "📦", "risk_assessment": "⚠️",
        "inventory_management": "📊", "seasonality": "🌊",
        "stockout_prediction": "🚨", "supplier_risk": "🏭",
        "route_optimization": "🚛", "report": "📋",
    }
    cols = st.columns(4)
    for i, (mod, status) in enumerate(fv.get("module_status", {}).items()):
        lbl = mod.replace("_", " ").title()
        ico = icons.get(mod, "🔧")
        with cols[i % 4]:
            if status["ready"]:
                st.success(f"{ico} **{lbl}**\n\n✅ Ready")
            else:
                m2 = status["missing"]
                ms = ", ".join(f"`{c}`" for c in m2[:2])
                if len(m2) > 2:
                    ms += f" +{len(m2)-2}"
                st.error(f"{ico} **{lbl}**\n\n❌ {ms}")

    # Master mapping table
    with st.expander("🗺️ Column Mapping", expanded=False):
        mapping = ps.get("master_schema", {}).get("mapping", {})
        conf    = ps.get("master_schema", {}).get("confidence", {})
        rows = [
            {
                "Required": c,
                "Mapped To": mapping.get(c) or "—",
                "Confidence": f"{conf.get(c, 0):.0%}" if mapping.get(c) else "—",
                "Status": "✅" if mapping.get(c) else "❌",
            }
            for c in ALL_REQUIRED_COLS
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Transform log
    with st.expander("📋 Transform Log", expanded=False):
        for line in ps.get("transform_log", []):
            if line.startswith("✅"):
                st.success(line, icon="✅")
            elif line.startswith("⚠"):
                st.warning(line, icon="⚠️")
            elif line.startswith("❌"):
                st.error(line, icon="❌")
            else:
                st.caption(line)

    # Download all canonical datasets
    st.markdown("**⬇️ Download Canonical Datasets:**")
    ds = get_datasets()
    dl_cols = st.columns(len(ds)) if ds else []
    for (domain, df_d), col in zip(ds.items(), dl_cols):
        col.download_button(
            f"⬇ {domain.title()}",
            data=df_d.to_csv(index=False).encode(),
            file_name=f"skvision_{domain}_canonical.csv",
            mime="text/csv",
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN PIPELINE ORCHESTRATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_autonomous_pipeline(
    dfs: list[tuple[str, pd.DataFrame]],
    user_description: str = "",
    desc_file_content: str = "",
) -> dict:
    """
    Full 11-stage pipeline. LLM is called sequentially — never concurrently.
    Stages: Profile → Describe → Map → Combine → Gaps → Validate → Engineer
            → Transform → Split → Validate → Save
    """
    n     = len(dfs)
    label = f"{n} file{'s' if n > 1 else ''}"
    prog  = st.progress(0, text="Starting pipeline…")

    # ── Stage 1: Profile each file (column-level only) ─────────────────────
    prog.progress(0.05, text=f"🔍 Stage 1/11 — Profiling {label}…")
    file_entries: list[tuple[str, pd.DataFrame, dict]] = []
    profiles: list[dict] = []
    for i, (fname, df) in enumerate(dfs):
        p = ColumnProfilerAgent(df, fname, user_description)
        profiles.append(p)
        file_entries.append((fname, df, p))

    # ── Stage 2: Schema mapping per file ───────────────────────────────────
    prog.progress(0.15, text="🗺 Stage 2/11 — Mapping schemas…")
    per_schemas: list[dict] = []
    for _, _, profile in file_entries:
        per_schemas.append(SchemaMapperAgent(profile))

    # ── Stage 3: Description fusion ─────────────────────────────────────────
    prog.progress(0.22, text="📖 Stage 3/11 — Fusing descriptions…")
    fusion_ctx = DescriptionFusionAgent(profiles, user_description, desc_file_content)

    # ── Stage 4: Smart combination ──────────────────────────────────────────
    prog.progress(0.30, text=f"🔗 Stage 4/11 — Combining {label}…")
    master_df, combo_report = SmartCombinerAgent(file_entries, fusion_ctx)

    # ── Stage 5: Profile the master ─────────────────────────────────────────
    prog.progress(0.40, text="🔍 Stage 5/11 — Profiling combined dataset…")
    master_profile = ColumnProfilerAgent(master_df, "master_combined", user_description)
    master_schema  = SchemaMapperAgent(master_profile)

    # ── Stage 6: Validate requirements ──────────────────────────────────────
    prog.progress(0.48, text="✅ Stage 6/11 — Validating requirements…")
    validation = RequirementValidatorAgent(master_schema)
    missing    = validation["all_missing_cols"]

    # ── Stage 7: Analyze missing fields ─────────────────────────────────────
    prog.progress(0.55, text="🔍 Stage 7/11 — Analyzing data gaps…")
    missing_analysis: dict = {"critical": [], "optional": [], "strategies": {}, "questions": []}
    if missing:
        missing_analysis = MissingFieldsAgent(
            list(master_df.columns), missing, fusion_ctx
        )

    # ── Stage 8: Feature engineering ────────────────────────────────────────
    prog.progress(0.63, text=f"⚙️ Stage 8/11 — Engineering {len(missing)} features…")
    fe_result: dict = {"derivable": [], "not_derivable": missing, "steps": []}
    if missing:
        fe_result = FeatureEngineeringAgent(
            master_df, missing, master_profile, master_schema, missing_analysis
        )

    # ── Stage 9: Transformations ─────────────────────────────────────────────
    prog.progress(0.72, text="🔄 Stage 9/11 — Applying transforms…")
    df_t, t_log = TransformationEngine(
        master_df, fe_result.get("steps", []), master_schema["mapping"]
    )

    # ── Stage 10: Split into canonical domains ────────────────────────────────
    prog.progress(0.82, text="✂️ Stage 10/11 — Splitting to canonical datasets…")
    canonical_datasets = SplitterAgent(df_t)

    # ── Stage 11: Final validation + save ────────────────────────────────────
    prog.progress(0.90, text="💾 Stage 11/11 — Saving canonical datasets…")
    final_v = FinalValidator(df_t)

    combined_name = "_".join(re.sub(r"[^\w]", "_", f[0])[:12] for f in dfs[:3])
    combo_hash    = _compute_multi_hash([(f[0], f[0].encode()) for f in dfs])

    saved_paths = {}
    if final_v["overall_ready"] or canonical_datasets:
        saved_paths = save_canonical_datasets(canonical_datasets, combined_name, combo_hash)

    prog.progress(1.0, text="✅ Pipeline complete!")
    prog.empty()

    state = {
        # File metadata
        "filenames":            [f[0] for f in dfs],
        "file_count":           n,
        "user_description":     user_description,
        # Per-file artifacts
        "per_file_profiles":    profiles,
        "per_file_schemas":     per_schemas,
        # Fusion context
        "fusion_context":       fusion_ctx,
        # Combination
        "combination_report":   combo_report,
        # Master artifacts
        "master_profile":       master_profile,
        "master_schema":        master_schema,
        "mapping":              master_schema["mapping"],
        "validation":           validation,
        "missing_analysis":     missing_analysis,
        "fe_result":            fe_result,
        "transform_log":        t_log,
        "final_validation":     final_v,
        "df_transformed":       df_t,
        # Canonical datasets
        "canonical_datasets":   canonical_datasets,
        "saved_domains":        saved_paths,
        "canonical_row_counts": {d: len(df) for d, df in canonical_datasets.items()},
        # Module flags
        "overall_ready":        final_v["overall_ready"],
        "enabled_modules":      final_v["enabled_modules"],
        "blocked_modules":      final_v["blocked_modules"],
        "still_missing":        final_v["all_missing_cols"],
        "filename":             combined_name,
        "ran_at":               _ts(),
    }

    st.session_state["pipeline_state"] = state

    # Push canonical datasets to session
    if canonical_datasets:
        st.session_state["datasets"] = canonical_datasets
    else:
        _push_datasets_from_pipeline(state)

    _log_upload([f[0] for f in dfs], canonical_datasets or {})
    return state


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PUBLIC UPLOAD SECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _compute_multi_hash(files_bytes: list[tuple[str, bytes]]) -> str:
    h = hashlib.md5()
    for name, b in sorted(files_bytes, key=lambda x: x[0]):
        h.update(name.encode())
        h.update(b[:4096])
    return h.hexdigest()


def render_upload_section() -> dict[str, pd.DataFrame]:
    """
    Main upload UI.

    AUTONOMOUS MODE (recommended):
      - Upload CSV(s) + optional description file (TXT/JSON)
      - AI profiles, combines, validates, engineers, and splits to 4 canonical datasets
      - Persistent storage: canonical datasets saved to disk + MongoDB
      - On re-visit, auto-restores from disk if available

    CLASSIC MODE (backward compatible):
      - Upload up to 4 pre-structured CSVs (demand/inventory/supplier/transport)
      - Direct load with type detection, no LLM required
      - Download + save buttons included
    """
    st.subheader("📂 Upload Supply Chain Datasets")

    # Check for existing canonical datasets
    manifest = _get_canonical_manifest()
    if manifest:
        existing_domains = list(manifest.get("domains", {}).keys())
        saved_at = manifest.get("saved_at", "")[:10]
        if existing_domains and not st.session_state.get("datasets"):
            st.info(
                f"📁 Found saved canonical datasets from **{saved_at}**: "
                f"{', '.join(existing_domains)}. Loading automatically…"
            )
            loaded = load_canonical_datasets()
            if loaded:
                st.session_state["datasets"] = loaded
                st.success(f"✅ Auto-restored: {', '.join(f'**{d}** ({len(df):,} rows)' for d, df in loaded.items())}")

    mode = st.radio(
        "Upload Mode",
        [
            "🧠 Autonomous (any CSV — AI maps, combines & prepares automatically)",
            "📋 Classic (pre-structured 4-file upload)",
        ],
        horizontal=True,
        key="upload_mode",
    )

    # ─── AUTONOMOUS ─────────────────────────────────────────────────────────
    if mode.startswith("🧠"):
        st.caption(
            "Upload **one or more** CSV files from any ERP, WMS, or TMS system. "
            "The AI pipeline will profile columns, intelligently combine files, "
            "identify gaps, and prepare 4 canonical datasets."
        )

        col_left, col_right = st.columns([3, 1])
        with col_left:
            uploaded_files = st.file_uploader(
                "Upload CSV/TSV files",
                type=["csv", "tsv"],
                accept_multiple_files=True,
                key="auto_uploader",
            )
        with col_right:
            desc_file = st.file_uploader(
                "Optional: Dataset description (TXT/JSON)",
                type=["txt", "json"],
                key="desc_uploader",
                help="Upload your dataset spec/README. AI uses this to better understand column meanings.",
            )

        user_desc = st.text_area(
            "📝 Describe your data (optional but recommended for better accuracy)",
            placeholder=(
                "e.g. 'sales.csv = weekly SKU-level sales from SAP; "
                "wms_export.csv = warehouse stock snapshot from Oracle WMS; "
                "vendor_kpi.csv = monthly supplier scorecard from procurement team.'"
            ),
            height=65,
            key="user_desc",
        )

        if not uploaded_files:
            # Show restore option if canonical data exists
            saved = get_datasets()
            if saved:
                st.markdown("**Currently loaded datasets:**")
                rcols = st.columns(min(len(saved), 4))
                for (domain, df), col in zip(saved.items(), rcols):
                    col.success(f"**{domain.title()}**\n{len(df):,} rows × {df.shape[1]} cols")
            return get_datasets()

        # Read bytes + compute combined hash
        files_bytes: list[tuple[str, bytes]] = []
        for uf in uploaded_files:
            b = uf.read()
            files_bytes.append((uf.name, b))

        desc_content = ""
        if desc_file:
            desc_content = desc_file.read().decode("utf-8", errors="ignore")

        combo_hash = _compute_multi_hash(files_bytes)

        if st.session_state.get("_file_hash") != combo_hash:
            # New file set → reset pipeline
            st.session_state["_file_hash"]         = combo_hash
            st.session_state["pipeline_state"]     = {}
            st.session_state["assistant_chat"]     = []
            st.session_state["assistant_resolved"] = set()

            loaded: list[tuple[str, pd.DataFrame]] = []
            for name, raw in files_bytes:
                df_raw = load_csv_bytes(raw, name)
                loaded.append((name, df_raw))
            st.session_state["_raw_dfs"]          = loaded
            st.session_state["_desc_content"]     = desc_content

        loaded       = st.session_state.get("_raw_dfs", [])
        desc_content = st.session_state.get("_desc_content", desc_content)

        if not loaded:
            return get_datasets()

        # Per-file preview
        st.markdown(f"**{len(loaded)} file(s) ready:**")
        prev_cols = st.columns(min(len(loaded), 4))
        for i, (fname, df_raw) in enumerate(loaded):
            prev_cols[i % 4].info(f"📄 **{fname}**\n{len(df_raw):,} rows × {df_raw.shape[1]} cols")

        if len(loaded) > 1:
            with st.expander("👀 Preview all files", expanded=False):
                tabs = st.tabs([f"📄 {f[0][:20]}" for f in loaded])
                for tab, (fname, df_raw) in zip(tabs, loaded):
                    with tab:
                        st.dataframe(df_raw.head(5), use_container_width=True)

        ps = st.session_state.get("pipeline_state", {})

        if not ps:
            btn_label = (
                f"🚀 Run AI Pipeline on {len(loaded)} file{'s' if len(loaded) > 1 else ''}"
            )
            if desc_content:
                st.caption(f"✅ Description file loaded ({len(desc_content):,} chars) — AI will use it for better mapping.")
            if st.button(btn_label, type="primary", use_container_width=True):
                ps = run_autonomous_pipeline(
                    loaded,
                    user_description=user_desc,
                    desc_file_content=desc_content,
                )
                st.rerun()
        else:
            st.divider()
            render_pipeline_status(ps)

            still_miss = ps.get("still_missing", [])
            if still_miss:
                st.divider()
                df_t = ps.get("df_transformed", loaded[0][1])
                df_t, needs_rerun = _render_gap_ui(ps, df_t)

                if needs_rerun:
                    ps["df_transformed"] = df_t
                    # Re-run validation + split
                    fv = FinalValidator(df_t)
                    canon = SplitterAgent(df_t)
                    saved_paths = {}
                    if fv["overall_ready"] or canon:
                        saved_paths = save_canonical_datasets(
                            canon, ps["filename"],
                            st.session_state.get("_file_hash", "")
                        )
                    ps.update({
                        "final_validation":   fv,
                        "still_missing":      fv["all_missing_cols"],
                        "enabled_modules":    fv["enabled_modules"],
                        "overall_ready":      fv["overall_ready"],
                        "canonical_datasets": canon,
                        "saved_domains":      saved_paths,
                        "canonical_row_counts": {d: len(df) for d, df in canon.items()},
                    })
                    st.session_state["pipeline_state"] = ps
                    if canon:
                        st.session_state["datasets"] = canon
                    else:
                        _push_datasets_from_pipeline(ps)
                    st.session_state["assistant_chat"]     = []
                    st.session_state["assistant_resolved"] = set()
                    st.rerun()

            if st.button("🔄 Reset & Re-run Pipeline", key="btn_rerun"):
                for k in ("pipeline_state", "assistant_chat", "assistant_resolved",
                          "_file_hash", "_raw_dfs", "_desc_content"):
                    st.session_state.pop(k, None)
                st.rerun()

        return get_datasets()

    # ─── CLASSIC ────────────────────────────────────────────────────────────
    st.caption("Upload up to 4 pre-structured CSVs: demand · inventory · supplier · transport.")

    col_a, col_b = st.columns(2)
    with col_a:
        uploaded_files = st.file_uploader(
            "Upload CSV files",
            type=["csv", "tsv"],
            accept_multiple_files=True,
            key="classic_uploader",
        )
    with col_b:
        st.markdown("**Expected columns per file:**")
        for domain, hints in SCHEMA_HINTS.items():
            st.caption(f"**{domain.title()}:** {', '.join(hints)}")

    datasets: dict[str, pd.DataFrame] = {}
    if not uploaded_files:
        return st.session_state.get("datasets", {})

    prog = st.progress(0)
    for i, f in enumerate(uploaded_files):
        raw = f.read()
        df  = load_csv_bytes(raw, f.name)
        dt  = _detect_dtype_classic(df)
        if dt == "unknown":
            st.warning(f"⚠ Could not detect type for **{f.name}** — treating as demand.")
            dt = "demand"
        datasets[dt] = df
        prog.progress((i + 1) / len(uploaded_files))

    st.session_state["datasets"] = datasets

    # Save to canonical dir in classic mode too
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
    for domain, df in datasets.items():
        path = CANONICAL_DIR / f"{domain}_canonical.csv"
        df.to_csv(path, index=False)

    _log_upload([f.name for f in uploaded_files], datasets)

    # Display metrics
    st.markdown("**Detected Datasets:**")
    cols = st.columns(len(datasets)) if datasets else []
    for (dt, df), col in zip(datasets.items(), cols):
        col.metric(dt.upper(), f"{len(df):,} rows", f"{df.shape[1]} cols")

    # Download buttons
    dl_cols = st.columns(len(datasets)) if datasets else []
    for (domain, df), col in zip(datasets.items(), dl_cols):
        col.download_button(
            f"⬇ {domain.title()}",
            data=df.to_csv(index=False).encode(),
            file_name=f"skvision_{domain}_canonical.csv",
            mime="text/csv",
            key=f"dl_classic_{domain}",
        )

    return datasets


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  INTERNAL HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _push_datasets_from_pipeline(ps: dict) -> None:
    """Fallback: populate session_state datasets from unified pipeline df."""
    df = ps.get("df_transformed")
    if df is None:
        return
    cols = set(df.columns)
    ds: dict[str, pd.DataFrame] = {}
    if any(c in cols for c in ("units_sold", "product_name", "timestamp")):
        ds["demand"] = df
    if any(c in cols for c in ("stock_units", "warehouse_id", "reorder_level")):
        ds["inventory"] = df
    if any(c in cols for c in ("supplier_name", "fulfillment_rate", "supplier_id")):
        ds["supplier"] = df
    if any(c in cols for c in ("route_type", "delay_probability", "shipping_costs")):
        ds["transport"] = df
    if not ds:
        ds["demand"] = df
    st.session_state["datasets"] = ds


def _detect_dtype_classic(df: pd.DataFrame) -> str:
    cols = set(df.columns.str.lower())
    scores: dict[str, int] = {}
    for dt, req in SCHEMA_HINTS.items():
        scores[dt] = sum(1 for c in req if c in cols)
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] >= 2 else "unknown"


def _log_upload(filenames: list[str], datasets: dict) -> None:
    try:
        save_result(COLL_UPLOADS, {
            "files":          filenames,
            "detected_types": list(datasets.keys()),
            "row_counts":     {k: len(v) for k, v in datasets.items()},
        })
    except Exception as exc:
        logger.warning("Upload log failed: %s", exc)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PUBLIC API  (all existing callers unchanged)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_datasets() -> dict[str, pd.DataFrame]:
    """Return currently loaded canonical datasets."""
    return st.session_state.get("datasets", {})


def require_dataset(dtype: str) -> pd.DataFrame | None:
    ds = get_datasets()
    if dtype in ds:
        return ds[dtype]
    ps = st.session_state.get("pipeline_state", {})
    df_t = ps.get("df_transformed")
    if df_t is not None:
        return df_t
    st.info(
        f"ℹ Upload a **{dtype}** dataset on the Home page. "
        "In Autonomous mode, any CSV with relevant columns is enough."
    )
    return None


def is_module_enabled(module_name: str) -> bool:
    ps = st.session_state.get("pipeline_state", {})
    if not ps:
        return True
    return module_name in ps.get("enabled_modules", [])


def get_pipeline_state() -> dict:
    return st.session_state.get("pipeline_state", {})


def auto_load_reference_files() -> dict:
    """
    Startup: try canonical dir first (fastest), then dev reference CSVs.
    Only loads if no datasets already in session.
    """
    # Skip if already loaded
    if st.session_state.get("datasets"):
        return st.session_state["datasets"]

    # Try canonical datasets from previous session
    canonical = load_canonical_datasets()
    if canonical:
        st.session_state["datasets"] = canonical
        logger.info("[AutoLoad] Restored %d canonical datasets from disk", len(canonical))
        return canonical

    # Dev fallback: reference CSV paths
    paths = {
        "demand":    "/mnt/user-data/uploads/daily_product_demand.csv",
        "inventory": "/mnt/user-data/uploads/warehouse_inventory.csv",
        "supplier":  "/mnt/user-data/uploads/supplier_supply.csv",
        "transport": "/mnt/user-data/uploads/transport_route_data.csv",
    }
    ds: dict[str, pd.DataFrame] = {}
    for dtype, path in paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            ds[dtype] = _clean(df)
    if ds:
        st.session_state.setdefault("datasets", ds)
        logger.info("[AutoLoad] Loaded %d reference files", len(ds))
    return ds
