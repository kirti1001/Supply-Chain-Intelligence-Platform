"""
data_loader.py — SKVision Supply Chain Intelligence Platform
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7-Stage Pipeline — now supports MULTIPLE CSVs in Autonomous mode:

  1. DataProfilerAgent         — understand schema + sample data (per file)
  2. SchemaMapperAgent         — map user cols → required cols (with confidence)
  3. MultiFileIntegratorAgent  — ★ NEW: classify + smart-merge N files
  4. RequirementValidatorAgent — check which modules have enough cols
  5. FeatureEngineeringAgent   — derive missing cols via safe transforms
  6. DataAssistantChatbot      — ask user if still missing (with memory)
  7. TransformationEngine      — apply controlled (no-exec) transforms
  8. FinalValidator            — final readiness gate + save to disk

SAFETY: LLM never executes code directly. All transforms are
        applied via a whitelisted set of pandas operations only.

MULTI-FILE STRATEGY:
  • Each uploaded CSV is individually profiled + schema-mapped.
  • MultiFileIntegratorAgent classifies each file as demand/inventory/
    supplier/transport/mixed and determines the best merge strategy
    (join on shared keys, concat on matching schema, or stack columns).
  • The merged "master" DataFrame flows through the rest of the pipeline
    exactly like a single-CSV upload — all downstream agents unchanged.
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

from db_ops import save_result, COLL_UPLOADS

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("data_loader")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


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

# These are always computed, never block a module
DERIVED_COLS: set[str] = {
    "demand_risk_scores", "inventory_risk_scores",
    "supplier_risk_scores", "transport_risk_scores",
}

# Per-module minimum columns needed
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

PREPARED_DIR = Path("dataset/prepared")

# Domain→signature columns (used by classifier + classic mode)
SCHEMA_HINTS: dict[str, list[str]] = {
    "demand":    ["timestamp", "product_name", "units_sold"],
    "inventory": ["warehouse_id", "product_name", "stock_units", "reorder_level"],
    "supplier":  ["supplier_id", "product_name", "fulfillment_rate",
                  "supplier_reliability_score"],
    "transport": ["timestamp", "product_name", "supplier_id",
                  "warehouse_id", "route_type"],
}

# Natural-language join-key candidates per domain pair
_JOIN_CANDIDATES: list[str] = [
    "product_name", "product_id", "sku", "item", "item_name",
    "supplier_id", "supplier_name", "warehouse_id", "order_id",
    "timestamp", "date",
]

# Whitelisted safe transform operations
_SAFE_OPS: set[str] = {
    "rename", "fill_constant", "fill_from_col", "compute_ratio",
    "compute_diff", "compute_product", "clip", "to_lower",
    "map_values", "derive_status", "log1p", "rolling_mean", "date_extract",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SAFE LLM WRAPPER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _llm(messages: list[dict], json_mode: bool = False,
         max_tokens: int = 2000) -> str:
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 0 — Basic cleaning
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if any(kw in col.lower() for kw in ("date", "time", "stamp", "_dt")):
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
#  STAGE 1 — DataProfilerAgent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def DataProfilerAgent(df: pd.DataFrame, user_description: str = "") -> dict:
    """
    Statistical + LLM-semantic profiling of an arbitrary DataFrame.
    Returns structured profile dict used by every downstream agent.
    """
    logger.info("[Profiler] %d rows x %d cols", len(df), df.shape[1])

    col_profiles: list[dict] = []
    for col in df.columns:
        s = df[col]
        null_pct = round(float(s.isnull().mean() * 100), 1)
        uniq = int(s.nunique())
        sample = [str(v) for v in s.dropna().head(5).tolist()]

        if pd.api.types.is_numeric_dtype(s):
            kind = "numeric"
            stats: dict = {
                "min": round(float(s.min()), 3) if not s.empty else None,
                "max": round(float(s.max()), 3) if not s.empty else None,
                "mean": round(float(s.mean()), 3) if not s.empty else None,
                "std": round(float(s.std()), 3) if not s.empty else None,
            }
        elif pd.api.types.is_datetime64_any_dtype(s):
            kind = "datetime"
            stats = {
                "min": str(s.min()),
                "max": str(s.max()),
                "range_days": int((s.max() - s.min()).days) if not s.empty else 0,
            }
        else:
            kind = "categorical"
            top = s.value_counts().head(5).to_dict()
            stats = {"top_values": {str(k): int(v) for k, v in top.items()}}

        col_profiles.append({
            "column": col, "dtype": str(s.dtype), "kind": kind,
            "null_pct": null_pct, "unique_count": uniq,
            "sample": sample, "stats": stats,
        })

    safe_rows = [{k: str(v) for k, v in r.items()}
                 for r in df.head(3).to_dict(orient="records")]

    prompt = f"""You are a supply-chain data profiler.
Dataset: {len(df)} rows, {df.shape[1]} columns.
User description: "{user_description or 'none'}"
Column profiles: {json.dumps(col_profiles, default=str)[:3000]}
Sample rows: {json.dumps(safe_rows)[:800]}

Respond ONLY with valid JSON:
{{
  "dataset_type": "demand|inventory|supplier|transport|mixed|unknown",
  "dataset_summary": "one sentence",
  "organization_type": "retail|manufacturing|pharma|logistics|generic",
  "has_timeseries": true,
  "date_column": "col_name or null",
  "product_column": "col_name or null",
  "quantity_column": "col_name or null",
  "likely_join_keys": ["col_name"],
  "columns": [
    {{
      "column": "original_col_name",
      "semantic": "what this means in supply chain",
      "maps_to": "required_col_name or null",
      "confidence": 0.0,
      "notes": ""
    }}
  ]
}}"""

    raw = _llm([{"role": "user", "content": prompt}], json_mode=True)
    llm_profile = _parse(raw)

    return {
        "row_count": len(df),
        "col_count": df.shape[1],
        "columns": df.columns.tolist(),
        "col_profiles": col_profiles,
        "llm_profile": llm_profile,
        "user_description": user_description,
        "profiled_at": datetime.datetime.utcnow().isoformat(),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 2 — SchemaMapperAgent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_HEURISTICS: dict[str, list[str]] = {
    "timestamp":                  ["timestamp","date","time","dt","order_date","sale_date","trans_date"],
    "product_name":               ["product_name","product","item","sku","item_name","product_desc"],
    "units_sold":                 ["units_sold","quantity","qty","sales","demand","volume","sold","qty_sold"],
    "warehouse_id":               ["warehouse_id","warehouse","wh_id","depot","location_id","wh_code"],
    "stock_units":                ["stock_units","stock","inventory","on_hand","available","qty_on_hand","closing_stock"],
    "reorder_level":              ["reorder_level","reorder","min_stock","min_qty","safety_stock","reorder_point"],
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


def SchemaMapperAgent(profile: dict) -> dict:
    """
    Map original column names → required column names.
    Priority: exact match > LLM suggestion (conf≥0.65) > heuristic fuzzy.
    """
    logger.info("[SchemaMapper] Mapping columns…")
    llm_cols: list[dict] = profile.get("llm_profile", {}).get("columns", [])
    orig_cols: list[str] = profile.get("columns", [])

    mapping:    dict[str, str | None] = {c: None for c in ALL_REQUIRED_COLS}
    confidence: dict[str, float]      = {c: 0.0  for c in ALL_REQUIRED_COLS}

    lower_orig = {c.lower(): c for c in orig_cols}

    # Pass 1 — LLM suggestions ≥0.65
    for lc in llm_cols:
        orig   = lc.get("column", "")
        req    = lc.get("maps_to")
        conf   = float(lc.get("confidence", 0.0))
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

    # Pass 3 — Exact name match (highest priority, overrides everything)
    for req_col in ALL_REQUIRED_COLS:
        if req_col in lower_orig:
            mapping[req_col]    = lower_orig[req_col]
            confidence[req_col] = 1.0

    mapped = sum(1 for v in mapping.values() if v is not None)
    logger.info("[SchemaMapper] Mapped %d/%d", mapped, len(ALL_REQUIRED_COLS))
    return {
        "mapping": mapping,
        "confidence": confidence,
        "mapped_count": mapped,
        "total_required": len(ALL_REQUIRED_COLS),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 3 (NEW) — MultiFileIntegratorAgent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _classify_file_domain(df: pd.DataFrame, profile: dict) -> str:
    """
    Heuristic + LLM-backed domain classification for a single DataFrame.
    Returns one of: demand | inventory | supplier | transport | mixed
    """
    # 1. LLM says so
    llm_type = profile.get("llm_profile", {}).get("dataset_type", "")
    if llm_type in ("demand", "inventory", "supplier", "transport"):
        return llm_type

    # 2. Heuristic signature matching
    cols = set(df.columns.str.lower())
    scores: dict[str, int] = {}
    for domain, hints in SCHEMA_HINTS.items():
        scores[domain] = sum(1 for h in hints if h in cols)
    best = max(scores, key=lambda d: scores[d])
    if scores[best] >= 2:
        return best
    return "mixed"


def _find_join_keys(dfs: dict[str, pd.DataFrame]) -> list[str]:
    """
    Find column names present in ≥2 DataFrames — candidates for merging.
    Prefer semantically important keys (product_name, supplier_id, etc.).
    """
    from collections import Counter
    col_counter: Counter = Counter()
    for df in dfs.values():
        col_counter.update(set(df.columns.str.lower()))

    # Columns in at least 2 files
    shared = {c for c, cnt in col_counter.items() if cnt >= 2}
    # Prioritise known join keys
    priority = [k for k in _JOIN_CANDIDATES if k in shared]
    others   = sorted(shared - set(priority))
    return priority + others


def _merge_two(
    left: pd.DataFrame,
    right: pd.DataFrame,
    join_keys: list[str],
    how: str = "outer",
) -> pd.DataFrame:
    """
    Merge two DataFrames on the best available shared key.
    Falls back to concat if no shared key found.
    """
    left_cols  = set(left.columns.str.lower())
    right_cols = set(right.columns.str.lower())
    usable_keys = [k for k in join_keys if k in left_cols and k in right_cols]

    if usable_keys:
        key = usable_keys[0]
        logger.info("[Integrator] Merging on key='%s' (%s)", key, how)
        try:
            merged = pd.merge(left, right, on=key, how=how,
                              suffixes=("", f"__{right.shape[1]}r"))
            # Drop suffix duplicates that add no info
            dup_cols = [c for c in merged.columns if c.endswith(f"__{right.shape[1]}r")]
            merged.drop(columns=dup_cols, inplace=True, errors="ignore")
            return merged
        except Exception as exc:
            logger.warning("[Integrator] Merge failed (%s), falling back to concat", exc)

    # Schema-concat: both have same columns → stack rows
    if set(left.columns) == set(right.columns):
        logger.info("[Integrator] Schema match → concat rows")
        return pd.concat([left, right], ignore_index=True)

    # Last resort: concat columns (wide join)
    logger.info("[Integrator] No common key → concat columns")
    min_rows = min(len(left), len(right))
    return pd.concat(
        [left.iloc[:min_rows].reset_index(drop=True),
         right.iloc[:min_rows].reset_index(drop=True)],
        axis=1,
    )


def MultiFileIntegratorAgent(
    files: list[tuple[str, pd.DataFrame, dict]],  # (filename, df, profile)
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    ★ New Stage 3 — Intelligently integrates N DataFrames into one master df.

    Strategy decision tree (per-pair):
      1. Shared key exists  → pd.merge (outer) on best key
      2. Identical schemas  → pd.concat (stack rows = same table type)
      3. Otherwise          → wide concat on min-row count

    Returns (master_df, integration_report).
    """
    if len(files) == 1:
        fname, df, profile = files[0]
        domain = _classify_file_domain(df, profile)
        return df.copy(), {
            "strategy": "single_file",
            "files": [{"name": fname, "domain": domain,
                       "rows": len(df), "cols": df.shape[1]}],
            "join_keys_used": [],
            "merge_log": [f"Single file '{fname}' — no merge needed."],
        }

    logger.info("[Integrator] Integrating %d files…", len(files))

    # Classify each file
    classified: dict[str, dict] = {}
    for fname, df, profile in files:
        domain = _classify_file_domain(df, profile)
        classified[fname] = {
            "domain": domain, "df": df,
            "rows": len(df), "cols": df.shape[1],
        }
        logger.info("[Integrator] '%s' → %s (%d rows)", fname, domain, len(df))

    # Group same-domain files → concat within group first
    domain_groups: dict[str, list[pd.DataFrame]] = {}
    for fname, meta in classified.items():
        d = meta["domain"]
        domain_groups.setdefault(d, []).append(meta["df"])

    # Within-group concat
    domain_dfs: dict[str, pd.DataFrame] = {}
    merge_log: list[str] = []
    for domain, dfs_list in domain_groups.items():
        if len(dfs_list) == 1:
            domain_dfs[domain] = dfs_list[0]
        else:
            try:
                stacked = pd.concat(dfs_list, ignore_index=True)
                stacked.drop_duplicates(inplace=True)
                domain_dfs[domain] = stacked
                merge_log.append(
                    f"✅ Stacked {len(dfs_list)} '{domain}' files → "
                    f"{len(stacked):,} rows"
                )
            except Exception as exc:
                domain_dfs[domain] = dfs_list[0]
                merge_log.append(f"⚠ Could not stack '{domain}' files: {exc}")

    # Find global join keys
    join_keys = _find_join_keys(domain_dfs)
    merge_log.append(f"🔑 Candidate join keys: {join_keys or ['none found']}")

    # Cross-domain merge (reduce all domain_dfs into one master)
    domain_order = ["demand", "inventory", "supplier", "transport", "mixed"]
    present = [d for d in domain_order if d in domain_dfs]
    # Any domains not in order (edge case)
    present += [d for d in domain_dfs if d not in present]

    master = domain_dfs[present[0]]
    for i in range(1, len(present)):
        right_domain = present[i]
        right_df = domain_dfs[right_domain]
        n_before = len(master)
        master = _merge_two(master, right_df, join_keys, how="outer")
        master.drop_duplicates(inplace=True)
        master.reset_index(drop=True, inplace=True)
        merge_log.append(
            f"✅ Merged '{present[i-1]}' + '{right_domain}' → "
            f"{len(master):,} rows × {master.shape[1]} cols "
            f"(was {n_before:,} rows)"
        )

    # LLM sanity check on the merged result
    cols_summary = master.columns.tolist()
    prompt = f"""You are validating a merged supply-chain DataFrame.
Files integrated: {[f[0] for f in files]}
Domains detected: {list(classified.values())}
Final columns: {cols_summary}
Final shape: {master.shape}

Reply ONLY with valid JSON:
{{
  "merge_quality": "good|fair|poor",
  "warnings": ["..."],
  "suggested_join_keys": ["..."],
  "notes": "..."
}}"""
    llm_raw = _parse(_llm([{"role": "user", "content": prompt}], json_mode=True))

    report = {
        "strategy": "multi_file_merge",
        "files": [
            {"name": f[0], "domain": classified[f[0]]["domain"],
             "rows": classified[f[0]]["rows"],
             "cols": classified[f[0]]["cols"]}
            for f in files
        ],
        "domain_groups": {d: len(dfs) for d, dfs in domain_groups.items()},
        "join_keys_used": join_keys[:3],
        "merge_log": merge_log,
        "llm_assessment": llm_raw,
        "final_shape": list(master.shape),
    }

    logger.info("[Integrator] Master df: %d rows × %d cols", *master.shape)
    return master, report


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 4 — RequirementValidatorAgent
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

    logger.info("[Validator] Enabled=%s  Blocked=%s", enabled, blocked)
    return {
        "module_status":    module_status,
        "enabled_modules":  enabled,
        "blocked_modules":  blocked,
        "all_missing_cols": all_missing,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 5 — FeatureEngineeringAgent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def FeatureEngineeringAgent(
    df: pd.DataFrame,
    missing_cols: list[str],
    profile: dict,
    schema_result: dict,
) -> dict:
    if not missing_cols:
        return {"derivable": [], "not_derivable": [], "steps": []}

    logger.info("[FeatureEng] Deriving: %s", missing_cols)
    avail = list(df.columns)
    mapping = schema_result["mapping"]

    prompt = f"""Supply chain feature engineering task.
Available columns: {avail}
Missing required columns: {missing_cols}
Current column mapping (required→original): {json.dumps({k:v for k,v in mapping.items() if v})}
Dataset type: {profile.get('llm_profile',{}).get('dataset_type','unknown')}

For EACH missing column, try to derive it from available columns.
Allowed operations ONLY: {sorted(_SAFE_OPS)}

Return ONLY valid JSON:
{{
  "derivable": ["col1"],
  "not_derivable": ["col2"],
  "steps": [
    {{
      "target_col": "missing_col",
      "operation": "safe_op",
      "source_cols": ["existing_col_a"],
      "params": {{}},
      "explanation": "reason"
    }}
  ]
}}

Rules:
- source_cols MUST be from {avail}
- Use fill_constant with value=0 if truly unknown
- supply_variation_days = actual_days - ideal_days
- fulfillment_rate = fulfilled_qty / ordered_qty
- inventory_status via derive_status: stock_col vs reorder_col
- delay_probability: fill_constant 0.3 if no data"""

    raw  = _parse(_llm([{"role": "user", "content": prompt}],
                       json_mode=True, max_tokens=1500))
    result: dict = raw if raw else {}

    # Safety: validate each step
    validated: list[dict] = []
    for step in result.get("steps", []):
        op  = step.get("operation", "")
        src = step.get("source_cols", [])
        if op not in _SAFE_OPS:
            logger.warning("[FeatureEng] Rejected unsafe op: %s", op)
            continue
        if op in ("fill_constant", "date_extract"):
            validated.append(step)
        elif all(c in avail for c in src):
            validated.append(step)
        else:
            logger.warning("[FeatureEng] Rejected step — src not in df: %s", src)

    result["steps"] = validated
    logger.info("[FeatureEng] %d steps validated", len(validated))
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 7 — TransformationEngine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def TransformationEngine(
    df: pd.DataFrame,
    steps: list[dict],
    mapping: dict,
) -> tuple[pd.DataFrame, list[str]]:
    df  = df.copy()
    log: list[str] = []

    # Alias mapped columns
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
                    df[target] = (
                        pd.to_numeric(df[src[0]], errors="coerce") / denom
                    ).fillna(0).clip(0, 1)
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
                    lo = float(params.get("min",
                               pd.to_numeric(df[col], errors="coerce").min()))
                    hi = float(params.get("max",
                               pd.to_numeric(df[col], errors="coerce").max()))
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
                    df[target] = np.log1p(
                        pd.to_numeric(df[col], errors="coerce").fillna(0))
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
                        "month": dts.dt.month,
                        "year":  dts.dt.year,
                        "dayofweek": dts.dt.dayofweek,
                        "quarter": dts.dt.quarter,
                    }
                    if part in part_map:
                        df[target] = part_map[part]
                        log.append(f"✅ date_extract({part}): '{target}' ← '{col}'")

        except Exception as exc:
            log.append(f"❌ Error op='{op}' target='{target}': {exc}")
            logger.error("[Transform] %s", exc)

    return df, log


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 6 — DataAssistantChatbot
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _parse_user_formula(user_msg: str, target_col: str,
                        df_cols: list[str]) -> dict | None:
    prompt = f"""Parse the user's instruction into a single safe transform step.
Target column to create: '{target_col}'
Available columns in dataset: {df_cols}
User instruction: "{user_msg}"
Allowed operations: {sorted(_SAFE_OPS)}

Return ONLY valid JSON (null if not parseable):
{{
  "target_col": "{target_col}",
  "operation": "safe_op_name",
  "source_cols": ["existing_col"],
  "params": {{}},
  "explanation": "..."
}}"""

    raw = _parse(_llm([{"role": "user", "content": prompt}],
                      json_mode=True, max_tokens=400))
    if not raw or "operation" not in raw:
        return None
    if raw.get("operation") not in _SAFE_OPS:
        return None
    return raw


def render_data_assistant(
    df: pd.DataFrame,
    missing_cols: list[str],
    pipeline_state: dict,
) -> tuple[pd.DataFrame, bool]:
    if not missing_cols:
        return df, False

    KEY = "assistant_chat"
    if KEY not in st.session_state:
        st.session_state[KEY] = []
        first_msg = (
            f"👋 I need a little help to fully prepare your dataset.\n\n"
            f"**Still missing:** {', '.join(f'`{c}`' for c in missing_cols)}\n\n"
            "For each one you can:\n"
            "- Type the **name** of an existing column that contains this data\n"
            "- Give a **formula** like `fulfillment_rate = fulfilled_qty / order_qty`\n"
            "- Say `skip` to leave it blank (that module will be disabled)\n"
            "- Say `fill 0` to fill with a default numeric value\n\n"
            f"Let's start — what should **`{missing_cols[0]}`** be?"
        )
        st.session_state[KEY].append({"role": "assistant", "content": first_msg})

    chat_box = st.container(height=300)
    with chat_box:
        for msg in st.session_state[KEY]:
            av = "🤖" if msg["role"] == "assistant" else "👤"
            with st.chat_message(msg["role"], avatar=av):
                st.markdown(msg["content"])

    user_input = st.chat_input("How to derive missing column?", key="da_input")
    needs_rerun = False

    if user_input:
        st.session_state[KEY].append({"role": "user", "content": user_input})

        resolved   = st.session_state.get("assistant_resolved", set())
        still_miss = [c for c in missing_cols if c not in resolved]

        if not still_miss:
            st.session_state[KEY].append({
                "role": "assistant",
                "content": "✅ All columns addressed! Refreshing pipeline…"
            })
            return df, True

        target = still_miss[0]
        txt    = user_input.strip()

        if txt.lower() == "skip":
            resolved.add(target)
            st.session_state["assistant_resolved"] = resolved
            nxt = [c for c in still_miss[1:]]
            reply = (f"⏭ Skipped `{target}`.\nNext: **`{nxt[0]}`**?" if nxt
                     else "✅ Done! Refreshing…")
            needs_rerun = not bool(nxt)
            st.session_state[KEY].append({"role": "assistant", "content": reply})
            return df, needs_rerun

        m_fill = re.match(r"fill\s+([+-]?\d*\.?\d+)", txt, re.IGNORECASE)
        if m_fill:
            val = float(m_fill.group(1))
            df[target] = val
            pipeline_state["mapping"][target] = target
            resolved.add(target)
            st.session_state["assistant_resolved"] = resolved
            nxt = [c for c in still_miss[1:]]
            reply = (f"✅ `{target}` filled with `{val}`.\nNext: **`{nxt[0]}`**?" if nxt
                     else "🎉 All done! Refreshing pipeline…")
            needs_rerun = not bool(nxt)
            st.session_state[KEY].append({"role": "assistant", "content": reply})
            return df, needs_rerun

        lower_cols = {c.lower(): c for c in df.columns}
        if txt.lower() in lower_cols:
            actual = lower_cols[txt.lower()]
            df[target] = df[actual]
            pipeline_state["mapping"][target] = actual
            resolved.add(target)
            st.session_state["assistant_resolved"] = resolved
            nxt = [c for c in still_miss[1:]]
            reply = (f"✅ Mapped `{actual}` → `{target}`.\nNext: **`{nxt[0]}`**?" if nxt
                     else "🎉 All done! Refreshing pipeline…")
            needs_rerun = not bool(nxt)
            st.session_state[KEY].append({"role": "assistant", "content": reply})
            return df, needs_rerun

        step = _parse_user_formula(txt, target, list(df.columns))
        if step:
            new_df, log = TransformationEngine(df, [step], pipeline_state.get("mapping", {}))
            if target in new_df.columns:
                df = new_df
                pipeline_state["mapping"][target] = target
                resolved.add(target)
                st.session_state["assistant_resolved"] = resolved
                nxt = [c for c in still_miss[1:]]
                reply = (f"✅ Derived `{target}` ({log[-1] if log else ''}).\n"
                         + (f"Next: **`{nxt[0]}`**?" if nxt else "🎉 All done! Refreshing…"))
                needs_rerun = not bool(nxt)
            else:
                reply = (f"⚠ Couldn't derive `{target}`. "
                         "Try a simpler formula or type a column name directly.")
        else:
            reply = (f"❓ I didn't understand that for `{target}`.\n"
                     "Try: column name · formula · `skip` · `fill 0`")

        st.session_state[KEY].append({"role": "assistant", "content": reply})

    return df, needs_rerun


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 8 — FinalValidator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def FinalValidator(df: pd.DataFrame, _prev: dict) -> dict:
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
    all_miss = sorted(
        {c for s in module_status.values() for c in s["missing"]} - DERIVED_COLS
    )
    logger.info("[FinalValidator] enabled=%s", enabled)
    return {
        "module_status":    module_status,
        "enabled_modules":  enabled,
        "blocked_modules":  blocked,
        "all_missing_cols": all_miss,
        "overall_ready":    len(enabled) >= 2,
        "columns_in_df":    list(avail),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SAVE PREPARED DATASET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_prepared_dataset(df: pd.DataFrame, source_name: str = "dataset") -> str:
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^\w]", "_", source_name.lower())[:30]
    path = PREPARED_DIR / f"{slug}_{ts}.csv"
    df.to_csv(path, index=False)
    logger.info("[Save] %s (%d rows)", path, len(df))
    return str(path)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MULTI-FILE PIPELINE HASH  (prevents rerunning on same file set)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _compute_multi_hash(files_bytes: list[tuple[str, bytes]]) -> str:
    h = hashlib.md5()
    for name, b in sorted(files_bytes, key=lambda x: x[0]):
        h.update(name.encode())
        h.update(b[:4096])   # first 4 KB per file is enough for change detection
    return h.hexdigest()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FULL PIPELINE ORCHESTRATOR  (multi-file aware)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_autonomous_pipeline(
    dfs: list[tuple[str, pd.DataFrame]],   # [(filename, df), ...]
    user_description: str = "",
) -> dict:
    """
    Runs the full 8-stage pipeline on one or more DataFrames.
    Stage 6 (chatbot) is rendered separately in the UI.

    Single-file call: pass [(filename, df)]
    Multi-file call:  pass [(f1, df1), (f2, df2), ...]
    """
    n = len(dfs)
    label = f"{n} file{'s' if n > 1 else ''}"

    # ── Stage 1: Profile each file individually ───────────────────────────────
    profiles: list[dict] = []
    file_entries: list[tuple[str, pd.DataFrame, dict]] = []
    for i, (fname, df) in enumerate(dfs):
        with st.spinner(f"🔍 Stage 1/{8} — Profiling '{fname}' ({i+1}/{n})…"):
            p = DataProfilerAgent(df, user_description)
            profiles.append(p)
            file_entries.append((fname, df, p))

    # ── Stage 2: Schema-map each file ─────────────────────────────────────────
    schemas: list[dict] = []
    for i, (fname, df, profile) in enumerate(file_entries):
        with st.spinner(f"🗺 Stage 2/8 — Mapping schema for '{fname}' ({i+1}/{n})…"):
            schemas.append(SchemaMapperAgent(profile))

    # ── Stage 3: Multi-file integration → master df ───────────────────────────
    with st.spinner(f"🔗 Stage 3/8 — Integrating {label}…"):
        master_df, integration_report = MultiFileIntegratorAgent(file_entries)

    # Re-profile the merged master
    with st.spinner("🔍 Stage 3/8 — Profiling merged dataset…"):
        master_profile = DataProfilerAgent(master_df, user_description)

    # ── Stage 4 (was 3): Validate requirements on master ─────────────────────
    with st.spinner("✅ Stage 4/8 — Validating requirements…"):
        master_schema = SchemaMapperAgent(master_profile)
        validation    = RequirementValidatorAgent(master_schema)

    # ── Stage 5 (was 4): Feature engineering ─────────────────────────────────
    missing = validation["all_missing_cols"]
    fe_result: dict = {"derivable": [], "not_derivable": missing, "steps": []}
    if missing:
        with st.spinner(f"⚙️ Stage 5/8 — Engineering {len(missing)} features…"):
            fe_result = FeatureEngineeringAgent(
                master_df, missing, master_profile, master_schema
            )

    # ── Stage 7: Transformations ──────────────────────────────────────────────
    with st.spinner("🔄 Stage 7/8 — Applying transforms…"):
        df_t, t_log = TransformationEngine(
            master_df, fe_result.get("steps", []), master_schema["mapping"]
        )

    # ── Stage 8: Final validation ─────────────────────────────────────────────
    final_v = FinalValidator(df_t, validation)

    # Build combined filename slug for saving
    combined_name = "_".join(
        re.sub(r"[^\w]", "_", f[0])[:12] for f in dfs[:3]
    )

    state = {
        # Multi-file metadata
        "filenames":          [f[0] for f in dfs],
        "file_count":         n,
        "integration_report": integration_report,
        "per_file_profiles":  profiles,
        "per_file_schemas":   schemas,
        # Pipeline artefacts
        "filename":           combined_name,
        "user_description":   user_description,
        "profile":            master_profile,
        "schema":             master_schema,
        "mapping":            master_schema["mapping"],
        "validation":         validation,
        "fe_result":          fe_result,
        "transform_log":      t_log,
        "final_validation":   final_v,
        "df_transformed":     df_t,
        "overall_ready":      final_v["overall_ready"],
        "enabled_modules":    final_v["enabled_modules"],
        "blocked_modules":    final_v["blocked_modules"],
        "still_missing":      final_v["all_missing_cols"],
        "ran_at":             datetime.datetime.utcnow().isoformat(),
    }

    if final_v["overall_ready"]:
        state["prepared_path"] = save_prepared_dataset(df_t, combined_name)

    st.session_state["pipeline_state"] = state
    _push_datasets_from_pipeline(state)
    return state


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UI — Integration Report Panel  (★ new)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_integration_report(ps: dict) -> None:
    """Show multi-file integration summary below the upload widget."""
    ir = ps.get("integration_report", {})
    if not ir or ir.get("strategy") == "single_file":
        return

    st.markdown("#### 🔗 Multi-File Integration Report")

    files_meta = ir.get("files", [])
    if files_meta:
        cols = st.columns(min(len(files_meta), 4))
        for i, fm in enumerate(files_meta):
            cols[i % 4].metric(
                label=f"📄 {fm['name'][:20]}",
                value=fm["domain"].upper(),
                delta=f"{fm['rows']:,} rows · {fm['cols']} cols",
            )

    st.markdown(f"**Strategy:** `{ir.get('strategy', 'n/a')}` | "
                f"**Join keys:** `{', '.join(ir.get('join_keys_used', [])) or 'none'}`")

    with st.expander("📋 Merge Log", expanded=False):
        for line in ir.get("merge_log", []):
            if line.startswith("✅"):
                st.success(line)
            elif line.startswith("⚠"):
                st.warning(line)
            elif line.startswith("❌"):
                st.error(line)
            else:
                st.caption(line)

    llm_assess = ir.get("llm_assessment", {})
    if llm_assess:
        quality = llm_assess.get("merge_quality", "unknown")
        colour  = {"good": "✅", "fair": "⚠️", "poor": "❌"}.get(quality, "ℹ️")
        st.info(f"{colour} **Merge quality:** {quality.upper()} — "
                f"{llm_assess.get('notes', '')}")
        for w in llm_assess.get("warnings", []):
            st.warning(w)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UI — Pipeline Status Panel
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_pipeline_status(ps: dict) -> None:
    if not ps:
        return

    fv    = ps.get("final_validation", {})
    ready = fv.get("overall_ready", False)
    ena   = fv.get("enabled_modules", [])
    blk   = fv.get("blocked_modules", [])
    miss  = fv.get("all_missing_cols", [])

    if ready:
        st.success(f"✅ **Dataset Ready** — {len(ena)} modules enabled")
    else:
        st.warning(f"⚠️ **Partial** — {len(ena)} modules ready, {len(blk)} need more data")

    # Multi-file badge
    fc = ps.get("file_count", 1)
    if fc > 1:
        st.info(f"🔗 **{fc} files merged** — "
                f"{', '.join(ps.get('filenames', []))}")

    profile = ps.get("profile", {})
    schema  = ps.get("schema", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows",        f"{profile.get('row_count', 0):,}")
    c2.metric("Columns",     f"{profile.get('col_count', 0)}")
    c3.metric("Cols Mapped", f"{schema.get('mapped_count', 0)}/{schema.get('total_required', 0)}")
    c4.metric("Modules On",  f"{len(ena)}/8")
    c5.metric("Missing",     f"{len(miss)}")

    # Integration report (multi-file only)
    render_integration_report(ps)

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

    # Per-file schema tabs (multi-file only)
    per_schemas = ps.get("per_file_schemas", [])
    per_profiles = ps.get("per_file_profiles", [])
    filenames = ps.get("filenames", [])
    if len(per_schemas) > 1:
        with st.expander("🗂️ Per-File Schema Mappings", expanded=False):
            tabs = st.tabs([f"📄 {fn[:20]}" for fn in filenames])
            for tab, fname, schema_i, profile_i in zip(
                tabs, filenames, per_schemas, per_profiles
            ):
                with tab:
                    st.caption(f"{profile_i.get('row_count',0):,} rows · "
                               f"{profile_i.get('col_count',0)} cols · "
                               f"domain: {profile_i.get('llm_profile',{}).get('dataset_type','?')}")
                    m = schema_i.get("mapping", {})
                    c = schema_i.get("confidence", {})
                    rows = [
                        {
                            "Required": rc,
                            "Mapped To": m.get(rc) or "—",
                            "Confidence": f"{c.get(rc,0):.0%}" if m.get(rc) else "—",
                            "Status": "✅" if m.get(rc) else "❌",
                        }
                        for rc in ALL_REQUIRED_COLS if m.get(rc)
                    ]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                                 hide_index=True)

    # Master mapping table
    with st.expander("🗺️ Master Column Mapping", expanded=False):
        mapping = ps.get("mapping", {})
        conf    = ps.get("schema", {}).get("confidence", {})
        rows = [
            {
                "Required Column": c,
                "Mapped To": mapping.get(c) or "—",
                "Confidence": f"{conf.get(c,0):.0%}" if mapping.get(c) else "—",
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

    llm_p = ps.get("profile", {}).get("llm_profile", {})
    if llm_p.get("dataset_summary"):
        st.info(f"🧠 **AI Understanding:** {llm_p['dataset_summary']}")

    df_t = ps.get("df_transformed")
    if df_t is not None:
        st.download_button(
            "⬇️ Download Prepared Master Dataset",
            data=df_t.to_csv(index=False).encode(),
            file_name="skvision_prepared_master.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_missing_fields_panel(ps: dict) -> None:
    miss = ps.get("still_missing", [])
    if not miss:
        return
    st.markdown("#### ❓ Missing Fields")
    fe = ps.get("fe_result", {})
    not_der = fe.get("not_derivable", [])
    for col in miss:
        msg = (f"**`{col}`** — auto-derivation failed"
               if col in not_der
               else f"**`{col}`** — needs confirmation")
        st.warning(msg)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PUBLIC UPLOAD SECTION  (★ Autonomous now accepts multiple CSVs)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_upload_section() -> dict[str, pd.DataFrame]:
    """
    Main upload UI.  Two modes:
      🧠 Autonomous — 1–N CSVs, any schema, full 8-stage pipeline with
                      automatic multi-file integration (Stage 3 new)
      📋 Classic    — 4 pre-structured CSVs (backward compat)
    """
    st.subheader("📂 Upload Datasets")

    mode = st.radio(
        "Mode",
        ["🧠 Autonomous (any CSV — AI maps & merges automatically)",
         "📋 Classic (pre-structured 4-file upload)"],
        horizontal=True,
        key="upload_mode",
    )

    # ── AUTONOMOUS ─────────────────────────────────────────────────────────────
    if mode.startswith("🧠"):
        st.caption(
            "Upload **one or more** supply chain CSV files. "
            "The AI pipeline profiles, classifies, merges, and prepares them "
            "into a single master dataset automatically."
        )

        uploaded_files = st.file_uploader(
            "Upload CSV(s)",
            type=["csv", "tsv"],
            accept_multiple_files=True,   # ★ Changed: now multi-file
            key="auto_uploader",
        )
        user_desc = st.text_area(
            "📝 Describe your datasets (optional but recommended)",
            placeholder=(
                "e.g. 'demand.csv = weekly sales by SKU; "
                "inventory.csv = warehouse stock levels; "
                "supplier.csv = vendor performance scores.'"
            ),
            height=75,
            key="user_desc",
        )

        if not uploaded_files:
            return get_datasets()

        # Read bytes + compute combined hash
        files_bytes: list[tuple[str, bytes]] = []
        for uf in uploaded_files:
            b = uf.read()
            files_bytes.append((uf.name, b))
        combo_hash = _compute_multi_hash(files_bytes)

        if st.session_state.get("_file_hash") != combo_hash:
            # New file set → reset pipeline
            st.session_state["_file_hash"]          = combo_hash
            st.session_state["pipeline_state"]      = {}
            st.session_state["assistant_chat"]      = []
            st.session_state["assistant_resolved"]  = set()

            loaded: list[tuple[str, pd.DataFrame]] = []
            for name, raw in files_bytes:
                df_raw = load_csv_bytes(raw, name)
                loaded.append((name, df_raw))
            st.session_state["_raw_dfs"] = loaded

        loaded = st.session_state.get("_raw_dfs", [])
        if not loaded:
            return get_datasets()

        # Per-file preview
        st.markdown(f"**{len(loaded)} file(s) loaded:**")
        prev_cols = st.columns(min(len(loaded), 4))
        for i, (fname, df_raw) in enumerate(loaded):
            prev_cols[i % 4].success(
                f"📄 **{fname}**\n\n{len(df_raw):,} rows × {df_raw.shape[1]} cols"
            )

        if len(loaded) > 1:
            with st.expander("👀 Preview all files", expanded=False):
                tabs = st.tabs([f"📄 {f[0][:20]}" for f in loaded])
                for tab, (fname, df_raw) in zip(tabs, loaded):
                    with tab:
                        st.dataframe(df_raw.head(5), use_container_width=True)

        ps = st.session_state.get("pipeline_state", {})

        if not ps:
            btn_label = (
                f"🚀 Run Autonomous Pipeline on {len(loaded)} file(s)"
                if len(loaded) > 1
                else "🚀 Run Autonomous Pipeline"
            )
            if st.button(btn_label, type="primary", use_container_width=True):
                ps = run_autonomous_pipeline(loaded, user_description=user_desc)
                st.rerun()
        else:
            st.divider()
            render_pipeline_status(ps)

            still_miss = ps.get("still_missing", [])
            if still_miss:
                st.divider()
                render_missing_fields_panel(ps)
                st.markdown("#### 🤖 Data Assistant")
                df_t = ps.get("df_transformed", loaded[0][1])
                df_t, needs_rerun = render_data_assistant(df_t, still_miss, ps)

                if needs_rerun:
                    ps["df_transformed"] = df_t
                    fv = FinalValidator(df_t, ps["final_validation"])
                    ps.update({
                        "final_validation": fv,
                        "still_missing":    fv["all_missing_cols"],
                        "enabled_modules":  fv["enabled_modules"],
                        "overall_ready":    fv["overall_ready"],
                    })
                    if fv["overall_ready"]:
                        ps["prepared_path"] = save_prepared_dataset(
                            df_t, ps["filename"]
                        )
                    st.session_state["pipeline_state"] = ps
                    _push_datasets_from_pipeline(ps)
                    st.rerun()

            if st.button("🔄 Re-run Pipeline", key="btn_rerun"):
                for k in ("pipeline_state", "assistant_chat",
                          "assistant_resolved", "_file_hash", "_raw_dfs"):
                    st.session_state.pop(k, None)
                st.rerun()

        return get_datasets()

    # ── CLASSIC ────────────────────────────────────────────────────────────────
    st.caption("Upload up to 4 CSVs: demand · inventory · supplier · transport.")
    uploaded_files = st.file_uploader(
        "Upload CSV files",
        type=["csv", "tsv"],
        accept_multiple_files=True,
        key="classic_uploader",
    )
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
    _log_upload([f.name for f in uploaded_files], datasets)

    cols = st.columns(len(datasets)) if datasets else []
    for (dt, df), col in zip(datasets.items(), cols):
        col.metric(dt.upper(), f"{len(df):,} rows", f"{df.shape[1]} cols")

    return datasets


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  INTERNAL HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _push_datasets_from_pipeline(ps: dict) -> None:
    """Populate session_state["datasets"] from the unified pipeline df."""
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
    _log_upload(ps.get("filenames", [ps.get("filename", "unknown")]), ds)


def _detect_dtype_classic(df: pd.DataFrame) -> str:
    cols = set(df.columns.str.lower())
    for dt, req in SCHEMA_HINTS.items():
        if all(c.lower() in cols for c in req):
            return dt
    return "unknown"


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
        "In Autonomous mode a single CSV (or multi-CSV) is enough."
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
    """Dev convenience: load reference CSVs if present."""
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
    return ds