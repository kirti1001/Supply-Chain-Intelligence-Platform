"""
db_ops.py — Centralized MongoDB read/write helpers.
All intermediate & final results are stored as structured JSON documents.
"""

from __future__ import annotations
import json
import datetime
from typing import Any
import streamlit as st
from bson import ObjectId
from settings import get_collection


# ─── JSON serialization helper ───────────────────────────────────────────────
def _serialize(obj: Any) -> Any:
    """Recursively make objects JSON/BSON safe."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if hasattr(obj, "item"):          # numpy scalar
        return obj.item()
    if hasattr(obj, "tolist"):        # numpy array
        return obj.tolist()
    return obj


def _ts() -> str:
    return datetime.datetime.utcnow().isoformat()


# ─── Generic save / load ─────────────────────────────────────────────────────
def save_result(collection_name: str, payload: dict) -> str | None:
    """Insert a document; return inserted_id as string."""
    col = get_collection(collection_name)
    if col is None:
        st.warning("MongoDB not connected – result not saved.")
        return None
    payload = _serialize(payload)
    payload.setdefault("created_at", _ts())
    result = col.insert_one(payload)
    return str(result.inserted_id)


def load_latest(collection_name: str, filters: dict | None = None, n: int = 1) -> list[dict]:
    """Return last n documents from a collection, optionally filtered."""
    col = get_collection(collection_name)
    if col is None:
        return []
    q = filters or {}
    cursor = col.find(q).sort("created_at", -1).limit(n)
    docs = []
    for d in cursor:
        d["_id"] = str(d["_id"])
        docs.append(d)
    return docs


def load_all(collection_name: str, filters: dict | None = None) -> list[dict]:
    return load_latest(collection_name, filters, n=10_000)


def delete_document(collection_name: str, doc_id: str) -> bool:
    col = get_collection(collection_name)
    if col is None:
        return False
    res = col.delete_one({"_id": ObjectId(doc_id)})
    return res.deleted_count > 0


def load_by_id(collection_name: str, doc_id: str) -> dict | None:
    col = get_collection(collection_name)
    if col is None:
        return None
    doc = col.find_one({"_id": ObjectId(doc_id)})
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc


# ─── Typed helpers for each module ──────────────────────────────────────────
COLL_FORECAST  = "demand_forecasts"
COLL_RISK      = "risk_assessments"
COLL_INVENTORY = "inventory_results"
COLL_SEASONAL  = "seasonality_results"
COLL_STOCKOUT  = "stockout_predictions"
COLL_SUPPLIER  = "supplier_risk"
COLL_ROUTES    = "route_optimizations"
COLL_REPORTS   = "reports"
COLL_UPLOADS   = "uploaded_datasets"


def save_forecast(product: str, horizon: int, result: dict) -> str | None:
    return save_result(COLL_FORECAST, {
        "product": product,
        "horizon_days": horizon,
        "result": result,
        "module": "demand_forecast",
    })


def save_risk(scope: str, result: dict) -> str | None:
    return save_result(COLL_RISK, {
        "scope": scope,
        "result": result,
        "module": "risk_assessment",
    })


def save_inventory(product: str, warehouse: str, result: dict) -> str | None:
    return save_result(COLL_INVENTORY, {
        "product": product,
        "warehouse": warehouse,
        "result": result,
        "module": "inventory_management",
    })


def save_seasonal(product: str, result: dict) -> str | None:
    return save_result(COLL_SEASONAL, {
        "product": product,
        "result": result,
        "module": "seasonality",
    })


def save_stockout(product: str, result: dict) -> str | None:
    return save_result(COLL_STOCKOUT, {
        "product": product,
        "result": result,
        "module": "stockout",
    })


def save_supplier_risk(supplier: str, result: dict) -> str | None:
    return save_result(COLL_SUPPLIER, {
        "supplier": supplier,
        "result": result,
        "module": "supplier_risk",
    })


def save_route(result: dict) -> str | None:
    return save_result(COLL_ROUTES, {
        "result": result,
        "module": "route_optimization",
    })


def save_report(title: str, report_type: str, content: dict) -> str | None:
    return save_result(COLL_REPORTS, {
        "title": title,
        "report_type": report_type,
        "content": content,
        "module": "report",
    })


def get_latest_forecast(product: str) -> dict | None:
    docs = load_latest(COLL_FORECAST, {"product": product}, n=1)
    return docs[0] if docs else None


def get_latest_inventory(product: str, warehouse: str | None = None) -> dict | None:
    flt: dict = {"product": product}
    if warehouse:
        flt["warehouse"] = warehouse
    docs = load_latest(COLL_INVENTORY, flt, n=1)
    return docs[0] if docs else None
