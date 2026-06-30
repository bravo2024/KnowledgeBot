from __future__ import annotations
"""End-to-end retrieval evaluation harness for the KnowledgeBot RAG system.

Runs the fitted retriever over the labelled eval queries and aggregates the
rank-aware metrics from ``src.core``.
"""
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.core import (
    average_precision_at_k,
    faithfulness,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def evaluate_retriever(retriever, queries: list[dict[str, Any]], k_values: tuple[int, ...] = (1, 3, 5)) -> dict[str, Any]:
    """Evaluate a retriever against labelled queries.

    Returns a dict with ``aggregate`` metrics and per-query ``per_query`` rows.
    """
    max_k = max(k_values)
    per_query: list[dict[str, Any]] = []
    for q in queries:
        retrieved = retriever.retrieve(q["query"], k=max_k)
        retrieved_ids = [r["doc_id"] for r in retrieved]
        rel = q.get("relevant_doc_ids", [])
        ans = retriever.extractive_answer(q["query"], k=3)
        row: dict[str, Any] = {
            "query_id": q["id"],
            "query": q["query"],
            "mrr": mrr(retrieved_ids, rel),
            "map": average_precision_at_k(retrieved_ids, rel, max_k),
            "faithfulness": faithfulness(ans["answer"], ans["context"]),
            "top_doc_id": retrieved_ids[0] if retrieved_ids else None,
            "relevant_doc_ids": rel,
        }
        for k in k_values:
            row[f"recall@{k}"] = recall_at_k(retrieved_ids, rel, k)
            row[f"precision@{k}"] = precision_at_k(retrieved_ids, rel, k)
            row[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, rel, k)
        per_query.append(row)

    agg: dict[str, Any] = {}
    for k in k_values:
        agg[f"recall@{k}"] = float(np.mean([r[f"recall@{k}"] for r in per_query]))
        agg[f"precision@{k}"] = float(np.mean([r[f"precision@{k}"] for r in per_query]))
        agg[f"ndcg@{k}"] = float(np.mean([r[f"ndcg@{k}"] for r in per_query]))
    agg["mrr"] = float(np.mean([r["mrr"] for r in per_query]))
    agg["map"] = float(np.mean([r["map"] for r in per_query]))
    agg["faithfulness"] = float(np.mean([r["faithfulness"] for r in per_query]))
    agg["n_queries"] = len(per_query)
    return {"aggregate": agg, "per_query": per_query, "k_values": list(k_values)}


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_metrics(metrics: dict[str, Any], path: str = "models/metrics.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_jsonable(metrics), f, indent=2)


def print_report(metrics: dict[str, Any]) -> None:
    agg = metrics["aggregate"]
    print("=" * 54)
    print("  KnowledgeBot retrieval evaluation")
    print("=" * 54)
    order = ["n_queries", "recall@1", "recall@3", "recall@5",
             "precision@1", "precision@3", "precision@5",
             "mrr", "map", "ndcg@5", "faithfulness"]
    for key in order:
        if key not in agg:
            continue
        val = agg[key]
        if isinstance(val, float):
            print(f"    {key:14s}: {val:.4f}")
        else:
            print(f"    {key:14s}: {val}")
