from __future__ import annotations
"""RAG-specific plots for the KnowledgeBot Streamlit dashboard."""
from collections import Counter
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

THEME = {
    "bg": "#0e1117", "fg": "#ffffff", "grid": "#1a1f2e",
    "cyan": "#22d3ee", "violet": "#a78bfa", "orange": "#f97316",
    "rose": "#f43f5e", "amber": "#fbbf24", "green": "#22c55e",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": THEME["bg"], "axes.facecolor": THEME["bg"],
        "axes.edgecolor": THEME["grid"], "axes.labelcolor": THEME["fg"],
        "text.color": THEME["fg"], "xtick.color": THEME["fg"],
        "ytick.color": THEME["fg"], "grid.color": THEME["grid"],
        "grid.alpha": 0.3, "legend.facecolor": "#1a1f2e",
        "legend.edgecolor": THEME["grid"], "legend.labelcolor": THEME["fg"],
    })


def plot_corpus_by_category(documents: list[dict[str, Any]]):
    _style()
    counts = Counter(d["category"] for d in documents)
    labels = sorted(counts)
    vals = [counts[l] for l in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [THEME["cyan"], THEME["violet"], THEME["orange"],
              THEME["green"], THEME["amber"], THEME["rose"]]
    ax.bar(labels, vals, color=colors[: len(labels)], alpha=0.85)
    ax.set_title("Knowledge-base documents by category", color=THEME["fg"])
    ax.tick_params(axis="x", rotation=20, labelsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    return fig


def plot_recall_at_k(metrics: dict[str, Any], title: str = "Retrieval Recall / Precision @k"):
    _style()
    agg = metrics["aggregate"]
    ks = sorted(int(k.split("@")[1]) for k in agg if k.startswith("recall@"))
    recalls = [agg[f"recall@{k}"] for k in ks]
    precisions = [agg[f"precision@{k}"] for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, recalls, "o-", color=THEME["cyan"], lw=2, label="Recall@k")
    ax.plot(ks, precisions, "s-", color=THEME["violet"], lw=2, label="Precision@k")
    ax.set_xlabel("k")
    ax.set_ylabel("score")
    ax.set_title(title, color=THEME["fg"])
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    return fig


def plot_score_distribution(results: list[dict[str, Any]], title: str = "Retrieval cosine scores"):
    _style()
    scores = [r["score"] for r in results]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=min(20, max(5, len(scores))),
            color=THEME["cyan"], alpha=0.8, edgecolor=THEME["grid"])
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("documents")
    ax.set_title(title, color=THEME["fg"])
    ax.grid(True, alpha=0.2)
    return fig


def plot_retrieval_scores(query_results: list[dict[str, Any]], title: str = "Top passage scores"):
    _style()
    labels = [f"{r['doc_id']}  ({r['score']:.2f})" for r in query_results]
    vals = [r["score"] for r in query_results]
    fig, ax = plt.subplots(figsize=(8, max(3, len(vals) * 0.4)))
    ax.barh(range(len(vals)), vals, color=THEME["green"], alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("cosine similarity")
    ax.set_title(title, color=THEME["fg"])
    ax.grid(True, alpha=0.2, axis="x")
    return fig
