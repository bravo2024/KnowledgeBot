from __future__ import annotations
"""Rank-aware retrieval and answer-grounding metrics for the KnowledgeBot RAG.

All metrics operate on a list of retrieved document ids together with the
ground-truth relevant ids.

Metrics
-------
recall_at_k      - fraction of relevant documents found in the top-k
precision_at_k   - fraction of the top-k that are relevant
mrr              - reciprocal rank of the first relevant document
ndcg_at_k        - normalised discounted cumulative gain (binary relevance)
faithfulness     - SQuAD-style token-F1 between an answer and its context
average_precision_at_k - AP@k (building block for MAP)
"""
import math
import re
from collections import Counter
from typing import Iterable, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an the and or of to in on for is are was were be been being with as at by "
    "this that these those it its from your you we our they their i".split()
)


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2 and t not in _STOPWORDS]


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant or k <= 0:
        return 0.0
    top = list(retrieved_ids)[:k]
    hits = sum(1 for r in top if r in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = set(relevant_ids)
    top = list(retrieved_ids)[:k]
    hits = sum(1 for r in top if r in relevant)
    return hits / k


def mrr(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    relevant = set(relevant_ids)
    for rank, r in enumerate(retrieved_ids, start=1):
        if r in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = set(relevant_ids)
    if k <= 0:
        return 0.0
    dcg = 0.0
    for i, r in enumerate(retrieved_ids[:k], start=1):
        if r in relevant:
            dcg += 1.0 / math.log2(i + 1)
    n_rel = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


def faithfulness(answer: str, context: str) -> float:
    """Token-level F1 between the generated answer and the retrieved context.

    Measures how strongly every answer token is grounded in the supporting
    passage (an extractive answer drawn from the context scores highly).
    """
    ans = _tokenize(answer)
    ctx = _tokenize(context)
    if not ans or not ctx:
        return 0.0
    ans_counts = Counter(ans)
    ctx_counts = Counter(ctx)
    common = sum((ans_counts & ctx_counts).values())
    if common == 0:
        return 0.0
    precision = common / len(ans)
    recall = common / len(ctx)
    return 2 * precision * recall / (precision + recall)


def average_precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Average precision truncated at k (building block for MAP)."""
    relevant = set(relevant_ids)
    if not relevant or k <= 0:
        return 0.0
    hits = 0
    score = 0.0
    for i, r in enumerate(retrieved_ids[:k], start=1):
        if r in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k)
