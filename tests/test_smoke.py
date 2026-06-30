from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import make_corpus
from src.model import TfidfRetriever, build_tfidf_index
from src.core import recall_at_k, precision_at_k, mrr, ndcg_at_k, faithfulness
from src.evaluate import evaluate_retriever


def test_corpus_shape_and_integrity():
    c = make_corpus()
    assert c["n_documents"] >= 30
    assert c["n_queries"] >= 15
    ids = {d["id"] for d in c["documents"]}
    assert len(ids) == c["n_documents"]  # ids are unique
    for q in c["queries"]:
        assert q["relevant_doc_ids"], f"query {q['id']} has no relevant docs"
        assert all(r in ids for r in q["relevant_doc_ids"])


def test_retriever_fit_and_ranking():
    c = make_corpus()
    r = TfidfRetriever().fit(c["documents"])
    res = r.retrieve("password policy remote access", k=5)
    assert len(res) == 5
    assert res[0]["score"] >= res[-1]["score"]  # sorted descending
    assert all("doc_id" in x and "content" in x and "score" in x for x in res)


def test_extractive_answer_returns_grounded_text():
    c = make_corpus()
    r = build_tfidf_index(c["documents"])
    ans = r.extractive_answer("How many days of paid time off do I get?", k=3)
    assert isinstance(ans["answer"], str) and len(ans["answer"]) > 0
    assert ans["source_doc_id"] in {d["id"] for d in c["documents"]}
    # the extractive answer must be a substring of the source context
    assert ans["answer"] in ans["context"]


def test_metrics_unit_behaviour():
    assert recall_at_k(["a", "b", "c"], ["b"], 3) == 1.0
    assert recall_at_k(["a", "b", "c"], ["z"], 3) == 0.0
    assert precision_at_k(["a", "b"], ["b"], 2) == 0.5
    assert abs(mrr(["a", "b", "c"], ["b"]) - 0.5) < 1e-9
    assert mrr(["a", "b", "c"], ["z"]) == 0.0
    assert 0.0 <= ndcg_at_k(["a", "b", "c"], ["c"], 3) <= 1.0
    f = faithfulness("the password policy", "password policy requires eight characters")
    assert 0.0 <= f <= 1.0 and f > 0.0


def test_end_to_end_evaluation():
    c = make_corpus()
    r = build_tfidf_index(c["documents"])
    m = evaluate_retriever(r, c["queries"], k_values=(1, 3, 5))
    agg = m["aggregate"]
    assert agg["n_queries"] == c["n_queries"]
    for key in ("recall@1", "recall@5", "precision@5", "mrr", "faithfulness"):
        assert 0.0 <= agg[key] <= 1.0
    # quality gate: TF-IDF over a small clean corpus should retrieve well
    assert agg["recall@5"] > 0.3
    assert agg["mrr"] > 0.2
    assert len(m["per_query"]) == c["n_queries"]
