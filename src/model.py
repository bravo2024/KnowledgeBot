from __future__ import annotations
"""TF-IDF retriever and extractive answer generator for KnowledgeBot.

The retriever builds a sparse TF-IDF document matrix (sublinear TF, L2
normalised) and ranks passages by cosine similarity to the query (computed
efficiently as a linear kernel over the normalised vectors).  Generation is
*extractive*: the answer is the single sentence from the top passage that
shares the most query terms, which keeps the system fully offline (no
external LLM calls).
"""
import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an the and or of to in on for is are was were be been being with as at by "
    "this that these those it its from your you we our they their i how do what "
    "when where which why can should".split()
)


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2 and t not in _STOPWORDS]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


class TfidfRetriever:
    """Sparse TF-IDF retriever with cosine-similarity ranking."""

    def __init__(self, ngram_range: tuple[int, int] = (1, 2), min_df: int = 1) -> None:
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizer: TfidfVectorizer | None = None
        self.doc_matrix = None
        self.documents: list[dict[str, Any]] = []
        self.id_to_idx: dict[str, int] = {}

    def fit(self, documents: list[dict[str, Any]]) -> "TfidfRetriever":
        self.documents = list(documents)
        self.id_to_idx = {d["id"]: i for i, d in enumerate(self.documents)}
        contents = [f"{d.get('title', '')}. {d.get('content', '')}" for d in self.documents]
        self.vectorizer = TfidfVectorizer(
            tokenizer=_tokenize,
            token_pattern=None,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,
            norm="l2",
        )
        self.doc_matrix = self.vectorizer.fit_transform(contents)
        return self

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if self.vectorizer is None or self.doc_matrix is None:
            raise RuntimeError("Retriever has not been fitted; call fit() first.")
        qv = self.vectorizer.transform([query])
        scores = linear_kernel(qv, self.doc_matrix).ravel()
        k = min(k, len(self.documents))
        top_idx = np.argsort(scores)[::-1][:k]
        results: list[dict[str, Any]] = []
        for i in top_idx:
            d = self.documents[i]
            results.append({
                "doc_id": d["id"],
                "title": d.get("title", ""),
                "category": d.get("category", ""),
                "score": float(scores[i]),
                "content": d.get("content", ""),
            })
        return results

    def extractive_answer(self, query: str, k: int = 3) -> dict[str, Any]:
        hits = self.retrieve(query, k=k)
        if not hits:
            return {"answer": "", "source_doc_id": None, "source_title": "",
                    "score": 0.0, "context": "", "retrieved": []}
        top = hits[0]
        q_tokens = set(_tokenize(query))
        sentences = _sentences(top["content"])
        if sentences and q_tokens:
            answer = max(sentences, key=lambda s: len(q_tokens & set(_tokenize(s))))
        elif sentences:
            answer = sentences[0]
        else:
            answer = top["content"]
        return {
            "answer": answer,
            "source_doc_id": top["doc_id"],
            "source_title": top["title"],
            "score": top["score"],
            "context": top["content"],
            "retrieved": hits,
        }


def build_tfidf_index(documents: list[dict[str, Any]]) -> TfidfRetriever:
    """Convenience wrapper: fit and return a TfidfRetriever."""
    return TfidfRetriever().fit(documents)


def retrieve(index: TfidfRetriever, query: str, k: int = 5) -> list[dict[str, Any]]:
    return index.retrieve(query, k)


def extractive_answer(index: TfidfRetriever, query: str, k: int = 3) -> dict[str, Any]:
    return index.extractive_answer(query, k)
