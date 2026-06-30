from __future__ import annotations
"""Build the KnowledgeBot TF-IDF index and write retrieval metrics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import argparse

from src.data import make_corpus
from src.model import build_tfidf_index
from src.evaluate import evaluate_retriever, save_metrics, print_report
from src.persist import save_model


def main() -> None:
    p = argparse.ArgumentParser(description="Train the KnowledgeBot TF-IDF RAG index")
    p.add_argument("--k", type=int, default=5, help="max k for evaluation")
    args = p.parse_args()

    corpus = make_corpus()
    print(f"Corpus: {corpus['n_documents']} documents, {corpus['n_queries']} eval queries")

    retriever = build_tfidf_index(corpus["documents"])
    metrics = evaluate_retriever(retriever, corpus["queries"], k_values=(1, 3, args.k))
    print_report(metrics)

    save_model(retriever, path="models/model.pkl")
    save_metrics(metrics, path="models/metrics.json")
    print("Saved index -> models/model.pkl and metrics -> models/metrics.json")


if __name__ == "__main__":
    main()
