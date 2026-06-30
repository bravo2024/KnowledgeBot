from __future__ import annotations
"""KnowledgeBot — Streamlit dashboard for an enterprise knowledge-base RAG.

The retriever is built in-memory from the synthetic corpus (no external LLM,
no stale artifacts).  Users can ask questions, inspect the extracted answer
and its source passage, and review retrieval quality over the labelled eval
queries.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st

from src.data import make_corpus
from src.model import build_tfidf_index
from src.evaluate import evaluate_retriever
from src.visualizations import (
    plot_corpus_by_category,
    plot_recall_at_k,
    plot_retrieval_scores,
)

st.set_page_config(page_title="KnowledgeBot | Wipro Enterprise KB RAG", layout="wide", page_icon="📚")


@st.cache_resource
def load_retriever():
    corpus = make_corpus()
    return build_tfidf_index(corpus["documents"]), corpus


retriever, corpus = load_retriever()

with st.sidebar:
    st.header("⚙ Knowledge base")
    st.metric("Documents", corpus["n_documents"])
    st.metric("Eval queries", corpus["n_queries"])
    st.caption("Wipro · Enterprise knowledge-base RAG — TF-IDF retrieval + extractive generation")
    st.pyplot(plot_corpus_by_category(corpus["documents"]))

st.title("📚 KnowledgeBot — Enterprise Knowledge RAG")
st.caption(
    "Ask a question and receive a grounded, extractive answer drawn from an "
    "internal policy / FAQ knowledge base. Ranking uses cosine similarity over "
    "an L2-normalised TF-IDF index."
)

tab_q, tab_eval, tab_browse = st.tabs(["🔍 Ask a question", "📊 Retrieval evaluation", "🗂 Browse corpus"])

with tab_q:
    query = st.text_input("Ask a question", value="What is the password policy?")
    k = st.slider("Top-k passages", 1, 10, 5)
    if query.strip():
        results = retriever.retrieve(query, k=k)
        ans = retriever.extractive_answer(query, k=k)
        st.subheader("Extractive answer")
        st.success(ans["answer"])
        st.caption(f"Source: {ans['source_doc_id']} — {ans['source_title']}  (score {ans['score']:.3f})")
        st.pyplot(plot_retrieval_scores(results))
        with st.expander("Retrieved passages"):
            for r in results:
                st.markdown(f"**{r['doc_id']} · {r['title']}** ({r['category']}) — `{r['score']:.3f}`")
                st.write(r["content"])

with tab_eval:
    metrics = evaluate_retriever(retriever, corpus["queries"], k_values=(1, 3, 5))
    agg = metrics["aggregate"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall@5", f"{agg['recall@5']:.3f}")
    c2.metric("MRR", f"{agg['mrr']:.3f}")
    c3.metric("Precision@5", f"{agg['precision@5']:.3f}")
    c4.metric("Faithfulness", f"{agg['faithfulness']:.3f}")
    st.pyplot(plot_recall_at_k(metrics))
    cols = ["query_id", "query", "top_doc_id", "mrr", "recall@1", "recall@5", "faithfulness"]
    st.dataframe(pd.DataFrame(metrics["per_query"])[cols], use_container_width=True)

with tab_browse:
    df = pd.DataFrame(corpus["documents"])
    cat = st.selectbox("Filter by category", ["All"] + corpus["categories"])
    view = df if cat == "All" else df[df["category"] == cat]
    st.dataframe(view[["id", "title", "category", "content"]], use_container_width=True)
