from __future__ import annotations
import numpy as np; import pandas as pd
FEATURE_NAMES = ["query_length","embedding_cosine_sim","passage_retrieval_score","reranker_score","answer_confidence","knowledge_graph_depth","document_recency","topic_match","user_expertise","session_length"]
CATEGORICAL_FEATURES = ["knowledge_graph_depth"]
NUMERICAL_FEATURES = ["query_length","embedding_cosine_sim","passage_retrieval_score","reranker_score","answer_confidence","document_recency","topic_match","user_expertise","session_length"]
TARGET_NAME = "answer_helpful"
def make_synthetic(n=10000,seed=42):
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({
        "query_length": rng.poisson(lam=60,size=n).clip(3,300).astype(int),
        "embedding_cosine_sim": rng.beta(7,3,size=n).round(4),
        "passage_retrieval_score": rng.beta(6,3,size=n).round(4),
        "reranker_score": rng.beta(5,3,size=n).round(4),
        "answer_confidence": rng.beta(6,2,size=n).round(3),
        "knowledge_graph_depth": rng.choice([1,2,3,4,5],size=n,p=[0.30,0.30,0.20,0.12,0.08]),
        "document_recency": rng.uniform(0,1,size=n).round(3),
        "topic_match": rng.beta(7,3,size=n).round(3),
        "user_expertise": rng.uniform(1,10,size=n).round(1),
        "session_length": rng.poisson(lam=5,size=n).clip(1,30),
    })
    embed=df["embedding_cosine_sim"]; ret=df["passage_retrieval_score"]; rerank=df["reranker_score"]
    conf=df["answer_confidence"]; kg=np.clip(df["knowledge_graph_depth"]/5,0,1); rec=df["document_recency"]
    topic=df["topic_match"]; expert=df["user_expertise"]/10; sess=np.clip(df["session_length"]/30,0,1)
    log_odds = 1.5 + 1.5*embed + 1.2*ret + 1.0*rerank + 1.2*conf + 0.8*kg + 1.0*rec + 1.2*topic + 0.8*expert - 0.5*sess + rng.normal(0,0.4,size=n)
    prob=1/(1+np.exp(-log_odds)); y=(prob>np.percentile(prob,65)).astype(np.float64)
    return {"X":df,"y":y,"features":FEATURE_NAMES,"df":df.assign(answer_helpful=y),"categorical_features":CATEGORICAL_FEATURES,"numerical_features":NUMERICAL_FEATURES,"n_samples":n,"n_features":len(FEATURE_NAMES),"positive_rate":y.mean()}
