# KnowledgeBot

> Enterprise **Retrieval-Augmented Generation** knowledge base — a TF-IDF retriever
> with extractive answer generation and rank-aware retrieval evaluation.

KnowledgeBot indexes a synthetic enterprise knowledge base of policy and FAQ
documents (IT, HR, Benefits, Security, Travel, Equipment) and answers employee
questions by retrieving the most relevant passage and extracting the best
supporting sentence. The system is fully offline: there are **no external LLM
API calls** — generation is extractive (the answer is a verbatim sentence drawn
from the retrieved context).

## Methodology

1. **Corpus** — 42 hand-authored knowledge-base documents across 6 categories,
   plus 18 labelled evaluation queries with known relevant document ids.
2. **Indexing** — a scikit-learn `TfidfVectorizer` builds a sparse document-term
   matrix with sublinear TF (`1 + log(tf)`) and L2 row normalisation. Bigrams
   are included to capture short phrases.
3. **Retrieval** — the query is vectorised with the fitted vocabulary and ranked
   against every document by **cosine similarity**, computed efficiently as a
   linear kernel over the normalised vectors.
4. **Generation (extractive)** — the top passage is selected, then the single
   sentence sharing the most query terms is returned as the answer. This
   guarantees the answer is grounded in retrieved evidence.
5. **Evaluation** — rank-aware metrics computed over the labelled queries.

### Metrics

| Metric | Description |
|---|---|
| Recall@k | fraction of relevant documents found in the top-k |
| Precision@k | fraction of the top-k that are relevant |
| MRR | reciprocal rank of the first relevant document |
| NDCG@k | normalised discounted cumulative gain (binary relevance) |
| MAP | mean average precision |
| Faithfulness | SQuAD-style token-F1 between the answer and its context |

## Quickstart

```bash
pip install -r requirements.txt
python train.py          # build the index, evaluate, save to models/
pytest -q                # run the smoke / evaluation tests
streamlit run app.py     # launch the dashboard
```

## Repository structure

```
KnowledgeBot/
  src/
    data.py          synthetic KB corpus + labelled eval queries
    model.py         TfidfRetriever (fit / retrieve / extractive_answer)
    core.py          Recall@k, Precision@k, MRR, NDCG, MAP, faithfulness
    evaluate.py      end-to-end retrieval evaluation harness
    visualizations.py RAG-specific plots
    persist.py       pickle save/load helpers
  train.py           build index + write metrics
  app.py             Streamlit RAG dashboard
  tests/             pytest smoke tests
```

## Reference

Lewis, P., Perez, E., Piktus, A., et al. (2020). *Retrieval-Augmented Generation
for Knowledge-Intensive NLP Tasks.* NeurIPS.

## License

MIT
