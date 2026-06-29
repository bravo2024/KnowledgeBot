# KnowledgeBot

> Enterprise RAG knowledge base with retrieval analytics and answer helpfulness prediction.

Trains four classifiers on synthetic enterprise query data to predict whether retrieved answers will be helpful. Dashboard provides RAG pipeline stage analysis (embedding similarity, reranker scores, passage retrieval), knowledge graph depth effects, and system performance metrics.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

Best model (Logistic Regression) holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 0.880 |
| Gini | 0.760 |
| KS Statistic | 0.658 |
| F1 Score | 0.774 |
| Accuracy | 0.832 |

5-fold CV AUC: 0.869 ± 0.022. Four models compared.

## Features

| Tab | What it does |
|---|---|
| **Explorer** | Query records overview, answer helpfulness distribution, feature descriptions |
| **Model Lab** | Multi-model comparison, ROC curves, calibration plots, CV results |
| **Retrieval Analytics** | Embedding cosine similarity and reranker score distributions by helpfulness, KG depth analysis |
| **Performance** | Passage retrieval score analysis, Precision@K metrics, system latency breakdown |

## Repo Structure

```
KnowledgeBot/
  src/         data, model, evaluate, persist modules
  train.py     training pipeline (multi-model + CV)
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic enterprise RAG dataset: embedding cosine similarity, reranker score, passage retrieval score, knowledge graph depth, query complexity, context relevance, and answer helpfulness label.

## License

MIT
