# Retrieval on SQuAD 2.0 dev, with Gemini, on a sample of questions

1204 paragraphs, 366 of them restricted. Fusion weights tuned on 200 questions from articles 1 to 18; every row below is measured on the 400 answerable questions from articles 19 to 35, asked by a user who may read the gold paragraph but not the restricted ones.

| retriever | recall@1 | recall@5 | recall@20 | mrr@20 | p50 ms | p99 ms | leaks |
|---|---|---|---|---|---|---|---|
| BM25 | 0.812 | 0.953 | 0.983 | 0.873 | 0.07 | 0.13 | 0 |
| hashing (lexical stand in) | 0.568 | 0.725 | 0.845 | 0.642 | 2.39 | 5.88 | 0 |
| WordLlama dense | 0.605 | 0.86 | 0.968 | 0.714 | 0.65 | 1.12 | 0 |
| hybrid: BM25 + WordLlama, plain RRF | 0.718 | 0.94 | 0.993 | 0.815 | 0.83 | 1.44 | 0 |
| hybrid: BM25 + WordLlama, tuned (dense weight 0.25, k 10) | 0.805 | 0.958 | 0.995 | 0.873 | 0.83 | 1.38 | 0 |
| Gemini dense | 0.818 | 0.973 | 1.0 | 0.886 | 2.46 | 7.22 | 0 |
| hybrid: BM25 + Gemini, plain RRF | 0.853 | 0.978 | 1.0 | 0.911 | 2.92 | 7.57 | 0 |
| hybrid: BM25 + Gemini, tuned (dense weight 1.0, k 10) | 0.85 | 0.98 | 1.0 | 0.911 | 2.82 | 7.49 | 0 |

Gemini question vectors were embedded up front in batches, so the Gemini latencies are search time only; a live question adds one embedding request to Google.
