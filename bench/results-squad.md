# Retrieval on SQuAD 2.0 dev

1204 paragraphs, 366 of them restricted. Fusion weights tuned on 1879 questions from articles 1 to 18; every row below is measured on the 2248 answerable questions from articles 19 to 35, asked by a user who may read the gold paragraph but not the restricted ones.

| retriever | recall@1 | recall@5 | recall@20 | mrr@20 | p50 ms | p99 ms | leaks |
|---|---|---|---|---|---|---|---|
| BM25 | 0.824 | 0.949 | 0.978 | 0.877 | 0.04 | 0.12 | 0 |
| hashing (lexical stand in) | 0.569 | 0.749 | 0.859 | 0.651 | 1.07 | 1.62 | 0 |
| WordLlama dense | 0.585 | 0.83 | 0.949 | 0.691 | 0.83 | 1.39 | 0 |
| hybrid: BM25 + WordLlama, plain RRF | 0.724 | 0.929 | 0.987 | 0.813 | 1.07 | 1.83 | 0 |
| hybrid: BM25 + WordLlama, tuned (dense weight 0.25, k 10) | 0.81 | 0.953 | 0.984 | 0.873 | 1.04 | 1.75 | 0 |
