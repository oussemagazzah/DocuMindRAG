# Benchmark corpus: sources and licences

The benchmark (`evaluation/benchmark.json`) is built on three public PDFs, 56 pages in total.
The files in `evaluation/corpus/` are unmodified copies of the originals. You can check them against the SHA-256 hashes below.
Each document keeps its own licence. The rest of the repository's licence does not apply to them.

| doc_id | Document | Pages | Licence | Source |
|---|---|---|---|---|
| `beir_2021` | Thakur, Reimers, Rücklé, Srivastava, Gurevych. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS 2021 Datasets and Benchmarks. arXiv:2104.08663v4 | 24 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | https://arxiv.org/abs/2104.08663v4 |
| `ragas_2023` | Es, James, Espinosa-Anke, Schockaert. *Ragas: Automated Evaluation of Retrieval Augmented Generation.* arXiv:2309.15217v2 | 8 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | https://arxiv.org/abs/2309.15217v2 |
| `oecd_interim_2025_09` | OECD (2025), *OECD Economic Outlook, Interim Report September 2025: Finding the Right Balance in Uncertain Times*, OECD Publishing, Paris, https://doi.org/10.1787/67b10c01-en | 24 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (stated on the report's copyright page) | https://www.oecd.org/content/dam/oecd/en/publications/reports/2025/09/oecd-economic-outlook-interim-report-september-2025_ae3d418b/67b10c01-en.pdf |

Licences were checked on 2026-10-19: on the arXiv abstract pages ("view license") and on page 4 of the OECD PDF.
The BEIR PDF is redistributed under CC BY-SA 4.0, and any adaptation of it must use the same licence.
The OECD notice requires citing the work, and says the licence does not cover third-party material in it.

| File | Bytes | SHA-256 |
|---|---|---|
| `corpus/beir_2021.pdf` | 2,094,824 | `1925755aeb627123d1a63c66f02e5aa2432f7e92751a02bea7c8c3b6bbb6ec33` |
| `corpus/ragas_2023.pdf` | 231,984 | `6c24762a3f277fd120ed1adf34b01420ecf92f18ff6d1b5435fa0a09cb804ecb` |
| `corpus/oecd_interim_2025_09.pdf` | 2,152,981 | `f3d64751363d12e1dd140b69ebb2c81ac7697f27ec6bf55194ce9f4484fb96df` |

## Why these documents

- **Different domains and styles.** An economics report full of figures, and two machine-learning papers that share a lot of vocabulary (retrieval, evaluation, relevance, annotators). Shared vocabulary makes distractor passages realistic.
- **Tables.** OECD Tables 1–3 (GDP and inflation by country), BEIR Tables 2, 3, 4 and 9 (nDCG@10, latency, Hole@10, Recall@100) and Ragas Table 1. PyMuPDF extracts tables as one number per line, with country or model names separated from their values. Questions that can only be answered from a table are marked `hard`.
- **Content that spans pages.** Multi-hop questions need evidence from two different pages.

## How the questions were written and checked

1. Each PDF's text was extracted with the same parser the application uses (`app.ingestion.pdf_parser`), page by page, and read in full.
2. Each question was written from a specific passage. `expected_pages` uses the **1-based PDF page index** (the OECD report's printed page numbers are 2 lower than the index).
3. `evidence_groups` records which pages supply each piece of evidence. A fact restated on several pages, such as an OECD summary number that also appears in a table, forms one group with several pages. A multi-hop question has one group per hop.
4. `expected_answer_contains` lists key facts. Each fact is a list of acceptable spellings (for example `["6100ms", "6,100 ms", "6.1 s"]`).
5. `python -m evaluation.check_benchmark` (also run in CI by `tests/test_benchmark.py`) verifies that:
   - the question counts match the metadata;
   - every fact appears, with word boundaries, on at least one of its gold pages;
   - every evidence group contains at least one fact;
   - every `absent_terms` entry of an unanswerable question appears **nowhere** in the corpus.

   On its first run the checker rejected one unanswerable question: "sports" appears in BEIR's Table 8. That question was replaced.
6. Unanswerable questions are deliberately close to the documents, for example a 2027 projection (the report stops at 2026), a model BEIR does not evaluate, or a metric Ragas does not report.

### Limitations

- 50 questions, written by one person, over 3 documents. Differences of a few points between systems are within noise.
- The checker proves a fact is *on* the gold page. It cannot prove the fact is the right answer to the question; that part rests on the author's reading.
- Answer correctness uses string matching of key facts. A correct answer phrased differently (for example "six point one seconds") counts as wrong.
- English only.
