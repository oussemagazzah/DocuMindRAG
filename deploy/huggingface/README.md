---
title: DocuMindRAG
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Hybrid-search RAG with page citations over 3 public PDFs
---

# DocuMindRAG demo

Read-only demo of [DocuMindRAG](https://github.com/oussemagazzah/DocuMindRAG): ask questions about three
preloaded public PDFs and get answers with page citations.

- **Retrieval:** dense (all-MiniLM-L6-v2) + BM25, fused with Reciprocal Rank Fusion, then reranked by a
  cross-encoder (ms-marco-MiniLM-L-6-v2). Runs on this Space's CPU, with Qdrant in in-memory mode.
- **Generation:** Gemini (`gemini-3.5-flash`, temperature 0) on the free tier. To stay within its 20 requests
  per day, the demo answers at most 18 questions per day. After that it shows the retrieved passages without a
  generated answer.
- **Limits:** uploads and deletes are disabled, and each visitor can send 6 queries per minute.

Documents (see `evaluation/sources.md` for licences): BEIR (Thakur et al., 2021, CC BY-SA 4.0),
Ragas (Es et al., 2023, CC BY 4.0) and the OECD Economic Outlook Interim Report, September 2025 (CC BY 4.0).
