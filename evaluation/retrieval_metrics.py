"""Page-level retrieval metrics against benchmark evidence groups.

Retrieval returns ranked chunks. We turn them into a ranked list of unique
(doc_id, page) pairs, keeping each page's best rank, because a citation points
to a page, not a chunk.

A question's gold evidence is a list of groups; any page in a group supplies
that piece of evidence (a fact restated on two pages is one group with two
pages, a two-hop question has two groups). With G groups:

- Recall@k: fraction of groups with at least one page in the top-k pages.
- MRR@10: 1 / rank of the first page that belongs to any group (0 if none in top 10).
- nDCG@5: binary gain for a page that satisfies a group not already satisfied
  higher up (restatements are not double-counted); ideal DCG puts min(G, 5)
  satisfying pages at ranks 1..min(G, 5).
"""

import math
import random
from statistics import mean

Page = tuple[str, int]


def ranked_pages(chunks) -> list[Page]:
    """Unique (doc_id, page_number) in rank order from ranked SearchResults."""
    seen: set[Page] = set()
    pages: list[Page] = []
    for c in chunks:
        key = (c.doc_id, c.page_number)
        if key not in seen:
            seen.add(key)
            pages.append(key)
    return pages


def _groups(doc_id: str, evidence_groups: list[list[int]]) -> list[set[Page]]:
    return [{(doc_id, p) for p in g} for g in evidence_groups]


def recall_at_k(pages: list[Page], groups: list[set[Page]], k: int) -> float:
    top = set(pages[:k])
    return sum(1 for g in groups if g & top) / len(groups)


def mrr_at_k(pages: list[Page], groups: list[set[Page]], k: int = 10) -> float:
    relevant = set().union(*groups)
    for rank, page in enumerate(pages[:k], start=1):
        if page in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(pages: list[Page], groups: list[set[Page]], k: int = 5) -> float:
    satisfied = [False] * len(groups)
    dcg = 0.0
    for rank, page in enumerate(pages[:k], start=1):
        hit = False
        for i, g in enumerate(groups):
            if not satisfied[i] and page in g:
                satisfied[i] = True
                hit = True
        if hit:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(groups), k) + 1))
    return dcg / ideal


def question_metrics(chunks, question: dict) -> dict:
    pages = ranked_pages(chunks)
    groups = _groups(question["doc_id"], question["evidence_groups"])
    return {
        "recall@1": recall_at_k(pages, groups, 1),
        "recall@3": recall_at_k(pages, groups, 3),
        "recall@5": recall_at_k(pages, groups, 5),
        "mrr@10": mrr_at_k(pages, groups, 10),
        "ndcg@5": ndcg_at_k(pages, groups, 5),
        "top_pages": [f"{d}:p{p}" for d, p in pages[:5]],
    }


METRIC_KEYS = ["recall@1", "recall@3", "recall@5", "mrr@10", "ndcg@5"]


def bootstrap_ci(values: list[float], n_resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean (seeded, so results are reproducible)."""
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(n_resamples))
    return round(means[int(0.025 * n_resamples)], 4), round(means[int(0.975 * n_resamples) - 1], 4)


def aggregate(per_question: list[dict], with_ci: bool = False) -> dict:
    out = {k: round(mean(q[k] for q in per_question), 4) for k in METRIC_KEYS} | {"n": len(per_question)}
    if with_ci:
        out["ci95"] = {k: bootstrap_ci([q[k] for q in per_question]) for k in ("recall@5", "mrr@10")}
    return out
