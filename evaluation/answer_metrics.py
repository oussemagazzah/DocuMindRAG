"""Answer-level metrics for the end-to-end evaluation (no LLM calls here)."""

import math
import re
from statistics import mean, median

import numpy as np

# The system prompt asks for "I don't have enough information in the provided
# documents..."; models paraphrase it, so match the common forms.
_REFUSAL_PATTERNS = [
    r"(?:do not|don't|does not|doesn't) (?:have|contain|provide|include|mention|specify|state|say|give|report)"
    r" (?:enough |sufficient |any |a |an |the )?(?:specific |explicit )?"
    r"(?:information|data|details?|figures?|mention|projections?|forecasts?|estimates?|numbers?|values?|scores?)",
    r"not enough information",
    r"insufficient information",
    r"no information (?:about|on|regarding)",
    r"(?:is|are) not (?:mentioned|provided|stated|specified|included|available|given|reported)",
    r"(?:cannot|can't|can not|unable to) (?:be )?(?:answer|determine|find|answered|determined|found)",
    r"(?:does|do) not (?:report|discuss|cover|address)",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL_RE.search(answer.replace("’", "'")))


def citation_precision(cited_pages: list[int], expected_pages: list[int]) -> float | None:
    """Fraction of distinct cited page numbers that are gold pages; None when nothing is cited.

    Citations carry a page number but no document name, so a cited page is matched
    against the gold document's pages.
    """
    cited = set(cited_pages)
    if not cited:
        return None
    return len(cited & set(expected_pages)) / len(cited)


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(mean(vals), 4) if vals else None


def _rate(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 4) if flags else None


def aggregate(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["category"] != "unanswerable"]
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]
    ok = [r for r in rows if not r.get("error")]
    latencies = [r["latency_ms"] for r in ok if r.get("latency_ms") is not None]
    out = {
        "n": len(rows),
        "errors": sum(1 for r in rows if r.get("error")),
        "truncated": sum(1 for r in rows if r.get("finish_reason") == "MAX_TOKENS"),
        "heuristic_groundedness": _mean([r.get("groundedness") for r in ok]),
        "judge_faithfulness": _mean([r.get("judge_faithfulness") for r in ok]),
        "latency_ms": {
            "median": round(median(latencies), 1) if latencies else None,
            "p95": round(float(np.percentile(latencies, 95)), 1) if latencies else None,
        },
    }
    if answerable:
        out |= {
            "answerable_n": len(answerable),
            "answer_correctness": _rate([r.get("correct", False) for r in answerable]),
            "fact_recall": _mean([r.get("fact_recall", 0.0) for r in answerable]),
            "false_refusal_rate": _rate([r.get("refused", False) and r.get("fact_recall", 0.0) == 0 for r in answerable]),
            "citation_rate": _rate([bool(r.get("cited_pages")) for r in answerable]),
            "citation_precision": _mean([r.get("citation_precision") for r in answerable]),
        }
    if unanswerable:
        out |= {
            "unanswerable_n": len(unanswerable),
            "refusal_rate": _rate([r.get("refused", False) for r in unanswerable]),
        }
    if answerable and unanswerable:
        correct = [r.get("correct", False) for r in answerable] + [r.get("refused", False) for r in unanswerable]
        out["overall_correct"] = _rate(correct)
    return out


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    mx, my = mean(x), mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / (sx * sy)


def spearman(x: list[float], y: list[float]) -> float | None:
    r = _pearson(_ranks(x), _ranks(y))
    return round(r, 4) if r is not None else None


def heuristic_vs_judge(rows: list[dict]) -> dict:
    """How the heuristic groundedness score relates to the judge's faithfulness score."""
    pairs = [(r["groundedness"], r["judge_faithfulness"]) for r in rows
             if r.get("groundedness") is not None and r.get("judge_faithfulness") is not None]
    if not pairs:
        return {"n": 0}
    h, j = [p[0] for p in pairs], [p[1] for p in pairs]
    h_pass = [v >= 0.7 for v in h]  # heuristic verdict "grounded"
    j_pass = [v == 1.0 for v in j]  # judge: every claim supported
    return {
        "n": len(pairs),
        "spearman": spearman(h, j),
        "pearson": round(p, 4) if (p := _pearson(h, j)) is not None else None,
        "binary_agreement": round(sum(a == b for a, b in zip(h_pass, j_pass, strict=True)) / len(pairs), 4),
        "both_pass": sum(a and b for a, b in zip(h_pass, j_pass, strict=True)),
        "heuristic_pass_judge_fail": sum(a and not b for a, b in zip(h_pass, j_pass, strict=True)),
        "heuristic_fail_judge_pass": sum(b and not a for a, b in zip(h_pass, j_pass, strict=True)),
        "both_fail": sum(not a and not b for a, b in zip(h_pass, j_pass, strict=True)),
        "definition": "pass = heuristic score >= 0.7 (verdict 'grounded') vs judge faithfulness == 1.0 (all claims supported)",
    }
