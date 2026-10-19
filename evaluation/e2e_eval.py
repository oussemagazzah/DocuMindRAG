"""End-to-end evaluation of the production pipeline: hybrid search -> cross-encoder
rerank (top 5) -> Gemini answer -> heuristic groundedness, plus an optional
LLM-as-judge faithfulness score from a different Gemini model.

Run through `python -m evaluation.run_eval --e2e [--judge]` (or `--all`), which pins
the generator model and temperature before the app settings are loaded.
"""

import random
import time

from app.config import get_settings
from app.retrieval.answer import SYSTEM_PROMPT, _format_context, build_user_message
from app.retrieval.groundedness import compute_groundedness, extract_cited_pages
from app.retrieval.reranker import rerank
from app.retrieval.search import hybrid_search
from evaluation.answer_metrics import aggregate, citation_precision, heuristic_vs_judge, is_refusal
from evaluation.facts import fact_in, fact_recall
from evaluation.judge import judge_answer
from evaluation.llm_client import GeminiClient
from evaluation.retrieval_metrics import question_metrics

settings = get_settings()
CATEGORIES = ("fact_retrieval", "multi_hop", "comparison", "unanswerable")


def evaluate_question(client: GeminiClient, q: dict, judge_model: str | None) -> dict:
    row = {"id": q["id"], "category": q["category"], "doc_id": q["doc_id"], "question": q["question"]}

    start = time.perf_counter()
    context = rerank(q["question"], hybrid_search(q["question"]), top_k=settings.rerank_top_k)
    retrieval_ms = (time.perf_counter() - start) * 1000
    row["context_pages"] = [f"{c.doc_id}:p{c.page_number}" for c in context]
    if q["category"] != "unanswerable":
        row["context_recall"] = question_metrics(context, q)["recall@5"]

    try:
        llm = client.generate(settings.gemini_model, SYSTEM_PROMPT, build_user_message(q["question"], context),
                              temperature=settings.gemini_temperature,
                              max_output_tokens=settings.gemini_max_output_tokens)
    except RuntimeError as e:
        return row | {"error": str(e)}

    answer = llm.text
    row |= {
        "answer": answer,
        "finish_reason": llm.finish_reason,
        "generation_ms": llm.latency_ms,
        "latency_ms": round(retrieval_ms + llm.latency_ms, 1) if llm.latency_ms is not None else None,
        "cached": llm.cached,
        "tokens": {"prompt": llm.prompt_tokens, "output": llm.output_tokens, "total": llm.total_tokens},
    }
    if not answer.strip():
        return row | {"error": f"empty answer (finish_reason={llm.finish_reason})"}

    cited = extract_cited_pages(answer)
    g = compute_groundedness(answer, context)
    row |= {
        "cited_pages": sorted(set(cited)),
        "refused": is_refusal(answer),
        "groundedness": g.score,
        "groundedness_verdict": g.verdict,
    }
    if q["category"] == "unanswerable":
        row["correct"] = row["refused"]
    else:
        facts = q["expected_answer_contains"]
        row |= {
            "fact_recall": round(fact_recall(facts, answer), 4),
            "missing_facts": [f[0] if isinstance(f, list) else f for f in facts if not fact_in(f, answer)],
            "citation_precision": citation_precision(cited, q["expected_pages"]),
        }
        row["correct"] = row["fact_recall"] == 1.0

    if judge_model:
        row |= judge_answer(client, judge_model, q["question"], _format_context(context), answer)
    return row


def review_sample(rows: list[dict], per_category: int = 5, seed: int = 0) -> list[str]:
    """Stratified, seeded sample of question ids for the manual agreement check."""
    rng = random.Random(seed)
    ids = []
    for cat in CATEGORIES:
        pool = sorted(r["id"] for r in rows if r["category"] == cat and not r.get("error"))
        ids += sorted(rng.sample(pool, min(per_category, len(pool))))
    return ids


def run_e2e(benchmark: dict, env: dict, judge_model: str | None, min_interval_s: float = 6.5,
            use_cache: bool = True) -> dict:
    client = GeminiClient(min_interval_s=min_interval_s, use_cache=use_cache)
    rows = []
    for i, q in enumerate(benchmark["questions"], 1):
        row = evaluate_question(client, q, judge_model)
        status = row.get("error") or f"correct={row.get('correct')} refused={row.get('refused')}"
        print(f"[{i:2}/{len(benchmark['questions'])}] {q['id']:8} {status}", flush=True)
        rows.append(row)

    return {
        "config": {
            "pipeline": f"hybrid (dense+BM25, RRF k={settings.rrf_k}, {settings.dense_top_k} candidates) -> "
                        f"cross-encoder rerank -> top {settings.rerank_top_k} chunks -> LLM",
            "generator_model": settings.gemini_model,
            "temperature": settings.gemini_temperature,
            "max_output_tokens": settings.gemini_max_output_tokens,
            "judge_model": judge_model,
            "judge_temperature": 0.0 if judge_model else None,
            "system_prompt": SYSTEM_PROMPT,
            "search_scope": "whole corpus, no document filter",
        },
        "environment": {k: env[k] for k in ("generated_at", "git_commit", "cpu", "device", "python")},
        "overall": aggregate(rows),
        "by_category": {cat: aggregate([r for r in rows if r["category"] == cat]) for cat in CATEGORIES},
        "heuristic_vs_judge": heuristic_vs_judge(rows) if judge_model else None,
        "manual_review_ids": review_sample(rows),
        "per_question": rows,
    }


def _fmt(v, pct: bool = True) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%" if pct else f"{v:.3f}"


def e2e_markdown(e2e: dict) -> str:
    cfg, overall = e2e["config"], e2e["overall"]
    rows = [("Overall", overall)] + [(c, e2e["by_category"][c]) for c in CATEGORIES]
    lines = [
        "# End-to-end answer quality",
        "",
        f"Pipeline: {cfg['pipeline']}. Generator `{cfg['generator_model']}`, temperature {cfg['temperature']}, "
        f"max {cfg['max_output_tokens']} output tokens. Search scope: {cfg['search_scope']}.",
        "",
        "| Category | n | Answer correctness | Fact recall | Refusal rate (unanswerable) | False refusals (answerable) "
        "| Citation rate | Citation precision | Heuristic groundedness | Judge faithfulness | Median latency |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, a in rows:
        latency = f"{a['latency_ms']['median']:.0f} ms" if a["latency_ms"]["median"] is not None else "n/a"
        lines.append(
            f"| {name} | {a['n']} | {_fmt(a.get('answer_correctness'))} | {_fmt(a.get('fact_recall'))} | "
            f"{_fmt(a.get('refusal_rate'))} | {_fmt(a.get('false_refusal_rate'))} | {_fmt(a.get('citation_rate'))} | "
            f"{_fmt(a.get('citation_precision'))} | {_fmt(a.get('heuristic_groundedness'), pct=False)} | "
            f"{_fmt(a.get('judge_faithfulness'), pct=False)} | {latency} |"
        )
    lines += [
        "",
        f"**Overall correct (answerable: all key facts present; unanswerable: refused): {_fmt(overall.get('overall_correct'))}** "
        f"of {overall['n']} questions. Errors: {overall['errors']}. Answers cut off at the token limit: {overall['truncated']}. "
        f"End-to-end p95 latency: {overall['latency_ms']['p95'] or 0:.0f} ms (network-bound; Gemini API from a home connection).",
        "",
        "## Definitions",
        "",
        "- **Answer correctness**: share of answerable questions whose answer contains *every* expected key fact "
        "(any listed spelling, word-boundary match). **Fact recall** gives partial credit: share of key facts present.",
        "- **Refusal rate**: share of unanswerable questions where the answer says the documents do not contain the "
        "answer (regex over common phrasings; see `evaluation/answer_metrics.py`). **False refusals**: answerable "
        "questions that were refused and contain none of the key facts.",
        "- **Citation rate / precision**: share of answerable answers with at least one `[p.X]` citation; of the cited "
        "page numbers, the share that are gold pages. Citations carry no document name, so pages are matched within "
        "the gold document.",
        "- **Heuristic groundedness** (the score the app shows): `0.5 x citation validity + 0.3 x context coverage + "
        "0.2 x hedging`. Citation validity = cited pages that were in the retrieved context (0.5 if nothing is cited); "
        "context coverage = share of retrieved chunks whose first 10 content words (4+ letters) appear in the answer; "
        "hedging = the answer contains an 'insufficient information' phrase. It checks **citation hygiene and word "
        "overlap, not whether claims are true**: no LLM, no entailment.",
        "- **Judge faithfulness** (second opinion): a different model splits the answer into claims and checks each "
        "against the retrieved context; score = supported / total claims (Ragas-style). Refusals with no factual claims "
        "get no score. The judge is also a Gemini model, so it is not independent of the generator.",
    ]
    hj = e2e.get("heuristic_vs_judge")
    if hj and hj.get("n"):
        lines += [
            "",
            "## Heuristic vs LLM judge",
            "",
            f"On {hj['n']} answers scored by both: Spearman {hj['spearman']}, Pearson {hj['pearson']}. "
            f"Binary agreement {_fmt(hj['binary_agreement'])} ({hj['definition']}).",
            "",
            "| | Judge: all claims supported | Judge: some claim unsupported |",
            "|---|---|---|",
            f"| Heuristic >= 0.7 | {hj['both_pass']} | {hj['heuristic_pass_judge_fail']} |",
            f"| Heuristic < 0.7 | {hj['heuristic_fail_judge_pass']} | {hj['both_fail']} |",
        ]
    lines += ["", f"Generated {e2e['environment']['generated_at']} at commit `{e2e['environment']['git_commit']}`.", ""]
    return "\n".join(lines)


def manual_review_markdown(e2e: dict, benchmark: dict) -> str:
    by_id = {q["id"]: q for q in benchmark["questions"]}
    rows = {r["id"]: r for r in e2e["per_question"]}
    lines = [
        "# Manual review: heuristic groundedness vs LLM judge",
        "",
        f"{len(e2e['manual_review_ids'])} answers, 5 per category, sampled with seed 0. For each one, read the answer "
        "against the retrieved pages (open the PDFs in `evaluation/corpus/`) and fill in the two `Human` lines. "
        "Then run `python -m evaluation.manual_agreement` to compute agreement with both automatic scores.",
        "",
        "- **Human faithfulness**: `1` = every factual claim is supported by the retrieved pages, `0.5` = some are, "
        "`0` = none / contradicted. Write `-` for a pure refusal with no factual claims.",
        "- **Human correct**: `y` / `n`: does the answer correctly answer the question (or correctly refuse)?",
        "",
    ]
    for qid in e2e["manual_review_ids"]:
        q, r = by_id[qid], rows[qid]
        facts = ", ".join(f[0] if isinstance(f, list) else f for f in q["expected_answer_contains"]) or "(unanswerable)"
        lines += [
            f"## {qid} ({q['category']})",
            "",
            f"**Question:** {q['question']}",
            "",
            f"**Gold:** pages {q['expected_pages'] or '-'} of `{q['doc_id']}`; key facts: {facts}",
            "",
            f"**Retrieved context:** {', '.join(r.get('context_pages', []))}",
            "",
            "**Answer:**",
            "",
            "> " + (r.get("answer") or r.get("error", "")).strip().replace("\n", "\n> "),
            "",
            f"Automatic: heuristic groundedness **{r.get('groundedness')}** ({r.get('groundedness_verdict')}); "
            f"judge faithfulness **{r.get('judge_faithfulness')}** "
            f"({r.get('judge_supported', '-')}/{r.get('judge_claims', '-')} claims); refused={r.get('refused')}; "
            f"key facts present={r.get('fact_recall', '-')}",
        ]
        if r.get("judge_unsupported"):
            lines += ["", "Judge's unsupported claims: " + "; ".join(f"\"{c}\"" for c in r["judge_unsupported"])]
        lines += ["", "Human faithfulness: ", "", "Human correct: ", "", "---", ""]
    return "\n".join(lines)
