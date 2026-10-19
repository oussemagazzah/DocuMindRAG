"""Agreement between the human labels in results/manual_review.md and the automatic scores.

Usage: python -m evaluation.manual_agreement
"""

import json
import re
import sys

from evaluation.check_benchmark import ROOT

REVIEW = ROOT / "evaluation" / "results" / "manual_review.md"
RESULTS = ROOT / "evaluation" / "results" / "results.json"


def parse_labels(text: str) -> dict[str, dict]:
    labels = {}
    for block in re.split(r"^## ", text, flags=re.MULTILINE)[1:]:
        qid = block.split()[0]
        faith = re.search(r"^Human faithfulness:\s*(\S*)", block, re.MULTILINE)
        correct = re.search(r"^Human correct:\s*(\S*)", block, re.MULTILINE)
        f = faith.group(1) if faith else ""
        c = correct.group(1).lower() if correct else ""
        labels[qid] = {
            "faithfulness": float(f) if re.fullmatch(r"[01](?:\.5)?|0?\.5", f) else None,
            "correct": {"y": True, "n": False}.get(c[:1]),
        }
    return labels


def agreement(labels: dict[str, dict], rows: dict[str, dict]) -> dict:
    out = {"labelled_faithfulness": 0, "labelled_correct": 0}
    h_agree = j_agree = auto_correct_agree = 0
    for qid, lab in labels.items():
        r = rows[qid]
        if lab["faithfulness"] is not None:
            out["labelled_faithfulness"] += 1
            human_ok = lab["faithfulness"] == 1.0
            h_agree += ((r.get("groundedness") or 0) >= 0.7) == human_ok
            if r.get("judge_faithfulness") is not None:
                j_agree += (r["judge_faithfulness"] == 1.0) == human_ok
        if lab["correct"] is not None:
            out["labelled_correct"] += 1
            auto_correct_agree += bool(r.get("correct")) == lab["correct"]
    n_f, n_c = out["labelled_faithfulness"], out["labelled_correct"]
    out["heuristic_agrees_with_human"] = round(h_agree / n_f, 4) if n_f else None
    out["judge_agrees_with_human"] = round(j_agree / n_f, 4) if n_f else None
    out["auto_correctness_agrees_with_human"] = round(auto_correct_agree / n_c, 4) if n_c else None
    out["definition"] = ("faithful = human label 1; heuristic says faithful when score >= 0.7; "
                         "judge says faithful when every claim is supported")
    return out


def main() -> int:
    labels = parse_labels(REVIEW.read_text(encoding="utf-8"))
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in results["end_to_end"]["per_question"]}
    stats = agreement(labels, rows)
    print(json.dumps(stats, indent=2))
    if not stats["labelled_faithfulness"] and not stats["labelled_correct"]:
        print("No human labels filled in yet.")
        return 1
    results["end_to_end"]["manual_agreement"] = stats
    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved to {RESULTS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
