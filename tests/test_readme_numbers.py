"""The README must quote the committed results, not hand-typed numbers."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
ABLATION = (ROOT / "evaluation" / "results" / "ablation.md").read_text(encoding="utf-8")
RESULTS = json.loads((ROOT / "evaluation" / "results" / "results.json").read_text(encoding="utf-8"))


def _table_rows(markdown: str, header_start: str) -> list[str]:
    lines = markdown.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header_start))
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        rows.append(line)
    return rows


def test_ablation_tables_in_readme_match_results():
    for header in ("| Configuration | Recall@1", "| Configuration | fact_retrieval"):
        for row in _table_rows(ABLATION, header):
            assert row in README, f"README is out of date with ablation.md: {row}"


def test_headline_percentages_match_results():
    configs = RESULTS["retrieval"]["configs"]
    expected = {
        f"{configs['hybrid_rerank']['overall']['recall@5'] * 100:.2f}%",
        f"{configs['dense']['overall']['recall@5'] * 100:.2f}%",
        f"{configs['bm25']['overall']['recall@1'] * 100:.2f}%",
    }
    headline = README.split("## Retrieval ablation")[0]
    for value in expected:
        assert value in headline, f"headline does not quote {value} from results.json"


def test_end_to_end_numbers_only_after_a_real_run():
    # Until results.json has an end_to_end section, the README must say TBD rather than show numbers.
    section = README.split("## End-to-end answer quality")[1].split("### What")[0]
    if "end_to_end" not in RESULTS:
        assert "TBD" in section
        assert not re.search(r"\|\s*\d+(\.\d+)?%", section)


def test_unverified_claims_removed():
    assert "92%" not in README
    assert "1.8s" not in README
    assert "github.com/username" not in README
