"""Regenerate the committed evaluation results.

    python -m evaluation.run_eval --all          # retrieval ablation + end-to-end
    python -m evaluation.run_eval --retrieval    # retrieval ablation only (no LLM, deterministic)
    python -m evaluation.run_eval --e2e --judge  # end-to-end answers + LLM-as-judge second opinion
    python -m evaluation.run_eval --e2e --limit 3  # smoke test, writes nothing

Writes evaluation/results/results.json, ablation.md, end_to_end.md and manual_review.md.
Sections not re-run are kept from the existing results.json.
"""

import os

# Must be set before any app module reads settings: the evaluation always runs
# against a fresh in-process Qdrant, never a server with user documents in it.
os.environ["QDRANT_LOCATION"] = ":memory:"
os.environ["COLLECTION_NAME"] = "documind_eval"
# The reported numbers use one fixed LLM configuration, regardless of the local .env.
GENERATOR_MODEL = "gemini-3.5-flash"
# gemini-3.8-flash timed out (504) on 2026-10-19; Gemma 4 ignored the JSON format.
JUDGE_MODEL = "gemini-3.1-flash-lite"
os.environ["GEMINI_MODEL"] = GENERATOR_MODEL
os.environ["GEMINI_TEMPERATURE"] = "0.0"
os.environ["GEMINI_MAX_OUTPUT_TOKENS"] = "1024"

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from evaluation.check_benchmark import ROOT, check, load_benchmark  # noqa: E402

RESULTS_DIR = ROOT / "evaluation" / "results"
SEED = 0


def _seed_everything():
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _ram_gb() -> float | str:
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return round(status.ullTotalPhys / 1024**3, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (OSError, ValueError, AttributeError):
        return "unknown"


def _hf_revision(repo_id: str) -> str:
    """Commit hash of the locally cached Hugging Face model snapshot."""
    from huggingface_hub.constants import HF_HUB_CACHE

    ref = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else "unknown"


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "app", "evaluation", ":!evaluation/results"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment(benchmark: dict, index_stats: dict, repeats: int) -> dict:
    from importlib.metadata import version

    import qdrant_client
    import sentence_transformers
    import torch

    from app.config import get_settings
    from app.ingestion.embedder import get_embedding_model

    s = get_settings()
    embedding_repo = s.embedding_model if "/" in s.embedding_model else f"sentence-transformers/{s.embedding_model}"
    questions = benchmark["questions"]
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "git_commit": _git_commit(),
        "seed": SEED,
        "embedding_model": embedding_repo,
        "embedding_revision": _hf_revision(embedding_repo),
        "reranker_model": s.reranker_model,
        "reranker_revision": _hf_revision(s.reranker_model),
        "chunk_size_tokens": s.chunk_size_tokens,
        "chunk_overlap_tokens": s.chunk_overlap_tokens,
        "dense_top_k": s.dense_top_k,
        "sparse_top_k": s.sparse_top_k,
        "rrf_k": s.rrf_k,
        "rerank_top_k": s.rerank_top_k,
        "bm25_k1": s.bm25_k1,
        "bm25_b": s.bm25_b,
        "bm25_avg_doc_len": s.bm25_avg_doc_len,
        "index": index_stats,
        "corpus_pages": sum(d["pages"] for d in index_stats.values()),
        "corpus_chunks": sum(d["chunks"] for d in index_stats.values()),
        "total_questions": len(questions),
        "answerable_questions": sum(1 for q in questions if q["category"] != "unanswerable"),
        "latency_repeats": repeats,
        "cpu": _cpu_name(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "device": str(get_embedding_model().device),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "sentence_transformers": sentence_transformers.__version__,
        "qdrant_client": version("qdrant-client") if qdrant_client else "unknown",
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="run retrieval ablation and end-to-end evaluation")
    parser.add_argument("--retrieval", action="store_true", help="run the retrieval ablation")
    parser.add_argument("--e2e", action="store_true", help="run the end-to-end answer evaluation (needs GEMINI_API_KEY)")
    parser.add_argument("--judge", action="store_true", help=f"add LLM-as-judge faithfulness ({JUDGE_MODEL}); implied by --all")
    parser.add_argument("--limit", type=int, help="e2e smoke test on the first N questions; writes no files")
    parser.add_argument("--min-interval", type=float, default=6.5, help="seconds between LLM calls (free-tier rate limit)")
    parser.add_argument("--no-cache", action="store_true", help="ignore cached LLM responses in .eval_cache/")
    parser.add_argument("--repeats", type=int, default=3, help="latency passes over all questions")
    args = parser.parse_args(argv)
    if not (args.all or args.retrieval or args.e2e):
        parser.error("choose --all, --retrieval or --e2e")

    benchmark = load_benchmark()
    errors = check(benchmark)
    if errors:
        print("Benchmark check failed:\n  " + "\n  ".join(errors))
        return 1

    if args.limit:
        if args.all or args.retrieval:
            parser.error("--limit only applies to --e2e")
        benchmark = benchmark | {"questions": benchmark["questions"][: args.limit]}

    _seed_everything()
    from evaluation.retrieval_eval import ablation_markdown, build_index, run_ablation

    print("Indexing corpus...", flush=True)
    index_stats = build_index(benchmark)
    env = environment(benchmark, index_stats, args.repeats)

    results_path = RESULTS_DIR / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    results["benchmark"] = {"version": benchmark["metadata"]["version"], "total_questions": len(benchmark["questions"])}

    if args.all or args.retrieval:
        print("Running retrieval ablation...", flush=True)
        results["retrieval"] = {"environment": env, "configs": run_ablation(benchmark, args.repeats)}
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "ablation.md").write_text(ablation_markdown(results["retrieval"]["configs"], env), encoding="utf-8")

    if args.all or args.e2e:
        from evaluation.e2e_eval import e2e_markdown, manual_review_markdown, run_e2e

        print("Running end-to-end evaluation...", flush=True)
        e2e = run_e2e(benchmark, env, JUDGE_MODEL if (args.all or args.judge) else None,
                      min_interval_s=args.min_interval, use_cache=not args.no_cache)
        if args.limit:
            print(json.dumps(e2e["overall"], indent=2))
            return 0
        results["end_to_end"] = e2e
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "end_to_end.md").write_text(e2e_markdown(e2e), encoding="utf-8")
        (RESULTS_DIR / "manual_review.md").write_text(manual_review_markdown(e2e, benchmark), encoding="utf-8")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {results_path.relative_to(ROOT)}")
    if args.all or args.retrieval:
        print((RESULTS_DIR / "ablation.md").read_text(encoding="utf-8"))
    if args.all or args.e2e:
        print((RESULTS_DIR / "end_to_end.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
