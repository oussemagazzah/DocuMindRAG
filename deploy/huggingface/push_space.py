"""Create (if needed) and update the Hugging Face Docker Space for the public demo.

    hf auth login                                   # once, with a write token
    python deploy/huggingface/push_space.py --space <user>/DocuMindRAG
    python deploy/huggingface/push_space.py --stage-only build/space   # just assemble the folder

Only what the Space needs is uploaded: the app, the frontend, the corpus PDFs with
their licences, requirements.txt, and this folder's Dockerfile and README.md (Space card).
Set GEMINI_API_KEY as a secret in the Space settings; it is never read or uploaded here.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

FILES = ["requirements.txt", "run.py", "evaluation/sources.md"]
DIRS = ["app", "static", "evaluation/corpus"]


def stage(dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for d in DIRS:
        shutil.copytree(ROOT / d, dest / d, ignore=ignore)
    for f in FILES:
        (dest / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dest / f)
    shutil.copy2(HERE / "Dockerfile", dest / "Dockerfile")
    shutil.copy2(HERE / "README.md", dest / "README.md")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", help="Space id, e.g. oussemagazzah/DocuMindRAG")
    parser.add_argument("--stage-only", metavar="DIR", help="assemble the Space folder in DIR and stop")
    args = parser.parse_args()

    if args.stage_only:
        print(f"Staged Space files in {stage(Path(args.stage_only))}")
        return 0
    if not args.space:
        parser.error("--space is required unless --stage-only is given")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        folder = stage(Path(tmp) / "space")
        api.upload_folder(
            repo_id=args.space,
            repo_type="space",
            folder_path=folder,
            commit_message="Update DocuMindRAG demo",
            delete_patterns=["app/**", "static/**", "evaluation/**"],
        )
    print(f"Pushed. Build logs: https://huggingface.co/spaces/{args.space}?logs=build")
    print("Remember: Space settings -> Variables and secrets -> add secret GEMINI_API_KEY.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
