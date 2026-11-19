#!/usr/bin/env python
"""Publish the ARB dataset and model artifacts to the Hugging Face Hub.

    export HF_TOKEN=hf_...            # never commit this
    python scripts/push_to_hub.py --repo <user-or-org>/agent-recovery-bench --dry-run
    python scripts/push_to_hub.py --repo <user-or-org>/agent-recovery-bench

Nothing is uploaded unless a token is present *and* --dry-run is absent. This
script never prints the token. The dataset repo gets the JSONL/parquet files and
the dataset card; model artifacts go to ``<repo>-models``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

CARDS = {
    "dataset": "docs/dataset_card.md",
    "model": "docs/model_card.md",
}
# Destination layout inside the dataset repo. Episodes and steps live in separate
# config directories on purpose: `datasets` requires every split of one DatasetDict
# to share a single Arrow schema, and the step table is a different shape. Two
# configs keep `load_dataset(repo, "episodes")` and `load_dataset(repo, "steps")`
# both working. Parquet is the canonical artifact (typed, self-describing); the
# JSONL is the line-delimited mirror for people who just want to grep.
LAYOUT: tuple[tuple[str, str], ...] = (
    ("arb_dataset.parquet", "episodes/arb_dataset.parquet"),
    ("arb_dataset.jsonl", "episodes/arb_dataset.jsonl"),
    ("arb_steps.parquet", "steps/arb_steps.parquet"),
    ("arb_steps.jsonl", "steps/arb_steps.jsonl"),
    ("arb_mini.parquet", "mini/arb_mini.parquet"),
    ("arb_mini.jsonl", "mini/arb_mini.jsonl"),
    # The LLM-controller arm is its own config: the same episode schema, but a
    # different sample (240 harmful pairs, LLM-chosen operators) and a different
    # row count, so folding it into `episodes` would change that config's size.
    ("arb_llm.parquet", "llm/arb_llm.parquet"),
    ("arb_llm.jsonl", "llm/arb_llm.jsonl"),
    ("llm_stats.json", "llm/build_stats.json"),
    ("arb_episodes.parquet", "arb_episodes.parquet"),
    ("build_stats.json", "build_stats.json"),
)
MODEL_FILES = ("detector.json", "recovery_ranker.json")
#: Supporting evaluation evidence shipped next to the model artifacts.
MODEL_EXTRA = (("results/llm_agent.json", "llm_agent_results.json"),)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="e.g. yourname/agent-recovery-bench")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        # Fall back to a credential stored by `hf auth login` / the app's secrets.
        try:
            from huggingface_hub import get_token

            token = get_token()
        except Exception:  # huggingface_hub absent or no credential store
            token = None
    staged = [(src, dest) for src, dest in LAYOUT
              if os.path.exists(os.path.join(args.data_dir, src))]
    data = [os.path.join(args.data_dir, src) for src, _ in staged]
    models = [os.path.join(args.models_dir, f) for f in MODEL_FILES
              if os.path.exists(os.path.join(args.models_dir, f))]
    extras = [p for p, _ in MODEL_EXTRA if os.path.exists(p)]

    print(f"repo       : {args.repo}")
    for src, dest in staged:
        print(f"  {src:26s} -> {dest}")
    print(f"data files : {len(data)}")
    print(f"model files: {len(models)} -> {[os.path.basename(p) for p in models]}")
    print(f"extras     : {[os.path.basename(p) for p in extras]}")
    if not data:
        print("no data files; run scripts/build_dataset.py first")
        return 1

    if args.dry_run:
        print("\nDRY RUN: nothing uploaded. Remove --dry-run (with HF_TOKEN set) to publish.")
        return 0
    if not token:
        print("\nNo Hugging Face credential found: export HF_TOKEN or run `hf auth login`.")
        return 1

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for src, dest in staged:
            target = os.path.join(tmp, dest)
            os.makedirs(os.path.dirname(target) or tmp, exist_ok=True)
            shutil.copy(os.path.join(args.data_dir, src), target)
        shutil.copy(CARDS["dataset"], os.path.join(tmp, "README.md"))
        api.upload_folder(folder_path=tmp, repo_id=args.repo, repo_type="dataset",
                          commit_message="ARB dataset release")
    print(f"dataset -> https://huggingface.co/datasets/{args.repo}")

    if models:
        mrepo = f"{args.repo}-models"
        api.create_repo(mrepo, private=args.private, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            for p in models:
                shutil.copy(p, os.path.join(tmp, os.path.basename(p)))
            for src, name in MODEL_EXTRA:
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(tmp, name))
            shutil.copy(CARDS["model"], os.path.join(tmp, "README.md"))
            api.upload_folder(folder_path=tmp, repo_id=mrepo,
                              commit_message="ARB detector + recovery ranker")
        print(f"models  -> https://huggingface.co/{mrepo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
