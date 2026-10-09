"""Push Bharat-Embed family to HF under eulogik org. Private repos.

Login: uses cached `hf auth login` (GautamKishore, admin of eulogik). No token in code.
Run: python scripts/push_hf.py --dry-run
Real: python scripts/push_hf.py (uploads what exists, skips missing weights)

Repos (all private):
- eulogik/bharat-embed-270m-gemma2 (weights if trained, else card plus eval)
- eulogik/bharat-embed-270m-gemma2-GGUF (quants if present)
- eulogik/bharat-embed-triplets-40k (dataset card plus SHA, rows live on request)
- eulogik/bharat-legal-embed-270m (adapter if trained, else card)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPOS = {
    "eulogik/bharat-embed-270m-gemma2": "model",
    "eulogik/bharat-embed-270m-gemma2-GGUF": "model",
    "eulogik/bharat-embed-triplets-40k": "dataset",
    "eulogik/bharat-legal-embed-270m": "model",
}

CARD_META = {
    "library_name": "sentence-transformers",
    "pipeline_tag": "sentence-similarity",
    "license": "apache-2.0",
    "base_model": "google/embeddinggemma-2",
}

KIO = Path("/Volumes/KIOXIA 1TB/bharat-embed")
ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"repos: {list(REPOS)}")
    print(f"meta: {CARD_META} private=True")
    if args.dry_run:
        print("dry run ok. would create private repos, push cards, eval json, weights if present.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    me = api.whoami()
    print(f"hf user: {me['name']}")
    assert any(o["name"] == "eulogik" for o in me.get("orgs", [])), "eulogik org missing"

    for repo_id, kind in REPOS.items():
        api.create_repo(repo_id, repo_type=kind, private=True, exist_ok=True)
        print(f"repo ready (private): {repo_id}")

    # Cards plus eval json always exist. Weights upload only if trained.
    api.upload_file(
        path_or_fileobj=str(ROOT / "cards/MAIN_MODEL_CARD.md"),
        path_in_repo="README.md",
        repo_id="eulogik/bharat-embed-270m-gemma2",
        repo_type="model",
    )
    eval_json = KIO / "eval/results.json"
    if eval_json.exists():
        api.upload_file(
            path_or_fileobj=str(eval_json),
            path_in_repo="eval/results.json",
            repo_id="eulogik/bharat-embed-270m-gemma2",
            repo_type="model",
        )
    ckpt = KIO / "checkpoints/best_indic_merged"
    if ckpt.exists():
        api.upload_folder(
            folder_path=str(ckpt),
            repo_id="eulogik/bharat-embed-270m-gemma2",
            repo_type="model",
        )
        print("weights pushed")
    else:
        print("weights missing, card only. Train first.")
    onnx_dir = KIO / "onnx"
    if (onnx_dir / "model_int8.onnx").exists():
        api.upload_folder(
            folder_path=str(onnx_dir),
            repo_id="eulogik/bharat-embed-270m-gemma2",
            repo_type="model",
            path_in_repo="onnx",
        )
        print("onnx pack pushed")
    assets = ROOT / "assets"
    if (assets / "truncation.png").exists():
        api.upload_folder(
            folder_path=str(assets),
            repo_id="eulogik/bharat-embed-270m-gemma2",
            repo_type="model",
            path_in_repo="assets",
        )
        print("assets pushed")
    legal = KIO / "checkpoints/legal_merged"
    if legal.exists():
        api.upload_file(
            path_or_fileobj=str(ROOT / "cards/LEGAL_ADAPTER_CARD.md"),
            path_in_repo="README.md",
            repo_id="eulogik/bharat-legal-embed-270m",
            repo_type="model",
        )
        api.upload_folder(
            folder_path=str(legal),
            repo_id="eulogik/bharat-legal-embed-270m",
            repo_type="model",
            path_in_repo="adapter",
        )
        print("legal adapter pushed")
    api.upload_file(
        path_or_fileobj=str(ROOT / "cards/DATASET_CARD.md"),
        path_in_repo="README.md",
        repo_id="eulogik/bharat-embed-triplets-40k",
        repo_type="dataset",
    )
    print("dataset card pushed")
    gguf_dir = KIO / "gguf"
    gguf_files = ["bharat-embed-BF16.gguf", "bharat-embed-Q8_0.gguf", "bharat-embed-Q6_K.gguf", "bharat-embed-Q4_K_M.gguf"]
    if (gguf_dir / "bharat-embed-Q4_K_M.gguf").exists():
        api.upload_file(
            path_or_fileobj=str(ROOT / "cards/GGUF_CARD.md"),
            path_in_repo="README.md",
            repo_id="eulogik/bharat-embed-270m-gemma2-GGUF",
            repo_type="model",
        )
        for gf in gguf_files:
            api.upload_file(
                path_or_fileobj=str(gguf_dir / gf),
                path_in_repo=gf,
                repo_id="eulogik/bharat-embed-270m-gemma2-GGUF",
                repo_type="model",
            )
        print("gguf pack pushed")
    print("push done")


if __name__ == "__main__":
    main()
