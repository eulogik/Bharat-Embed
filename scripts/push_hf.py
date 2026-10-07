"""Push model, adapter, dataset, and spaces metadata to HF.

Run: python scripts/push_hf.py --dry-run
Real run needs HF_TOKEN with write access to eulogik org.

Repos:
- eulogik/bharat-embed-270m-gemma2
- eulogik/bharat-embed-270m-gemma2-GGUF
- eulogik/bharat-embed-triplets-40k
- eulogik/bharat-legal-embed-270m
"""

import argparse

REPOS = [
    "eulogik/bharat-embed-270m-gemma2",
    "eulogik/bharat-embed-270m-gemma2-GGUF",
    "eulogik/bharat-embed-triplets-40k",
    "eulogik/bharat-legal-embed-270m",
]

CARD_META = {
    "library_name": "sentence-transformers",
    "pipeline_tag": "sentence-similarity",
    "license": "apache-2.0",
    "base_model": "google/embeddinggemma-2",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"repos: {REPOS}")
    print(f"meta: {CARD_META}")
    if args.dry_run:
        print("dry run ok. would create repos, push weights, cards, eval json, and collection.")
        return
    print("real run: huggingface_hub create_repo plus upload_folder per repo. Needs HF_TOKEN.")


if __name__ == "__main__":
    main()
