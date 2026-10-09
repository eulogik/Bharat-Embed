"""Real MTEB slice: base text-only vs merged ckpt. Hindi gates live here.

Tasks: IndicQARetrieval (retrieval, Hindi filter), IndicCrosslingualSTS (STS).
Run: python scripts/eval_mteb_real.py --models base,ours --dry-run
Real: encodes on CPU fp32 (slow but swap safe), writes Kioxia eval JSON.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TASKS = ["IndicQARetrieval", "IndicCrosslingualSTS"]
KIO = Path("/Volumes/KIOXIA 1TB/bharat-embed")


def load_st(path: str):
    import torch
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        path,
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"torch_dtype": torch.float32},
        device="cpu",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="base,ours")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(KIO / "eval/mteb_slice.json"))
    args = ap.parse_args()

    print(f"tasks={TASKS} models={args.models}")
    if args.dry_run:
        print("dry run ok. would run 2 tasks x N models on CPU and write scores json.")
        return

    import mteb

    tasks = mteb.get_tasks(tasks=TASKS, languages=["hin"])
    print(f"loaded {len(tasks)} tasks: {[t.metadata.name for t in tasks]}")
    picked = {t.metadata.name: t for t in tasks}
    assert "IndicCrosslingualSTS" in picked, f"missing STS task, have {list(picked)}"
    assert "IndicQARetrieval" in picked, f"missing retrieval task, have {list(picked)}"

    paths = {
        "base": "google/embeddinggemma-2",
        "ours": str(KIO / "checkpoints/best_indic_merged"),
    }
    results = {}
    for name in args.models.split(","):
        model = load_st(paths[name.strip()])
        model_results = {}
        for task_name in ["IndicCrosslingualSTS", "IndicQARetrieval"]:
            ev = mteb.MTEB(tasks=[picked[task_name]])
            res = ev.run(model, output_folder=str(KIO / "eval" / f"mteb_{name.strip()}"))
            model_results[task_name] = str(res)[:1500]
            print(f"{name} {task_name} done")
        results[name.strip()] = model_results

    import json

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
