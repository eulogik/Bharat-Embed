"""Charts and diagrams from measured numbers only. Nothing invented.

Run: python scripts/make_charts.py
Reads Kioxia eval jsons if present, writes assets/*.png (repo dir).
Hero plus architecture always render (static facts). Truncation renders
only when eval/truncate.json exists, else writes a placeholder-free skip.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)


def hero():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.text(0.05, 0.62, "Bharat-Embed 270M", fontsize=30, color="white", weight="bold")
    ax.text(0.05, 0.32, "Hinglish and Hindi retrieval RAG  |  text only  |  Apache-2.0  |  by Eulogik",
            fontsize=12, color="#cbd5e1")
    fig.savefig(ASSETS / "hero.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    print("hero ok")


def architecture():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    boxes = [("Text in", 0.02), ("270M tower", 0.24), ("Mean pool", 0.46), ("768d", 0.66), ("128d MRL", 0.82)]
    for label, x in boxes:
        ax.add_patch(patches.FancyBboxPatch((x, 0.3), 0.14, 0.4, boxstyle="round,pad=0.02",
                                             facecolor="#e0e7ff", edgecolor="#4f46e5"))
        ax.text(x + 0.07, 0.5, label, ha="center", va="center", fontsize=10)
    for i in range(len(boxes) - 1):
        ax.annotate("", xy=(boxes[i + 1][1], 0.5), xytext=(boxes[i][1] + 0.14, 0.5),
                    arrowprops={"arrowstyle": "->", "color": "#4f46e5"})
    fig.savefig(ASSETS / "architecture.png", bbox_inches="tight")
    print("architecture ok")


def truncation():
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = Path("/Volumes/KIOXIA 1TB/bharat-embed/eval/truncate.json")
    if not src.exists():
        print("truncate.json missing, skip chart")
        return
    d = json.loads(src.read_text())
    dims = [768, 512, 256, 128]
    vals = [d.get(f"ours_{x}") for x in dims]
    if any(v is None for v in vals):
        print(f"incomplete sweep {d}, skip chart")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(x) for x in dims], vals, color="#4f46e5")
    ax.set_ylim(min(vals) - 0.05, max(vals) + 0.02)
    ax.set_xlabel("dims")
    ax.set_ylabel("Hindi IndicQA NDCG@10")
    ax.set_title("Bharat-Embed holds to 128d (measured)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
    fig.savefig(ASSETS / "truncation.png", bbox_inches="tight")
    print(f"truncation ok: {vals}")


if __name__ == "__main__":
    hero()
    architecture()
    truncation()
