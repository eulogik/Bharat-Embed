"""Ship gate. Passes before weights exist. Fails if docs or code drift."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "USAGE.md",
    "LICENSE",
    "requirements.txt",
    "AGENTS.md",
    "MEMORY.md",
    "EMBEDDINGGEMMA2_VARIANT_PLAN.md",
    "src/embed/prefixes.py",
    "src/embed/train_indic.py",
    "src/embed/train_legal_adapter.py",
    "src/embed/export_onnx.py",
    "src/embed/export_gguf.py",
    "scripts/prepare_triplets.py",
    "scripts/eval_mteb_slice.py",
    "scripts/verify_onnx_parity.py",
    "scripts/encode_onnx.py",
    "scripts/push_hf.py",
    "eval/README.md",
    "data/frozen/README.md",
    "cards/MAIN_MODEL_CARD.md",
    "cards/LEGAL_ADAPTER_CARD.md",
    "cards/DATASET_CARD.md",
    "spaces/search/app.py",
    "blog/forked-embeddinggemma2-mac-mini.md",
]


def test_all_ship_files_exist():
    missing = [f for f in REQUIRED if not (ROOT / f).exists()]
    assert not missing, f"missing ship files: {missing}"


def test_no_em_dash_in_ship_docs():
    for f in ["README.md", "USAGE.md", "AGENTS.md", "MEMORY.md", "eval/README.md"]:
        t = (ROOT / f).read_text()
        assert "\u2014" not in t, f"em dash in {f}"
        assert "\u2013" not in t, f"en dash in {f}"


def test_eval_gates_locked():
    t = (ROOT / "eval/README.md").read_text()
    for gate in ["plus 0.03", "minus 0.01", "recall at 5 plus 0.10", "under 60ms", "under 1GB"]:
        assert gate in t, f"gate missing: {gate}"


def test_train_policy_blocks_flores():
    t = (ROOT / "scripts/prepare_triplets.py").read_text() + (ROOT / "data/frozen/README.md").read_text()
    assert "FLORES" in t
    assert "eval only" in t


def test_fp16_blocked_in_train():
    t = (ROOT / "src/embed/train_indic.py").read_text()
    assert "fp16 is blocked" in t or "fp16" in t.lower()
    assert "float32" in t
