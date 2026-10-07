test:
	python3 -m pytest tests/ -q

dry-run:
	python3 scripts/prepare_triplets.py --dry-run
	python3 src/embed/train_indic.py --dry-run
	python3 src/embed/train_legal_adapter.py --dry-run
	python3 scripts/eval_mteb_slice.py --dry-run
	python3 scripts/verify_onnx_parity.py --dry-run
	python3 scripts/encode_onnx.py --dry-run
	python3 scripts/push_hf.py --dry-run

freeze:
	python3 scripts/prepare_triplets.py

train:
	python3 src/embed/train_indic.py --data data/frozen/triplets_40k.jsonl --epochs 3 --batch 16

legal:
	python3 src/embed/train_legal_adapter.py --data data/frozen/legal_10k.jsonl --epochs 1
