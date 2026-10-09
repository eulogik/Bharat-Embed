# USAGE.md - Bharat-Embed

Copy paste snippets. Byte exact prefixes. Same dim both sides. Always renorm after truncate.

## Install

```bash
pip install -U sentence-transformers transformers
pip install onnxruntime qdrant-client
```

## Text only load (270M, no vision or audio RAM)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "eulogik/bharat-embed-270m-gemma2",
    config_kwargs={"vision_config": None, "audio_config": None},
    model_kwargs={"torch_dtype": "bfloat16"},
)

q = model.encode(
    "GST refund kaise claim karein?",
    prompt_name="SearchQuery",
    truncate_dim=256,
    normalize_embeddings=True,
)
d = model.encode(
    "title: GST refund | text: apply through portal with invoice proof",
    truncate_dim=256,
    normalize_embeddings=True,
)
print(model.similarity(q, d))
```

Rules:
- Query uses SearchQuery prefix. Doc uses title plus text form.
- truncate_dim must match for queries and docs.
- normalize_embeddings must be True after truncate.

## ONNX CPU serve

```bash
pip install onnxruntime qdrant-client
python scripts/encode_onnx.py --model onnx/int8 --queries queries.txt --truncate 256
```

## GGUF and Ollama

```bash
ollama run hf.co/eulogik/bharat-embed-270m-gemma2-GGUF:Q4_K_M
```

llama.cpp uses mean pooling for this family:

```bash
./build/bin/llama-embedding -m bharat-embed-Q4_K_M.gguf --pooling mean -p "rental agreement ki security deposit limit?"
```

## RAG stack

Retrieve 100 with Bharat-Embed 128d, rerank top 20 with eulogik/flashrank-pro-base, generate with your LLM.

Find with: embeddinggemma2 hindi hinglish indic rag on-device onnx gguf qdrant mrl apache 2.0
