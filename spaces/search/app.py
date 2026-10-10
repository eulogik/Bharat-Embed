"""Bharat-Embed live search. Hinglish query over a built-in Hindi corpus.

Space: eulogik/bharat-embed-search (CPU, free tier fits 270M fp32).
Model loads text only from the public HF repo. Dim toggle re-encodes.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_ID = "eulogik/bharat-embed-270m-gemma2"

CORPUS = [
    ("GST refund", "GST refund portal par invoice proof ke saath apply karein"),
    ("PAN card update", "PAN card me naam ya address update karne ke steps"),
    ("UPI fail", "UPI payment fail ho to 48 ghante me refund aata hai"),
    ("Train refund", "train ticket cancel par refund niyam"),
    ("Rental deposit", "rental agreement me security deposit ki limit"),
    ("Home loan EMI", "home loan EMI calculate karne ka formula"),
    ("Aadhaar change", "Aadhaar me address change online kaise karein"),
    ("Passport renew", "passport renew ke liye documents aur fees"),
    ("DIFC notice", "DIFC employment me notice period ka rule"),
    ("GST late fee", "GSTR late filing par late fee kitni lagti hai"),
]

_model = None


def get_model():
    global _model
    if _model is None:
        import torch

        _model = SentenceTransformer(
            MODEL_ID,
            config_kwargs={"vision_config": None, "audio_config": None},
            model_kwargs={"torch_dtype": torch.float32},
            device="cpu",
        )
    return _model


def search(query: str, dim: int):
    m = get_model()
    dim = int(dim)
    q = np.asarray(m.encode(f"task: search result | query: {query}", truncate_dim=dim, normalize_embeddings=True))
    rows = []
    for title, text in CORPUS:
        d = np.asarray(m.encode(f"title: {title} | text: {text}", truncate_dim=dim, normalize_embeddings=True))
        rows.append((float(q @ d), title, text))
    rows.sort(reverse=True)
    return "\n".join(f"{s:.3f}  {t} : {x}" for s, t, x in rows[:5])


try:
    import gradio as gr

    demo = gr.Interface(
        fn=search,
        inputs=[gr.Textbox(label="Hinglish query"), gr.Dropdown([768, 512, 256, 128], value=256, label="dim")],
        outputs=gr.Textbox(label="top hits"),
        title="Bharat-Embed search (by Eulogik)",
        description="270M text-only Hindi retrieval. Same dim both sides.",
    )
    if __name__ == "__main__":
        demo.launch()
except ImportError:
    pass
