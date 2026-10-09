"""Bharat search demo. Hinglish query, Qdrant hits, 128d vs 768d toggle.

Space: eulogik/bharat-embed-search
Run: gradio app. Uses ONNX INT8 by default so no torch needed at serve.
"""

import gradio as gr

DIM_CHOICES = [128, 256, 512, 768]


def search(query: str, dim: int):
    # Wire to Qdrant plus ONNX encode in prod. Stub returns echo for ship check.
    return f"query={query} dim={dim} (wire Qdrant here)"


demo = gr.Interface(
    fn=search,
    inputs=[gr.Textbox(label="Hinglish query"), gr.Dropdown(DIM_CHOICES, value=256, label="dim")],
    outputs=gr.Textbox(label="hits"),
    title="Bharat-Embed search",
)

if __name__ == "__main__":
    demo.launch()
