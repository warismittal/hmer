"""Gradio demo: draw or upload math -> LaTeX. Works on HF Spaces free CPU.
    python app.py   (expects best.pt next to it, or set CKPT env var)"""
import os
import numpy as np
import gradio as gr
from PIL import Image
from predict import load, predict

model, vocab = load(os.environ.get("CKPT", "best.pt"))


def run(sketch, upload):
    img = None
    if upload is not None:
        img = Image.fromarray(upload)
    elif sketch is not None:
        comp = sketch["composite"] if isinstance(sketch, dict) else sketch
        if comp is not None:
            arr = np.asarray(comp)
            if arr.ndim == 3 and arr.shape[2] == 4:          # transparent canvas -> white bg
                a = arr[..., 3:4] / 255.0
                arr = (arr[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
            img = Image.fromarray(arr)
    if img is None:
        return "", ""
    latex = predict(model, vocab, img)
    return latex, f"$${latex}$$"


with gr.Blocks(title="Handwritten Math → LaTeX") as demo:
    gr.Markdown("# ✍️ Handwritten Math → LaTeX\nCNN encoder + Transformer decoder trained on CROHME.")
    with gr.Row():
        sketch = gr.Sketchpad(label="Draw", type="numpy",
                              brush=gr.Brush(colors=["#000000"], default_size=4))
        upload = gr.Image(label="…or upload", type="numpy")
    btn = gr.Button("Convert", variant="primary")
    code = gr.Textbox(label="LaTeX", show_copy_button=True)
    render = gr.Markdown(latex_delimiters=[{"left": "$$", "right": "$$", "display": True}])
    btn.click(run, [sketch, upload], [code, render])

if __name__ == "__main__":
    demo.launch()
