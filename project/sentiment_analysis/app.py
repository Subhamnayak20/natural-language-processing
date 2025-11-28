# app.py
import os
import pickle
import traceback
from typing import Tuple
import gradio as gr

MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
PORT = int(os.environ.get("PORT", 8080))

def load_model(path: str):
    try:
        with open(path, "rb") as f:
            clf = pickle.load(f)
        print(f"Loaded model from {path}")
        return clf
    except Exception as e:
        print(f"Failed to load model from {path}: {e}")
        traceback.print_exc()
        return None

clf = load_model(MODEL_PATH)

def predict_review(text: str) -> Tuple[str, float]:
    if clf is None:
        return "model-not-loaded", 0.0
    try:
        pred = clf.predict([text])[0]
        label = "yes" if (isinstance(pred, (int, float)) and int(pred) == 1) or str(pred).lower() in ("1","yes","positive","pos","true") else "no"
        conf = 1.0
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba([text])
            try:
                idx = list(clf.classes_).index(int(pred)) if isinstance(pred, (int, float)) else probs.argmax(axis=1)[0]
            except Exception:
                idx = probs.argmax(axis=1)[0]
            conf = float(probs[0, idx])
        elif hasattr(clf, "decision_function"):
            import math
            df = clf.decision_function([text])[0]
            conf = 1.0 / (1.0 + math.exp(-df))
            conf = max(0.0, min(1.0, conf))
        return label, round(conf, 4)
    except Exception as e:
        traceback.print_exc()
        return f"error: {e}", 0.0

# CSS using Gradio static file path
CSS = """
body, .gradio-container {
  height: 100vh;
  margin: 0;
  /* IMPORTANT: use Gradio's static file URL format */
  background-image: url("/file=bg.jpg");
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
  font-family: Arial, Helvetica, sans-serif;
}
.center-card {
  width: 520px;
  max-width: 90%;
  margin: 6vh auto;
  padding: 22px;
  background: rgba(0,0,0,0.55);
  color: #fff;
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.55);
}
.gr-textbox { background: rgba(255,255,255,0.95); color: #111; border-radius: 8px; }
.gr-button { border-radius: 8px; padding: 10px 16px; font-weight: 600; }
"""

with gr.Blocks(title="IMDB Sentiment (yes/no)") as demo:
    gr.HTML(f"<style>{CSS}</style>")
    with gr.Row():
        with gr.Column(elem_classes="center-card"):
            gr.Markdown("## IMDB Sentiment\nsmall app — `yes` = positive, `no` = negative")
            review_input = gr.Textbox(lines=4, placeholder="Write a movie review here...", label="Your review")
            predict_btn = gr.Button("Submit")
            out_label = gr.Textbox(label="Prediction (yes / no)")
            out_conf = gr.Textbox(label="Confidence (0-1)")
            gr.Markdown("Examples: \n- `I loved the acting and the story.` → yes\n- `This was boring and a waste of time.` → no")
    predict_btn.click(fn=predict_review, inputs=review_input, outputs=[out_label, out_conf])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080, share=False)
