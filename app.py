"""Gradio app for Hugging Face Spaces deployment."""

try:
    import gradio as gr
except ImportError:  # pragma: no cover
    raise SystemExit("gradio is required: pip install gradio")

from leakauditkit.detectors.ngram import NgramDetector


def run_audit(train_text: str, eval_text: str, threshold: float):
    train = [{"id": f"t{i}", "text": t} for i, t in enumerate(train_text.splitlines()) if t.strip()]
    eval_ = [{"id": f"e{i}", "text": t} for i, t in enumerate(eval_text.splitlines()) if t.strip()]
    det = NgramDetector(n=5, threshold=threshold, min_shared=3)
    res = det.detect(train, eval_)
    lines = [f"matches={len(res.matches)} contamination_rate={res.contamination_rate:.2%}"]
    for m in res.matches[:20]:
        lines.append(f"{m.eval_id} <-> {m.train_id} (score={m.score})")
    return "\n".join(lines)


demo = gr.Interface(
    fn=run_audit,
    inputs=[gr.Textbox(lines=10, label="Train corpus (one doc per line)"),
            gr.Textbox(lines=10, label="Eval corpus (one doc per line)"),
            gr.Slider(0.1, 0.9, value=0.3, label="Threshold")],
    outputs=gr.Textbox(label="Audit report"),
    title="LeakAuditKit — data leakage audit",
    description="Paste train and eval documents; get contamination matches.",
)

if __name__ == "__main__":
    demo.launch()