"""Halluci-Nation live demo: damage a fingerprint, repair it with our models, check it still matches.

Run:  python demo/gradio_app.py [--host 0.0.0.0] [--port 7860]       (demo/README.md: login node or GPU node)
Pick a test print (or a random one, or upload your own), choose the damage (stored test damage,
stacked contaminants, or a smudge you draw), the mask (true or predicted by our segmenter) and up to two
models. The report scores every version exactly as evaluation/evaluate.py does (demo/scoring.py).
This file is only the layout; the work is in demo/handlers.py.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: makes config/fpr/demo importable

import gradio as gr

import config
from demo import damage, handlers, models
from fpr.data.degradations import CONTAMINANTS

MODELS = models.available_models()
SAMPLES = damage.sample_ids()


def on_run(*inputs):
    """handlers.run_demo, with input mistakes shown as a pop-up instead of a stack trace."""
    try:
        return handlers.run_demo(*inputs)
    except ValueError as error:
        raise gr.Error(str(error))


def model_choices(first: bool) -> list[tuple[str, str]]:
    choices = [(f"{name}: {models.CANDIDATES[name]}", name) for name in MODELS]
    return choices if first else [(handlers.NO_MODEL, handlers.NO_MODEL)] + choices


with gr.Blocks(title="Halluci-Nation fingerprint repair") as demo:
    gr.Markdown("## Halluci-Nation: repair a contaminated fingerprint, then check it still matches\n"
                f"Test prints: **{config.DEMO_SPLIT}** set, data **{config.DATA_VERSION}**. "
                "Models only ever saw training prints.")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                sample = gr.Dropdown(SAMPLES, value=SAMPLES[0], label="Test print (participant_finger)")
                random_button = gr.Button("Random test print → repair")
            upload = gr.Image(type="numpy", image_mode="L", label="…or upload a 500 ppi print (overrides the list)")
            mode = gr.Radio(damage.DAMAGE_MODES, value=damage.TEST_DAMAGE, label="Damage")
            contaminants = gr.CheckboxGroup(CONTAMINANTS, value=["mehendi"],
                                            label="Contaminants (several = stacked in this order)")
            with gr.Row():
                severity = gr.Slider(0, 1, value=0.6, step=0.05, label="Severity")
                seed = gr.Number(value=0, precision=0, label="Seed")
            mask_source = gr.Radio(damage.MASK_SOURCES, value=damage.TRUE_MASK, label="Damage mask given to the model")
            model_a = gr.Dropdown(model_choices(True), value=models.DEFAULT_MODEL, label="Model A")
            model_b = gr.Dropdown(model_choices(False), label="Model B (side by side)",
                                  value="E20_fingergan_finetune" if "E20_fingergan_finetune" in MODELS else handlers.NO_MODEL)
            run_button = gr.Button("Damage and repair", variant="primary")
        with gr.Column(scale=1):
            editor = gr.ImageEditor(type="numpy", image_mode="RGBA", format="png", label="Draw a smudge here "
                                    "(used when Damage = draw a smudge)",
                                    brush=gr.Brush(colors=["#202020", "#808080", "#ffffff"], default_size=12))
    with gr.Row():
        inputs_shown = [gr.Image(label=name) for name in
                        ("Clean print", "Damaged (red = true mask, cyan = predicted)", "Mask given to the model")]
    with gr.Row():
        outputs_a = [gr.Image(label="Model A: repaired"), gr.Image(label="Model A: |repaired − clean|, normalised")]
        outputs_b = [gr.Image(label="Model B: repaired"), gr.Image(label="Model B: |repaired − clean|, normalised")]
    report = gr.Markdown()

    run_inputs = [sample, upload, editor, mode, contaminants, severity, seed, mask_source, model_a, model_b]
    run_outputs = inputs_shown + outputs_a + outputs_b + [report]
    for trigger in (sample.change, upload.change, demo.load):
        trigger(handlers.editor_value, [sample, upload], editor)
    run_button.click(on_run, run_inputs, run_outputs)
    random_button.click(handlers.random_sample, None, sample).then(on_run, run_inputs, run_outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 on a compute node, so a tunnel can reach it")
    parser.add_argument("--port", type=int, default=config.DEMO_PORT)
    parser.add_argument("--no-warmup", action="store_true", help="skip the start-up run of every model")
    args = parser.parse_args()
    if not args.no_warmup:
        handlers.warm_up(MODELS, SAMPLES[0])
    demo.queue().launch(server_name=args.host, server_port=args.port)
