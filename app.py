"""NanoMermaid web interface.

Updated for the Tier-2 model. Changes from the original:
  * import path fixed -- src/ modules use flat imports, so src must be on sys.path
  * sample prompts replaced (the old ones targeted templates that no longer exist)
  * decoding defaults are now near-greedy, which is correct for structured output
  * device name is detected rather than hardcoded
  * the UI states the Title-Case requirement, which is the single most common
    reason a prompt produces hallucinated labels
"""
import html
import sys
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Colab (and any remote VM) serves on the VM's own localhost, which your browser
# cannot reach -- hence ERR_CONNECTION_REFUSED on 127.0.0.1:7860. share=True
# opens a public tunnel URL instead.
IN_COLAB = "google.colab" in sys.modules

import torch                                   # noqa: E402
from generate import MermaidGenerator          # noqa: E402
from templates import TEMPLATES                # noqa: E402

print("Starting NanoMermaid Web Interface...")
try:
    generator = MermaidGenerator()
except FileNotFoundError as exc:
    raise SystemExit(
        f"\n{exc}\n\nTrain the model first, or copy a checkpoint into "
        f"checkpoints/finetune_mermaid/best_mermaid_model.pt\n"
    )

DEVICE_NAME = (
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
)
N_PARAMS = generator.model.get_num_params()

# Prompts matching the 10 shapes the model was actually trained on.
# Note the Title-Case labels -- the model locates slot values by casing.
SAMPLE_PROMPTS = [
    "First Grab Milk, then Heat Pot, and finally Fill Cup.",
    "Do Log Name. If Word Wrong then Lock Door else Open Door.",
    "Check if Box Full. Yes leads to Save List, on no Ask Name.",
    "Begin with Open Door, then Take Key, then Read Note, and last Lock Door.",
    "Keep doing Count Coin until Bag Full, then Stop Game.",
    "After Play Song, run both Show Card and Send Letter.",
    "Both Read Book and Write Note lead to Save Page.",
    "Run Enter Number, then check Path Even: yes jump to B, no jump to C.",
    "Evaluate Check Card. If Card New use Open Box, if Card Old use Hide Box, if Card Lost use Send Note.",
]

SHAPE_HELP = """
| shape | example phrasing |
|---|---|
| 2-step chain | `Carry out **Grab Milk**, next **Heat Pot**.` |
| 3-step chain | `Three steps: **A**, **B**, **C**.` |
| 4-step chain | `Begin with **A**, then **B**, then **C**, and last **D**.` |
| decision | `Check if **Box Full**. Yes leads to **Save List**, on no **Ask Name**.` |
| step then decision | `Do **Log Name**. Next look at **Word Wrong**: yes **Lock Door**, no **Open Door**.` |
| goto labels | `Run **Enter Number**, then check **Path Even**: yes jump to **B**, no jump to **C**.` |
| three-way branch | `Evaluate **A**. If **X** use **B**, if **Y** use **C**, if **Z** use **D**.` |
| loop until | `Keep doing **Count Coin** until **Bag Full**, then go to **Stop Game**.` |
| fork | `After **A**, run both **B** and **C**.` |
| merge | `Either **A** or **B** lead to **C**.` |
"""


def render_mermaid_html(mermaid_code):
    """Embed Mermaid.js in an isolated iframe with a native SVG download button."""
    raw_doc = f"""<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>
        document.addEventListener("DOMContentLoaded", function() {{
            mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
        }});

        function downloadSVG() {{
            const svg = document.querySelector('svg');
            if (!svg) {{
                alert('Diagram not ready for download!');
                return;
            }}
            const svgData = new XMLSerializer().serializeToString(svg);
            const blob = new Blob([svgData], {{ type: 'image/svg+xml;charset=utf-8' }});
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'nano_mermaid_flowchart.svg';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        }}
    </script>
    <style>
        body {{
            margin: 0;
            padding: 15px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background-color: #ffffff;
            font-family: system-ui, -apple-system, sans-serif;
        }}
        .download-btn {{
            margin-top: 15px;
            background-color: #4f46e5;
            color: #ffffff;
            border: none;
            padding: 8px 18px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 13px;
            cursor: pointer;
            transition: background-color 0.2s ease;
        }}
        .download-btn:hover {{ background-color: #4338ca; }}
    </style>
</head>
<body>
    <pre class="mermaid">
{mermaid_code}
    </pre>
    <button class="download-btn" onclick="downloadSVG()">Download SVG Diagram</button>
</body>
</html>"""
    escaped_doc = html.escape(raw_doc)
    return (
        f'<iframe srcdoc="{escaped_doc}" style="width: 100%; height: 340px; '
        f'border: 1px solid #e5e7eb; border-radius: 12px; background: #ffffff;"></iframe>'
    )


def _casing_warning(prompt_text):
    """The model finds slot values by Title-Case. Warn when a prompt has none."""
    words = [w.strip(".,:;!?") for w in prompt_text.split()]
    content = [w for w in words[1:] if w.isalpha()]
    titled = [w for w in content if w[:1].isupper()]
    if content and len(titled) < 2:
        return (
            "⚠️ **No Title-Case labels detected.** This model locates node labels by "
            "capitalisation — it copies spans like `Enter Number`, not `enter number`. "
            "Lowercase prompts usually produce invented labels. Try capitalising the "
            "words you want to appear in the diagram."
        )
    return ""


def predict_diagram(prompt_text, temp, top_k, greedy):
    if not prompt_text.strip():
        return "<p style='color: gray;'>Please enter a workflow description.</p>", "", "0 ms", ""

    raw_code, latency = generator.generate(
        prompt_text, temperature=temp, top_k=int(top_k), greedy=greedy
    )
    return (
        render_mermaid_html(raw_code),
        raw_code,
        f"⚡ Inference Latency: {latency:.1f} ms ({DEVICE_NAME})",
        _casing_warning(prompt_text),
    )


# Gradio 6 moved `theme` from the Blocks constructor to launch().
_GR_MAJOR = int(gr.__version__.split(".")[0])
_BLOCKS_KW, _LAUNCH_KW = {"title": "NanoMermaid LLM"}, {}
if _GR_MAJOR >= 6:
    _LAUNCH_KW["theme"] = gr.themes.Soft()
else:
    _BLOCKS_KW["theme"] = gr.themes.Soft()

with gr.Blocks(**_BLOCKS_KW) as demo:
    gr.Markdown(
        f"""
        # 🧜‍♂️ NanoMermaid LLM — Diagram Synthesizer
        ### Pretrained & fine-tuned from scratch ({N_PARAMS/1e6:.1f}M parameters)
        A GPT-style transformer built from nothing — custom BPE tokenizer, pretrained on
        TinyStories, then fine-tuned to turn workflow descriptions into **Mermaid.js** diagrams.
        Running on **{DEVICE_NAME}**.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Workflow Description / Prompt",
                placeholder="e.g. First Grab Milk, then Heat Pot, and finally Fill Cup.",
                lines=4,
            )
            casing_note = gr.Markdown("")

            with gr.Accordion("What this model understands", open=False):
                gr.Markdown(
                    "**Write node labels in Title Case.** The model copies spans like "
                    "`Enter Number` out of your prompt — lowercase text gives it nothing "
                    "to copy and it will invent labels instead.\n\n"
                    f"It knows these {len(TEMPLATES)} diagram shapes:\n" + SHAPE_HELP
                )

            with gr.Row():
                greedy_toggle = gr.Checkbox(
                    value=True,
                    label="Greedy (deterministic)",
                    info="Recommended. Structured output has one correct answer.",
                )
            with gr.Row():
                temp_slider = gr.Slider(
                    minimum=0.05, maximum=1.5, value=0.1, step=0.05,
                    label="Temperature", info="Only used when Greedy is off",
                )
                topk_slider = gr.Slider(
                    minimum=1, maximum=100, value=20, step=1, label="Top-K",
                )

            submit_btn = gr.Button("🚀 Generate Diagram", variant="primary")

            gr.Examples(
                examples=SAMPLE_PROMPTS,
                inputs=prompt_input,
                label="Example Prompts (click to try)",
            )

        with gr.Column(scale=1):
            latency_badge = gr.Markdown("⚡ Inference Latency: Ready")
            gr.Markdown("### 📊 Rendered Visual Diagram")
            rendered_output = gr.HTML("*(Your rendered flowchart diagram will appear here)*")
            gr.Markdown("### 💻 Generated Raw Mermaid.js Code")
            raw_code_output = gr.Code(language="markdown", label="Mermaid Code")

    submit_btn.click(
        fn=predict_diagram,
        inputs=[prompt_input, temp_slider, topk_slider, greedy_toggle],
        outputs=[rendered_output, raw_code_output, latency_badge, casing_note],
    )
    prompt_input.submit(
        fn=predict_diagram,
        inputs=[prompt_input, temp_slider, topk_slider, greedy_toggle],
        outputs=[rendered_output, raw_code_output, latency_badge, casing_note],
    )

if __name__ == "__main__":
    if IN_COLAB:
        print("Colab detected -> opening a public share link "
              "(127.0.0.1 is unreachable from your browser here).")
    demo.launch(
        share=IN_COLAB,
        inbrowser=not IN_COLAB,
        **_LAUNCH_KW,
    )
