"""Show what the model ACTUALLY emits, before any post-processing.

generate.py runs the output through sanitize_mermaid_code(), which repairs
brackets and drops lines. If something looks wrong in the final diagram, this
tells you whether the model produced it or the sanitiser did.

Run:  python debug_raw.py "your prompt here"
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from paths import FINETUNE_BEST
from model import load_model_from_config
from tokenizer import load_tokenizer
from generate import MermaidGenerator, sanitize_mermaid_code


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else \
        "First Grab Milk, then Heat Pot, and finally Fill Cup."

    gen = MermaidGenerator()
    tok = gen.tokenizer

    p_ids = tok.encode(prompt).ids
    input_ids = [gen.mermaid_id] + p_ids + [gen.prompt_id]
    idx = torch.tensor([input_ids], dtype=torch.long, device=gen.device)

    print(f"\nPROMPT      : {prompt}")
    print(f"prompt tokens: {tok.encode(prompt).tokens}")

    idx, committed = gen._generate_constrained_header(idx, 0.1, greedy=True)
    print(f"\nheader committed to: {committed!r}")

    with torch.no_grad():
        for _ in range(180):
            logits, _ = gen.model(idx[:, -gen.config["max_seq_len"]:])
            nxt = int(logits[:, -1, :].argmax(dim=-1).item())
            if nxt == gen.end_id:
                break
            idx = torch.cat([idx, torch.tensor([[nxt]], device=gen.device)], dim=1)

    out_ids = idx[0].cpu().tolist()[len(input_ids):]
    print(f"\nRAW TOKENS  : {tok.decode(out_ids, skip_special_tokens=False)!r}")
    print(f"\nTOKEN LIST  : {[tok.id_to_token(i) for i in out_ids]}")

    raw = tok.decode(out_ids).replace("<|endoftext|>", "").strip()
    print(f"\n[1] RAW DECODED (straight from the model):\n{raw}")

    # [2] replicate generate()'s reassembly exactly -- the remaining suspect
    clean_type = committed.strip() if committed else None
    if clean_type:
        reassembled = clean_type + "\n" + raw.split(clean_type, 1)[-1]
    elif "graph" in raw:
        reassembled = "graph" + raw.split("graph", 1)[1]
    else:
        reassembled = "graph TD\n    A[" + raw
    print(f"\n[2] AFTER REASSEMBLY in generate():\n{reassembled}")

    print(f"\n[3] AFTER SANITISE:\n{sanitize_mermaid_code(reassembled, diagram_type=clean_type)}")

    print("\n" + "=" * 60)
    print("Where did the node letter disappear?")
    print("  missing already at [1] -> the MODEL dropped it")
    print("  present [1], gone [2]  -> the split() reassembly in generate() ate it")
    print("  present [2], gone [3]  -> sanitize_mermaid_code is at fault")


if __name__ == "__main__":
    main()
