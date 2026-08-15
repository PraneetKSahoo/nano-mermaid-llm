import argparse
import re
import time

import torch

from paths import FINETUNE_BEST
from model import load_model_from_config
from tokenizer import load_tokenizer

def _openers_from_templates():
    
    try:
        from templates import TEMPLATES
    except ImportError:
        return ["graph TD\n    A"]

    prefixes = set()
    for t in TEMPLATES:
        lines = t["mermaid"].split("\n")
        head = lines[0].rstrip()
        if len(lines) < 2:
            prefixes.add(head + "\n")
            continue
        body = lines[1]
        indent = body[: len(body) - len(body.lstrip())]
        node_id = body.lstrip()[:1]
        prefixes.add(f"{head}\n{indent}{node_id}")
    return sorted(prefixes)


DIAGRAM_OPENERS = _openers_from_templates()


def _header_of(prefix):
    """The diagram type ('graph TD') from a forced prefix ('graph TD\n    A')."""
    return prefix.split("\n", 1)[0].strip() if prefix else None

# Patterns that belong to a diagram type OTHER than the one we committed to.
# Any generated line matching one of these is foreign syntax and gets dropped.
_FOREIGN_LINE_PATTERNS = {
    "graph TD": [r"^\[\*\]", r"-->\s*\S+:\s", r"(-->>|->>)"],
    "graph TB": [r"^\[\*\]", r"-->\s*\S+:\s", r"(-->>|->>)"],
    "stateDiagram-v2": [r"^subgraph\b", r"(-->>|->>)", r"^\w+\["],
}


def sanitize_mermaid_code(code: str, diagram_type: str = None) -> str:
    lines = code.split("\n")
    sanitized = []

    clean_type = diagram_type.strip() if diagram_type else None
    foreign = _FOREIGN_LINE_PATTERNS.get(clean_type, []) if clean_type else []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if foreign and any(re.search(p, line) for p in foreign):
            continue

        # Collapse a duplicated node opener like "A[foo A[bar" -> "A[bar"
        if line.count("A[") > 1:
            line = "A[" + line.rsplit("A[", 1)[1]

        # Repair mismatched bracket/brace pairs
        line = re.sub(r"\[([^\]\}]*)\}", r"[\1]", line)
        line = re.sub(r"\{([^\]\}]*)\]", r"{\1}", line)
        line = re.sub(r"\]+", "]", line)
        line = re.sub(r"\}+", "}", line)

        if "[" in line and "]" not in line:
            line += "]"
        if "{" in line and "}" not in line:
            line += "}"

        sanitized.append(line)

    return "\n    ".join(sanitized)


class MermaidGenerator:
    def __init__(self, checkpoint_path=FINETUNE_BEST):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.config = load_model_from_config()
        self.tokenizer = load_tokenizer()

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint missing at {checkpoint_path}")

        print(f"Loading NanoMermaid weights from '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        if "sample_acc" in checkpoint:
            print(f"    (checkpoint all-slots accuracy: {checkpoint['sample_acc']:.3f})")

        self.mermaid_id = self.tokenizer.token_to_id("<|mermaid|>")
        self.prompt_id = self.tokenizer.token_to_id("<|prompt|>")
        self.end_id = self.tokenizer.token_to_id("<|endoftext|>")

        self._opener_token_ids = [self.tokenizer.encode(o).ids for o in DIAGRAM_OPENERS]
        print("NanoMermaid model ready for inference!")

    @torch.no_grad()
    def _generate_constrained_header(self, idx, temperature, greedy=False):
        """Force the opening tokens onto one of the known diagram openers."""
        candidates = list(range(len(self._opener_token_ids)))
        committed_len = 0

        while candidates:
            remaining = [c for c in candidates if len(self._opener_token_ids[c]) > committed_len]
            if not remaining:
                break

            allowed = {self._opener_token_ids[c][committed_len] for c in remaining}

            idx_cond = idx[:, -self.config["max_seq_len"]:]
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :].clone() / temperature

            mask = torch.full_like(logits, float("-inf"))
            allowed_idx = torch.tensor(sorted(allowed), device=self.device)
            mask[:, allowed_idx] = logits[:, allowed_idx]

            if greedy:
                idx_next = mask.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(mask, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            chosen = idx_next.item()
            candidates = [
                c for c in remaining
                if self._opener_token_ids[c][committed_len] == chosen
            ]
            committed_len += 1

            done = [c for c in candidates if len(self._opener_token_ids[c]) == committed_len]
            if done:
                return idx, DIAGRAM_OPENERS[done[0]]

        return idx, None

    @torch.no_grad()
    def generate(self, prompt: str, temperature: float = 0.1, top_k: int = 20,
                 max_new_tokens: int = 180, max_consecutive_repeat: int = 2,
                 greedy: bool = False):
        
        if greedy:
            temperature, top_k = 1.0, 1

        p_ids = self.tokenizer.encode(prompt).ids
        input_ids = [self.mermaid_id] + p_ids + [self.prompt_id]
        idx = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        start_time = time.time()
        idx, committed_type = self._generate_constrained_header(idx, temperature, greedy=greedy)

        generated_so_far = []
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config["max_seq_len"]:]
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :] / temperature

            if len(generated_so_far) >= max_consecutive_repeat:
                last_n = generated_so_far[-max_consecutive_repeat:]
                if len(set(last_n)) == 1:
                    logits[:, last_n[0]] = float("-inf")

            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            token_id = idx_next.item()
            generated_so_far.append(token_id)
            if token_id == self.end_id:
                break

        latency_ms = (time.time() - start_time) * 1000

        generated_ids = idx[0].cpu().numpy().tolist()[len(input_ids):]
        raw_decoded = self.tokenizer.decode(generated_ids).replace("<|endoftext|>", "").strip()

        clean_type = _header_of(committed_type)

        if clean_type:
            raw_code = raw_decoded
        elif "graph" in raw_decoded:
            raw_code = "graph" + raw_decoded.split("graph", 1)[1]
        else:
            raw_code = "graph TD\n    A[" + raw_decoded

        return sanitize_mermaid_code(raw_code, diagram_type=clean_type), latency_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Mermaid from a description.")
    parser.add_argument("prompt", nargs="?", default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    generator = MermaidGenerator()

    # Slot values here are deliberately ones the model never saw in fine-tuning,
    # so these prompts test copying rather than recall.
    prompts = [args.prompt] if args.prompt else [
        "Enter Number. If Even go to B, else go to C.",
        "First Open Door, then Take Key, and finally Read Note.",
        "Check if Box Full. If yes, Save List. If no, Ask Name.",
        "Do Count Coin until Bag Full, then Stop Game.",
        "Play Song splits into Show Card and Send Letter.",
    ]

    for p in prompts:
        code, latency = generator.generate(
            p, temperature=args.temperature, top_k=args.top_k, greedy=args.greedy
        )
        print("\n" + "=" * 60)
        print(f"PROMPT: {p}")
        print(f"Generated in {latency:.2f} ms")
        print(code)
    print("=" * 60)
