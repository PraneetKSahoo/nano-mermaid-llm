"""The metric that actually matters here: did the model COPY the slot values?

Validation cross-entropy is a poor proxy for this task. A model that nails the
template structure and guesses the slots still gets a respectable val loss,
because the slot tokens are a small fraction of the target sequence. 

Slot exact-match has no such blind spot. Either the value from the prompt appears
verbatim in the generated diagram or it does not.
"""
import re

import torch


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt, device, max_seq_len=512, max_new_tokens=128):
    """Deterministic decode. No sampling -- we are measuring the model, not the RNG."""
    mermaid_id = tokenizer.token_to_id("<|mermaid|>")
    prompt_id = tokenizer.token_to_id("<|prompt|>")
    end_id = tokenizer.token_to_id("<|endoftext|>")

    ids = [mermaid_id] + tokenizer.encode(prompt).ids + [prompt_id]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = []
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -max_seq_len:])
        nxt = int(logits[:, -1, :].argmax(dim=-1).item())
        if nxt == end_id:
            break
        out.append(nxt)
        idx = torch.cat([idx, torch.tensor([[nxt]], device=device)], dim=1)

    return tokenizer.decode(out)


def _normalize(code):
    return re.sub(r"\s+", " ", code).strip()


_NEEDLE_CACHE = {}


def _needle_context(template_name, slot_name):
    
    key = (template_name, slot_name)
    if key in _NEEDLE_CACHE:
        return _NEEDLE_CACHE[key]

    left = right = ""
    try:
        from templates import TEMPLATES
        pattern = next(t["mermaid"] for t in TEMPLATES if t["name"] == template_name)
        token = "{" + slot_name + "}"
        i = pattern.find(token)
        if i != -1:
            left = pattern[i - 1] if i > 0 else ""
            j = i + len(token)
            right = pattern[j] if j < len(pattern) else ""
            # "{{" in a format string renders as a single "{"
            if left == "{":
                left = "{"
            if right == "}":
                right = "}"
    except (ImportError, StopIteration):
        pass

    _NEEDLE_CACHE[key] = (left, right)
    return left, right


def _slot_hit(gen, sample, slot_name, value):
    left, right = _needle_context(sample.get("template", ""), slot_name)
    if not left and not right:
        return value in gen           # fallback: no template info available
    return f"{left}{value}{right}" in gen


@torch.no_grad()
def evaluate_slots(model, tokenizer, samples, device, max_seq_len=512, max_new_tokens=128):
    """Returns per-slot accuracy, whole-sample slot accuracy, and exact code match.

    slot_acc    -- fraction of individual slot values copied correctly
    sample_acc  -- fraction of samples where EVERY slot was copied correctly
    exact_match -- fraction where the generated diagram equals the target exactly
    """
    was_training = model.training
    model.eval()

    slots_total = slots_ok = 0
    samples_ok = exact_ok = 0
    per_sample = []

    for s in samples:
        gen = greedy_generate(
            model, tokenizer, s["prompt"], device,
            max_seq_len=max_seq_len, max_new_tokens=max_new_tokens,
        )
        values = list(s["slots"].values())
        hits = sum(1 for k, v in s["slots"].items() if _slot_hit(gen, s, k, v))

        slots_total += len(values)
        slots_ok += hits
        if hits == len(values):
            samples_ok += 1
        per_sample.append((bool(s.get("novel", False)), hits == len(values)))
        if _normalize(gen) == _normalize(s["mermaid"]):
            exact_ok += 1

    if was_training:
        model.train()

    n = max(1, len(samples))
    out = {
        "slot_acc": slots_ok / max(1, slots_total),
        "sample_acc": samples_ok / n,
        "exact_match": exact_ok / n,
    }
    # Break the score down by whether the slot values were built from the known
    # component lexicon or from arbitrary vocab spans. The gap between these two
    # numbers IS the generalisation gap: a model that scores well on pool values
    # and badly on novel ones is doing lexicon retrieval, not copying.
    for tag, want in (("pool", False), ("novel", True)):
        subset = [r for r in per_sample if r[0] is want]
        if subset:
            out[f"sample_acc_{tag}"] = sum(1 for _, ok in subset if ok) / len(subset)
    return out


@torch.no_grad()
def show_examples(model, tokenizer, samples, device, max_seq_len=512, n=3):
    """Print a few generations side by side with their targets."""
    for s in samples[:n]:
        gen = greedy_generate(model, tokenizer, s["prompt"], device, max_seq_len=max_seq_len)
        missing = [v for k, v in s["slots"].items() if not _slot_hit(gen, s, k, v)]
        print(f"\n  PROMPT : {s['prompt']}")
        print(f"  TARGET : {_normalize(s['mermaid'])}")
        print(f"  OUTPUT : {_normalize(gen)}")
        print(f"  MISSING SLOTS: {missing if missing else 'none'}")
