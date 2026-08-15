"""Which axis is the model actually failing on?

The validation set varies TWO things at once: slot values the model has never
seen, and paraphrases it has never seen. A single combined score cannot tell you
which one is costing you accuracy -- and the fix is completely different in each
case (more training steps vs more phrasings).

This script pulls them apart by building four eval sets:

  control  train values + train phrasings   -> should be near 1.0; if not,
                                               something is broken upstream
  values   NEW values   + train phrasings   -> pure copying ability
  phrasing train values + NEW phrasings     -> pure template-selection ability
  both     NEW values   + NEW phrasings     -> what train_finetune.py reports

Run:  python diagnose.py
"""
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from paths import FINETUNE_BEST
from model import load_model_from_config
from tokenizer import load_tokenizer
from slots import Lexicon
from templates import TEMPLATES
from templates import paraphrase_split
from dataset_mermaid import _template_slot_names, _render
from metrics import evaluate_slots, greedy_generate, _slot_hit, _normalize

N_PER_AXIS = 200
_SPLITS = {t["name"]: paraphrase_split(t) for t in TEMPLATES}
TRAIN_P = {k: v[0] for k, v in _SPLITS.items()}
VAL_P = {k: v[1] for k, v in _SPLITS.items()}


def build(lexicon, value_split, para_pool, n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        t = rng.choice(TEMPLATES)
        names = _template_slot_names(t)
        slots = lexicon.sample_slots(names, value_split, rng)
        s = _render(t, slots, rng.choice(para_pool[t["name"]]))
        s["novel"] = False
        out.append(s)
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_model_from_config()
    tokenizer = load_tokenizer()

    if not FINETUNE_BEST.exists():
        raise FileNotFoundError(f"No checkpoint at {FINETUNE_BEST}")
    ckpt = torch.load(FINETUNE_BEST, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"checkpoint from epoch {ckpt.get('epoch','?')}, "
          f"reported all-slots {ckpt.get('sample_acc', float('nan')):.3f}\n")

    lex = Lexicon(tokenizer)
    axes = {
        "control  (seen values,  seen phrasing)": ("train", TRAIN_P, 11),
        "values   (NEW values,   seen phrasing)": ("val", TRAIN_P, 22),
        "phrasing (seen values,  NEW phrasing) ": ("train", VAL_P, 33),
        "both     (NEW values,   NEW phrasing) ": ("val", VAL_P, 44),
    }

    print(f"{'axis':40s} {'slot':>6} {'allslots':>9} {'exact':>7}")
    print("-" * 66)
    results = {}
    for label, (split, descs, seed) in axes.items():
        samples = build(lex, split, descs, N_PER_AXIS, seed)
        m = evaluate_slots(model, tokenizer, samples, device,
                           max_seq_len=config["max_seq_len"])
        results[label] = (m, samples)
        print(f"{label:40s} {m['slot_acc']:6.3f} {m['sample_acc']:9.3f} {m['exact_match']:7.3f}")

    ctrl = results["control  (seen values,  seen phrasing)"][0]["sample_acc"]
    val_only = results["values   (NEW values,   seen phrasing)"][0]["sample_acc"]
    phr_only = results["phrasing (seen values,  NEW phrasing) "][0]["sample_acc"]
    print("-" * 66)
    print(f"cost of unseen VALUES   : {ctrl - val_only:+.3f}")
    print(f"cost of unseen PHRASING : {ctrl - phr_only:+.3f}")
    print("\nThe larger drop is your bottleneck. Unseen values -> the copy circuit")
    print("needs more training. Unseen phrasing -> you need more paraphrases per")
    print("template, and more steps will not help.")

    # Per-template breakdown on the phrasing axis: which shapes fail?
    print("\nper-template all-slots accuracy (NEW phrasing axis):")
    m, samples = results["phrasing (seen values,  NEW phrasing) "]
    by = defaultdict(lambda: [0, 0])
    for s in samples:
        gen = greedy_generate(model, tokenizer, s["prompt"], device,
                              max_seq_len=config["max_seq_len"])
        ok = all(_slot_hit(gen, s, k, v) for k, v in s["slots"].items())
        by[s["template"]][0] += int(ok)
        by[s["template"]][1] += 1
    for name in sorted(by, key=lambda k: by[k][0] / max(1, by[k][1])):
        good, total = by[name]
        print(f"  {name:20s} {good/max(1,total):5.2f}  (n={total})")

    # Does the model at least pick the right diagram shape?
    print("\ntemplate-shape match (ignoring slot contents), NEW phrasing:")
    shape_ok = 0
    for s in samples[:100]:
        gen = greedy_generate(model, tokenizer, s["prompt"], device,
                              max_seq_len=config["max_seq_len"])
        want = _normalize(s["mermaid"])
        # compare structural skeletons with slot text stripped out
        skel_want = want
        skel_got = _normalize(gen)
        for v in s["slots"].values():
            skel_want = skel_want.replace(v, "X")
            skel_got = skel_got.replace(v, "X")
        shape_ok += int(skel_want == skel_got)
    print(f"  {shape_ok/100:.2f} of samples had the CORRECT DIAGRAM SHAPE")
    print("  (low here + high slot accuracy = template selection is the problem,")
    print("   not copying)")


if __name__ == "__main__":
    main()
