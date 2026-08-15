"""Instruction-tuning dataset: natural-language description -> Mermaid diagram.

Changes from the original:
  * Slot values come from slots.py (hundreds per slot, value-level train/val
    holdout) instead of 10 hand-written strings per slot.
  * Each sample records its slot values, so we can measure whether the model
    actually copied them (see metrics.py).
  * Sequences are padded per-batch by a collate_fn instead of padded to 512 in
    __getitem__. The old version wasted ~90% of every forward pass on padding,
    which is why an epoch over 4000 tiny samples took 775 seconds.
"""
import json
import random
import re
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from paths import MERMAID_TRAIN_JSON, MERMAID_VAL_JSON, PROCESSED_DIR
from slots import Lexicon
from templates import TEMPLATES, paraphrase_split

_SLOT_PATTERN = re.compile(r"\{(\w+)\}")


def _template_slot_names(template):
    # Content slots are lowercase (act_1, cond_2, label_1); the paraphrase
    # grammar's connective placeholders are uppercase and already substituted.
    names = {n for n in _SLOT_PATTERN.findall(template["mermaid"]) if n.islower()}
    return sorted(names)


def _sample_slots(template, split, rng, lexicon):
    return lexicon.sample_slots(_template_slot_names(template), split, rng)


def _render(template, slot_vals, paraphrase):
    return {
        "template": template["name"],
        "prompt": paraphrase.format(**slot_vals),
        "mermaid": template["mermaid"].format(**slot_vals),
        "slots": slot_vals,
    }


def generate_mermaid_dataset_split(
    num_train_samples=40000,
    num_val_samples=1000,
    holdout_paraphrase_for_val=True,
    tokenizer=None,
    output_train_file=None,
    output_val_file=None,
    seed=42,
):
    """Build the fine-tuning pairs.

    Validation is held out along two independent axes:
      1. slot VALUES the model never saw in training (slots.py)
      2. the final paraphrase of each template, unseen in training

    So a model that has memorised anything at all will score badly, and only a
    model that reads the prompt and copies will score well.
    """
    output_train_file = Path(output_train_file or MERMAID_TRAIN_JSON)
    output_val_file = Path(output_val_file or MERMAID_VAL_JSON)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # paraphrase pools, split per template by templates.paraphrase_split
    para = {t["name"]: paraphrase_split(t) for t in TEMPLATES}
    train_paras = {k: v[0] for k, v in para.items()}
    val_paras = {k: (v[1] if holdout_paraphrase_for_val else v[0] + v[1])
                 for k, v in para.items()}

    if tokenizer is None:
        raise ValueError("generate_mermaid_dataset_split now needs a tokenizer "
                         "(the lexicon is filtered by token cost).")
    lexicon = Lexicon(tokenizer)
    print(lexicon.report())

    def _sample(split, para_pool, num_samples, novel_ratio=0.0):
        samples, seen, attempts = [], set(), 0
        max_attempts = num_samples * 20
        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            template = rng.choice(TEMPLATES)
            novel = rng.random() < novel_ratio
            if novel:
                slot_vals = lexicon.random_slots(_template_slot_names(template), rng)
            else:
                slot_vals = _sample_slots(template, split, rng, lexicon)
            rendered = _render(template, slot_vals,
                               rng.choice(para_pool[template["name"]]))
            rendered["novel"] = novel
            key = (rendered["prompt"], rendered["mermaid"])
            if key in seen:
                continue
            seen.add(key)
            samples.append(rendered)
        return samples

    train_samples = _sample("train", train_paras, num_train_samples)
    val_samples = _sample("val", val_paras, num_val_samples, novel_ratio=0.5)

    with open(output_train_file, "w") as f:
        json.dump(train_samples, f, indent=2)
    with open(output_val_file, "w") as f:
        json.dump(val_samples, f, indent=2)

    print(f"Train pairs: {len(train_samples):,} -> {output_train_file}")
    print(f"Val pairs:   {len(val_samples):,} -> {output_val_file}")
    n_novel = sum(1 for s_ in val_samples if s_.get("novel"))
    print(f"Val: unseen slot values + unseen paraphrase; {n_novel:,} of "
          f"{len(val_samples):,} use arbitrary vocab spans (never-seen words).")
    return train_samples, val_samples


class _EncoderMixin:
    """Shared prompt/diagram -> (x, y) encoding."""

    def _setup_ids(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mermaid_id = tokenizer.token_to_id("<|mermaid|>")
        self.prompt_id = tokenizer.token_to_id("<|prompt|>")
        self.end_id = tokenizer.token_to_id("<|endoftext|>")
        self.pad_id = tokenizer.token_to_id("<|pad|>")

    def _encode(self, sample):
        p_ids = self.tokenizer.encode(sample["prompt"]).ids
        m_ids = self.tokenizer.encode(sample["mermaid"]).ids

        full_ids = [self.mermaid_id] + p_ids + [self.prompt_id] + m_ids + [self.end_id]
        full_ids = full_ids[: self.max_seq_len]

        x = torch.tensor(full_ids[:-1], dtype=torch.long)
        y = torch.tensor(full_ids[1:], dtype=torch.long)

        # Loss is computed on the diagram only; the prompt is context.
        prompt_len = 1 + len(p_ids) + 1
        y[: prompt_len - 1] = -1
        return x, y


class InstructionMermaidDataset(Dataset, _EncoderMixin):
    """Fixed list of pairs. Used for validation, where we want a stable set."""

    def __init__(self, samples, tokenizer, max_seq_len=512):
        self.samples = samples
        self._setup_ids(tokenizer, max_seq_len)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._encode(self.samples[idx])


class OnTheFlyMermaidDataset(Dataset, _EncoderMixin):
    """Samples a FRESH pair on every access -- effectively infinite training data.

    Why this exists: the first fine-tuning run used a fixed set of 20,000
    pre-generated pairs. With 23M parameters against ~1M training tokens the
    model memorised that set outright -- train loss hit 0.0028 by epoch 4 and
    0.0000 by epoch 5, meaning 14 of 17 epochs ran with essentially no gradient
    signal. Slot accuracy stalled at 0.5-0.66 not because the model could do no
    better but because training had effectively stopped.

    Drawing a new combination each step makes memorisation impossible: with
    229M combinations per template the model will essentially never see the same
    pair twice, so the ONLY way to drive the loss down is to actually read the
    prompt and copy. Train loss should now settle at a low but non-zero value
    and keep producing gradient for the whole run.

    `samples_per_epoch` just defines how often we stop to evaluate; it is not a
    dataset size.
    """

    def __init__(self, tokenizer, samples_per_epoch, max_seq_len=512,
                 split="train", paraphrases=None, seed=0, novel_ratio=0.5):
        self.samples_per_epoch = samples_per_epoch
        self.split = split
        self.paraphrases = paraphrases or {
            t["name"]: paraphrase_split(t)[0] for t in TEMPLATES
        }
        self._seed = seed
        self._rng = None
        # Fraction of samples whose slots are arbitrary vocab spans rather than
        # pool values. At 0.0 the model can get by with lexicon retrieval; at
        # 0.5 it must learn to copy spans it has never seen.
        self.novel_ratio = novel_ratio
        self.lexicon = Lexicon(tokenizer)
        self._setup_ids(tokenizer, max_seq_len)

    def _get_rng(self):
        # Lazily seed per DataLoader worker so workers don't emit identical draws.
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            self._rng = random.Random(f"{self._seed}-{wid}-{id(self)}")
        return self._rng

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        rng = self._get_rng()
        template = rng.choice(TEMPLATES)
        names = _template_slot_names(template)
        if rng.random() < self.novel_ratio:
            slot_vals = self.lexicon.random_slots(names, rng)
        else:
            slot_vals = self.lexicon.sample_slots(names, self.split, rng)
        sample = _render(template, slot_vals,
                         rng.choice(self.paraphrases[template["name"]]))
        return self._encode(sample)


def collate_dynamic(batch, pad_id):
    """Pad to the longest sequence in THIS batch, not to max_seq_len.

    Real sequences here are ~45-75 tokens. Padding every one to 512 meant ~88%
    of each forward pass was spent on padding tokens contributing nothing.
    """
    max_len = max(x.size(0) for x, _ in batch)
    xs = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    ys = torch.full((len(batch), max_len), -1, dtype=torch.long)  # -1 = ignore
    for i, (x, y) in enumerate(batch):
        xs[i, : x.size(0)] = x
        ys[i, : y.size(0)] = y
    return xs, ys


def get_mermaid_dataloader(
    tokenizer,
    batch_size=32,
    max_seq_len=512,
    train_file=None,
    val_file=None,
    num_workers=0,
    on_the_fly=True,
    samples_per_epoch=20000,
    novel_ratio=0.2,
):
    """Training data is generated on the fly by default; validation stays fixed.

    Pass on_the_fly=False to fall back to the pre-generated JSON (useful only
    for reproducing the old memorising behaviour).
    """
    val_file = Path(val_file or MERMAID_VAL_JSON)
    if not val_file.exists():
        generate_mermaid_dataset_split(tokenizer=tokenizer)
    with open(val_file) as f:
        val_samples = json.load(f)

    if on_the_fly:
        train_ds = OnTheFlyMermaidDataset(
            tokenizer, samples_per_epoch, max_seq_len, split="train",
            novel_ratio=novel_ratio,
        )
    else:
        train_file = Path(train_file or MERMAID_TRAIN_JSON)
        if not train_file.exists():
            generate_mermaid_dataset_split(tokenizer=tokenizer)
        with open(train_file) as f:
            train_samples = json.load(f)
        train_ds = InstructionMermaidDataset(train_samples, tokenizer, max_seq_len)

    val_ds = InstructionMermaidDataset(val_samples, tokenizer, max_seq_len)
    collate = partial(collate_dynamic, pad_id=val_ds.pad_id)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=not on_the_fly, pin_memory=True,
        collate_fn=collate, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=True,
        collate_fn=collate, num_workers=num_workers,
    )
    return train_loader, val_loader, val_samples


def _unused_iter_tokenizer_corpus(num_pairs=40000, seed=7):
    """Text for tokenizer training: every slot value plus many rendered pairs.

    Includes BOTH train and val slot values. That is intentional and is not
    leakage -- the tokenizer only learns how to segment strings, not which
    diagram goes with which prompt. If val values tokenized worse than train
    values, val loss would be measuring tokenizer coverage instead of copying.
    """
    from slots import all_values

    values = all_values()
    for _ in range(5):                       # repeat so merges are learned
        for chunk_start in range(0, len(values), 64):
            yield " ".join(values[chunk_start:chunk_start + 64])

    rng = random.Random(seed)
    for _ in range(num_pairs):
        template = rng.choice(TEMPLATES)
        split = "train" if rng.random() < 0.85 else "val"
        slot_vals = _sample_slots(template, split, rng)
        rendered = _render(template, slot_vals,
                           rng.choice(paraphrase_split(template)[0]))
        yield rendered["prompt"] + "\n" + rendered["mermaid"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate the Mermaid instruction pairs.")
    parser.add_argument("--train-samples", type=int, default=40000,
                        help="Fewer samples = shorter epochs. 12000 is still ample: the "
                             "anti-memorisation property comes from the size of the slot "
                             "space (1858 distinct values, 229M combinations), not from "
                             "the number of samples.")
    parser.add_argument("--val-samples", type=int, default=1000)
    args = parser.parse_args()
    from tokenizer import load_tokenizer
    generate_mermaid_dataset_split(
        num_train_samples=args.train_samples,
        num_val_samples=args.val_samples,
        tokenizer=load_tokenizer(),
    )
