"""Which simple flowchart words does your existing tokenizer handle cheaply?

The point: instead of retraining the tokenizer to fit a vocabulary, pick a
vocabulary that fits the tokenizer you already have. Words costing 1-2 tokens
are cheap to copy; words costing 4+ are the ones that broke on 'Scan Barcode'.

Run:  python pick_vocab.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from tokenizer import load_tokenizer

CANDIDATES = {
    "actions": [
        "Enter", "Read", "Check", "Show", "Print", "Start", "Stop", "Wait",
        "Add", "Count", "Pick", "Open", "Close", "Save", "Send", "Ask",
        "Draw", "Play", "Jump", "Move", "Find", "Try", "Turn", "Push",
        "Pull", "Fill", "Clean", "Cut", "Fix", "Hide", "Call", "Take",
    ],
    "objects": [
        "Number", "Name", "Answer", "Word", "Letter", "Box", "Door", "Light",
        "Game", "Song", "Book", "Card", "Ball", "Cake", "List", "Score",
        "Page", "Key", "Bag", "Toy", "Note", "Gift", "Map", "Coin",
    ],
    "conditions": [
        "Even", "Odd", "Big", "Small", "Empty", "Full", "Open", "Locked",
        "Ready", "Done", "Same", "New", "Old", "True", "False", "Right",
        "Wrong", "Hot", "Cold", "Happy", "Sad", "Fast", "Slow",
    ],
    "labels": ["A", "B", "C", "D", "E", "Step One", "Step Two", "Yes", "No"],
}


def main():
    tok = load_tokenizer()
    cheap, costly = [], []

    for group, words in CANDIDATES.items():
        print(f"\n--- {group} ---")
        for w in sorted(words):
            n = len(tok.encode(" " + w).ids)
            flag = "ok  " if n <= 2 else "COSTLY"
            print(f"  {n:2d} tok  {flag}  {w}")
            (cheap if n <= 2 else costly).append(w)

    total = len(cheap) + len(costly)
    print(f"\n{'='*46}")
    print(f"cheap (1-2 tokens): {len(cheap):3d} / {total}  ({100*len(cheap)/total:.0f}%)")
    print(f"costly (3+ tokens): {len(costly):3d} / {total}")
    if costly:
        print(f"\ndrop these from the lexicon: {', '.join(sorted(costly))}")

    # Two-word labels built from the cheap words are what slots would look like
    if len(cheap) >= 4:
        pairs = [f"{cheap[i]} {cheap[-i-1]}" for i in range(min(6, len(cheap) // 2))]
        print("\nexample two-word labels from the cheap set:")
        for p in pairs:
            print(f"  {len(tok.encode(' ' + p).ids):2d} tok  {p}")

    counts = Counter(len(tok.encode(" " + w).ids) for g in CANDIDATES.values() for w in g)
    print("\ntoken-cost histogram:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
