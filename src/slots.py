"""Tier-2 lexicon: simple everyday English, filtered by what the tokenizer
already handles cheaply.

The earlier design picked a vocabulary and then trained the tokenizer on it,
which made those 150 words privileged single tokens while ordinary English
fragmented into 6-8 pieces. The model learned to copy 2-token spans and fell
apart on anything real ("Scan Barcode" -> 8 tokens).

This version inverts that. A large candidate list is filtered at runtime against
whatever tokenizer you have: words costing more than `max_tokens` are dropped.
Nothing is privileged, no tokenizer retraining is needed, and the lexicon
adapts automatically if the tokenizer is ever rebuilt.

Categories:
  act   -- "Verb Noun" labels, e.g. "Read Card"     (~1500 values)
  cond  -- conditions, e.g. "Even" or "Door Open"   (~1400 values)
  label -- bare structural labels: A-E, Yes, No     (closed set, no holdout)
"""
import random

CAND_ACTIONS = [
    "Add", "Ask", "Call", "Check", "Count", "Cut", "Enter", "Fill", "Find",
    "Fix", "Hide", "Jump", "Move", "Open", "Pick", "Play", "Print", "Pull",
    "Push", "Read", "Save", "Send", "Show", "Start", "Stop", "Take", "Try",
    "Wait", "Give", "Get", "Put", "Set", "Make", "Keep", "Let", "Run",
    "Walk", "Look", "See", "Tell", "Say", "Help", "Hold", "Bring", "Buy",
    "Sell", "Eat", "Drink", "Sing", "Dance", "Sleep", "Wake", "Wash",
    "Cook", "Feed", "Water", "Plant", "Build", "Break", "Throw", "Catch",
    "Kick", "Ride", "Drive", "Fly", "Swim", "Climb", "Dig", "Paint",
    "Write", "Sign", "Mark", "Pack", "Lift", "Drop", "Lock", "Ring",
]

CAND_OBJECTS = [
    "Bag", "Ball", "Book", "Box", "Cake", "Card", "Coin", "Door", "Game",
    "Key", "Letter", "Light", "List", "Map", "Name", "Note", "Number",
    "Page", "Song", "Toy", "Word", "Bell", "Bird", "Boat", "Cat", "Chair",
    "Cup", "Dog", "Egg", "Fish", "Flower", "Hat", "Hand", "Home", "Horse",
    "House", "Ice", "Jar", "Lamp", "Leaf", "Line", "Lunch", "Milk", "Moon",
    "Nest", "Path", "Pen", "Pie", "Plate", "Pot", "Rock", "Room", "Rope",
    "Seed", "Ship", "Shoe", "Sign", "Star", "Stone", "Sun", "Table",
    "Tree", "Wall", "Wheel", "Window", "Bread", "Bus", "Train", "Clock",
]

CAND_CONDITIONS = [
    "Even", "Odd", "Big", "Full", "Ready", "Done", "Same", "New", "Old",
    "True", "Right", "Hot", "Cold", "Fast", "Slow", "Happy", "Sad",
    "Locked", "Open", "Safe", "Late", "Early", "Clear", "Dark", "Light",
    "Loud", "Quiet", "Dry", "Wet", "Soft", "Hard", "Free", "Busy", "Lost",
    "Found", "Good", "Bad", "Long", "Short", "Deep", "High", "Low",
]

CAND_LABELS = ["A", "B", "C", "D", "E", "Yes", "No"]

VAL_VALUE_FRACTION = 0.15
_SPLIT_SEED = 1234

# Structural rather than content. A flowchart only has so many node letters, so
# these are legitimately closed and shared between train and val.
CLOSED_CATEGORIES = {"label"}


def category_of(slot_name):
    """Templates name slots act_1, cond_2, label_1 -> category is the prefix."""
    return slot_name.rsplit("_", 1)[0] if "_" in slot_name else slot_name


class Lexicon:
    """Tokenizer-filtered vocabulary with a deterministic train/val value split."""

    def __init__(self, tokenizer, max_tokens=2, val_fraction=VAL_VALUE_FRACTION):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

        self.actions = self._filter(CAND_ACTIONS)
        self.objects = self._filter(CAND_OBJECTS)
        self.conditions = self._filter(CAND_CONDITIONS)
        self.labels = self._filter(CAND_LABELS)

        if min(len(self.actions), len(self.objects), len(self.conditions)) < 8:
            raise ValueError(
                "Too few cheap words survived filtering. Raise max_tokens to 3, "
                "or check that the tokenizer loaded correctly."
            )

        self.pools = {
            "act": sorted(f"{a} {o}" for a in self.actions for o in self.objects),
            "cond": sorted(
                set(self.conditions)
                | {f"{o} {c}" for o in self.objects for c in self.conditions}
            ),
            "label": sorted(self.labels),
        }

        self.train_pools, self.val_pools = {}, {}
        for name, vals in self.pools.items():
            if name in CLOSED_CATEGORIES:
                self.train_pools[name] = self.val_pools[name] = vals
                continue
            shuffled = list(vals)
            random.Random(_SPLIT_SEED).shuffle(shuffled)
            n_val = max(1, int(len(shuffled) * val_fraction))
            self.val_pools[name] = shuffled[:n_val]
            self.train_pools[name] = shuffled[n_val:]

        self._fragments = self._build_fragments()

    def _filter(self, words):
        keep = [w for w in words
                if len(self.tokenizer.encode(" " + w).ids) <= self.max_tokens]
        return sorted(set(keep))

    def _build_fragments(self, min_len=2, max_len=9):
        frags = set()
        for tok in self.tokenizer.get_vocab():
            piece = tok.replace("Ġ", "")
            if piece.isalpha() and min_len <= len(piece) <= max_len:
                frags.add(piece.lower())
        return sorted(frags)

    def values(self, slot_name, split):
        pools = self.train_pools if split == "train" else self.val_pools
        return pools[category_of(slot_name)]

    def sample_slots(self, slot_names, split, rng):
        chosen, used = {}, set()
        for name in slot_names:
            pool = self.values(name, split)
            v = rng.choice(pool)
            if category_of(name) not in CLOSED_CATEGORIES:
                for _ in range(30):
                    if v not in used:
                        break
                    v = rng.choice(pool)
            used.add(v)
            chosen[name] = v
        return chosen

    def random_value(self, rng):
        """An arbitrary multi-token span -- no lexicon to memorise.

        Kept at a low mix ratio for Tier 2: enough to stop the model relying
        purely on lexicon retrieval, not so much that training is mostly noise.
        """
        parts = []
        for _ in range(rng.choice([1, 2, 2])):
            n_pieces = rng.choice([1, 2, 2, 3])
            w = "".join(rng.choice(self._fragments) for _ in range(n_pieces))[:14]
            parts.append(w.capitalize())
        return " ".join(parts)

    def random_slots(self, slot_names, rng):
        chosen = {}
        for name in slot_names:
            if category_of(name) in CLOSED_CATEGORIES:
                chosen[name] = rng.choice(self.pools["label"])
            else:
                chosen[name] = self.random_value(rng)
        return chosen

    def report(self):
        lines = [
            f"Lexicon (<= {self.max_tokens} tokens per word):",
            f"  actions {len(self.actions):3d} | objects {len(self.objects):3d} "
            f"| conditions {len(self.conditions):3d} | labels {len(self.labels):2d}",
        ]
        for name, vals in self.pools.items():
            tr, va = len(self.train_pools[name]), len(self.val_pools[name])
            closed = "  (closed set, shared)" if name in CLOSED_CATEGORIES else ""
            lines.append(f"  pool '{name}': {len(vals):5d} values -> {tr} train / {va} val{closed}")
        lines.append(f"  vocab fragments for novel spans: {len(self._fragments)}")
        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tokenizer import load_tokenizer

    lex = Lexicon(load_tokenizer())
    print(lex.report())
    rng = random.Random(0)
    print("\nsample act values: ", [rng.choice(lex.pools["act"]) for _ in range(5)])
    print("sample cond values:", [rng.choice(lex.pools["cond"]) for _ in range(5)])
    print("sample novel spans:", [lex.random_value(rng) for _ in range(4)])
