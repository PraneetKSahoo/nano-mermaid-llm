import itertools
import random
import re

_WORD_SLOT = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")

TEMPLATES = [
    {
        "name": "linear2",
        "mermaid": "graph TD\n    A[{act_1}] --> B[{act_2}]",
        "frames": [
            "{OPEN} {act_1}{SEP} {THEN} {act_2}.",
            "{act_1} {LEADS} {act_2}.",
            "{DO} {act_1}{SEP} {THEN} {act_2}.",
            "Two steps{COLON} {act_1}{SEP} {THEN} {act_2}.",
        ],
        "words": {
            "OPEN": ["First", "Start with", "Begin with", "You start with", "Step one is"],
            "THEN": ["then", "next", "after that", "followed by", "then do"],
            "LEADS": ["leads to", "goes to", "comes before", "is followed by", "points to"],
            "DO": ["Do", "Run", "Perform", "Carry out", "Handle"],
            "SEP": [","],
            "COLON": [":", " are"],
        },
    },
    {
        "name": "linear3",
        "mermaid": "graph TD\n    A[{act_1}] --> B[{act_2}]\n    B --> C[{act_3}]",
        "frames": [
            "{OPEN} {act_1}{SEP} {THEN} {act_2}{SEP} {FIN} {act_3}.",
            "{DO} {act_1}{SEP} {THEN} {act_2}{SEP} {THEN} {act_3}.",
            "Three steps{COLON} {act_1}, {act_2}, {act_3}.",
            "The order is {act_1}, {act_2}, {THEN} {act_3}.",
            "{act_1} {LEADS} {act_2} which {LEADS} {act_3}.",
        ],
        "words": {
            "OPEN": ["First", "Start with", "Begin with", "You start with", "Step one is"],
            "THEN": ["then", "next", "after that", "followed by"],
            "FIN": ["and finally", "and last", "ending with", "and then"],
            "DO": ["Do", "Run", "Perform", "Carry out"],
            "LEADS": ["leads to", "goes to", "moves to"],
            "SEP": [","],
            "COLON": [":", " are"],
        },
    },
    {
        "name": "linear4",
        "mermaid": ("graph TD\n    A[{act_1}] --> B[{act_2}]\n"
                    "    B --> C[{act_3}]\n    C --> D[{act_4}]"),
        "frames": [
            "{OPEN} {act_1}{SEP} {THEN} {act_2}{SEP} {THEN} {act_3}{SEP} {FIN} {act_4}.",
            "Four steps{COLON} {act_1}, {act_2}, {act_3}, {act_4}.",
            "{DO} {act_1}{SEP} {THEN} {act_2}, {act_3}, {FIN} {act_4}.",
            "The order is {act_1}, {act_2}, {act_3}, {THEN} {act_4}.",
        ],
        "words": {
            "OPEN": ["First", "Start with", "Begin with", "You start with"],
            "THEN": ["then", "next", "after that"],
            "FIN": ["and finally", "and last", "ending with"],
            "DO": ["Do", "Run", "Perform", "Carry out"],
            "SEP": [","],
            "COLON": [":", " are"],
        },
    },
    {
        "name": "decision",
        "mermaid": ("graph TD\n    A{{{cond_1}}} -- Yes --> B[{act_1}]\n"
                    "    A -- No --> C[{act_2}]"),
        "frames": [
            "{CHECK} {cond_1}. {IFYES} {act_1}{SEP} {IFNO} {act_2}.",
            "Is it {cond_1}? Yes {MEANS} {act_1}, no {MEANS} {act_2}.",
            "If {cond_1} then {act_1} else {act_2}.",
            "{CHECK} {cond_1}, going to {act_1} or {act_2}.",
            "When {cond_1}, {DO} {act_1}, otherwise {act_2}.",
        ],
        "words": {
            "CHECK": ["Check", "Test", "Look at", "Verify", "Evaluate", "Check if",
                      "Decide on", "Branch on", "Ask whether"],
            "IFYES": ["If yes,", "On yes,", "If so,", "True gives", "Yes leads to"],
            "IFNO": ["if no,", "on no,", "if not,", "false gives", "no leads to"],
            "MEANS": ["means", "gives", "leads to"],
            "DO": ["do", "run", "perform"],
            "SEP": [".", ","],
        },
    },
    {
        "name": "step_then_decision",
        "mermaid": ("graph TD\n    A[{act_1}] --> B{{{cond_1}}}\n"
                    "    B -- Yes --> C[{act_2}]\n    B -- No --> D[{act_3}]"),
        "frames": [
            "{DO} {act_1}. {THEN} {CHECK} {cond_1}. {IFYES} {act_2}, {IFNO} {act_3}.",
            "{OPEN} {act_1}. Is it {cond_1}? Yes {act_2}, no {act_3}.",
            "After {act_1}, if {cond_1} then {act_2} else {act_3}.",
            "{DO} {act_1} and {CHECK} {cond_1}, going to {act_2} or {act_3}.",
        ],
        "words": {
            "DO": ["Do", "Run", "Perform", "Start with", "Carry out"],
            "OPEN": ["First", "Begin with", "Start with"],
            "THEN": ["Then", "Next", "After that"],
            "CHECK": ["check", "test", "look at", "verify", "evaluate", "decide on"],
            "IFYES": ["if yes", "on yes", "if so", "yes gives"],
            "IFNO": ["if no", "on no", "if not", "no gives"],
        },
    },
    {
        "name": "goto_labels",
        "mermaid": ("graph TD\n    A[{act_1}] --> B{{{cond_1}}}\n"
                    "    B -- Yes --> C[{label_1}]\n    B -- No --> D[{label_2}]"),
        "frames": [
            "{DO} {act_1}. If {cond_1} {GO} {label_1}, else {GO} {label_2}.",
            "{DO} {act_1}, then {CHECK} {cond_1}: yes {GO} {label_1}, no {GO} {label_2}.",
            "{OPEN} {act_1}. When {cond_1} pick {label_1}, if not pick {label_2}.",
            "After {act_1}, {cond_1} sends you to {label_1} and not to {label_2}.",
        ],
        "words": {
            "DO": ["Do", "Run", "Perform", "Carry out", "Start with", "Begin by"],
            "OPEN": ["First", "Start at", "Begin with", "Open with"],
            "GO": ["go to", "jump to", "move to", "head to", "branch to", "continue to"],
            "CHECK": ["check", "test", "evaluate", "look at", "decide on"],
        },
    },
    {
        "name": "three_way",
        "mermaid": ("graph TD\n    A{{{act_1}}} -- {cond_1} --> B[{act_2}]\n"
                    "    A -- {cond_2} --> C[{act_3}]\n    A -- {cond_3} --> D[{act_4}]"),
        "frames": [
            "{CHECK} {act_1}. If {cond_1} {DO} {act_2}, if {cond_2} {DO} {act_3}, if {cond_3} {DO} {act_4}.",
            "{act_1} has three cases{COLON} {cond_1} gives {act_2}, {cond_2} gives {act_3}, {cond_3} gives {act_4}.",
            "{CHECK} {act_1}: {cond_1} {TO} {act_2}, {cond_2} {TO} {act_3}, {cond_3} {TO} {act_4}.",
            "Three outcomes from {act_1}{COLON} {cond_1} {TO} {act_2}, {cond_2} {TO} {act_3}, {cond_3} {TO} {act_4}.",
            "{act_1} splits on {cond_1}, {cond_2} and {cond_3} into {act_2}, {act_3} and {act_4}.",
        ],
        "words": {
            "CHECK": ["Check", "Test", "Look at", "Evaluate", "Decide", "Branch",
                      "Inspect", "Examine"],
            "DO": ["do", "run", "pick", "choose", "go to", "use"],
            "TO": ["to", "leads to", "gives", "means", "goes to", "picks"],
            "COLON": [":", " -"],
        },
    },
    {
        "name": "loop",
        "mermaid": ("graph TD\n    A[{act_1}] --> B{{{cond_1}}}\n"
                    "    B -- No --> A\n    B -- Yes --> C[{act_2}]"),
        "frames": [
            "{REPEAT} {act_1} {UNTIL} {cond_1}, then {DO} {act_2}.",
            "{REPEAT} {act_1} while not {cond_1}. When {cond_1}, {DO} {act_2}.",
            "{act_1} {LOOPS} {UNTIL} {cond_1} is true, then {DO} {act_2}.",
            "{DO} {act_1}. If not {cond_1} try again, otherwise {act_2}.",
            "While not {cond_1}, keep {act_1}. Once {cond_1}, {act_2}.",
        ],
        "words": {
            "REPEAT": ["Do", "Repeat", "Keep doing", "Loop on", "Cycle through", "Try",
                       "Run", "Go over"],
            "LOOPS": ["loops back", "repeats", "runs again", "cycles", "goes round again",
                      "retries"],
            "DO": ["do", "run", "go to", "finish with", "move to", "end with"],
            "UNTIL": ["until", "till", "up until"],
        },
    },
    {
        "name": "fork",
        # Every frame carries an explicit parallelism cue. "X then Y and Z" with
        # no cue is genuinely ambiguous with linear3.
        "mermaid": "graph TD\n    A[{act_1}] --> B[{act_2}]\n    A --> C[{act_3}]",
        "frames": [
            "{act_1} {SPLITS} {act_2} {AND} {act_3}.",
            "After {act_1}, {DOBOTH} {act_2} {AND} {act_3}.",
            "{act_1} {FEEDS} both {act_2} {AND} {act_3}.",
            "From {act_1} there are two paths, {act_2} {AND} {act_3}.",
            "{DO} {act_1}, then run {act_2} {AND} {act_3} {TOGETHER}.",
            "One step {act_1} becomes two{COLON} {act_2} {AND} {act_3}.",
            "{DO} {act_1} first, then {act_2} {AND} {act_3} {TOGETHER}.",
            "{act_1} {FEEDS} two paths{COLON} {act_2} {AND} {act_3}.",
        ],
        "words": {
            "SPLITS": ["splits into", "branches into", "divides into", "forks into",
                       "sends work to", "opens up into", "fans out into"],
            "FEEDS": ["feeds", "drives", "triggers", "starts", "kicks off"],
            "DOBOTH": ["do both", "run both", "start both", "handle both"],
            "DO": ["Do", "Run", "Perform", "Carry out"],
            "AND": ["and", "as well as", "plus"],
            "TOGETHER": ["at the same time", "in parallel", "side by side", "together"],
            "COLON": [":", " -"],
        },
    },
    {
        "name": "merge",
        "mermaid": "graph TD\n    A[{act_1}] --> C[{act_3}]\n    B[{act_2}] --> C",
        "frames": [
            "Both {act_1} {AND} {act_2} {LEAD} {act_3}.",
            "{act_1} {AND} {act_2} {JOIN} {act_3}.",
            "After {EITHER} {act_1} or {act_2}, {DO} {act_3}.",
            "Whether you {act_1} or {act_2}, you {FINISH} {act_3}.",
            "{act_3} follows both {act_1} {AND} {act_2}.",
            "Two paths {act_1} {AND} {act_2} both {LEAD} {act_3}.",
            "{EITHER} {act_1} or {act_2} {LEAD} {act_3}.",
            "{act_1} {AND} {act_2} both {LEAD} {act_3}.",
        ],
        "words": {
            "LEAD": ["lead to", "go to", "feed", "end at", "point to", "arrive at",
                     "run into", "finish at"],
            "JOIN": ["join at", "come together at", "merge into", "meet at",
                     "combine at", "collapse into"],
            "EITHER": ["either", "one of", "any of"],
            "AND": ["and", "as well as", "plus"],
            "DO": ["do", "run", "perform"],
            "FINISH": ["finish with", "end with", "close with"],
        },
    },
]

VAL_PARAPHRASE_FRACTION = 0.15
_PARAPHRASE_SEED = 99
_CACHE = {}


def enumerate_paraphrases(template):
    """All surface forms from this template's frames x connective words."""
    name = template["name"]
    if name in _CACHE:
        return _CACHE[name]

    out = []
    for frame in template["frames"]:
        keys = sorted(set(_WORD_SLOT.findall(frame)))
        if not keys:
            out.append(frame)
            continue
        option_lists = [template["words"][k] for k in keys]
        for combo in itertools.product(*option_lists):
            s = frame
            for k, v in zip(keys, combo):
                s = s.replace("{" + k + "}", v)
            out.append(s)

    # Collapse artefacts from optional punctuation ("," + " " etc.)
    cleaned = []
    for s in out:
        s = re.sub(r"\s+", " ", s).replace(" ,", ",").replace(" .", ".")
        s = s.replace(",.", ".").replace("..", ".").replace(",,", ",")
        cleaned.append(s.strip())

    result = sorted(set(cleaned))
    _CACHE[name] = result
    return result


def paraphrase_split(template):
    
    all_p = enumerate_paraphrases(template)
    shuffled = list(all_p)
    random.Random(_PARAPHRASE_SEED).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * VAL_PARAPHRASE_FRACTION))
    return shuffled[n_val:], shuffled[:n_val]


def report():
    lines = ["paraphrases per template (grammar-generated):"]
    tot_tr = tot_va = 0
    for t in TEMPLATES:
        tr, va = paraphrase_split(t)
        tot_tr += len(tr)
        tot_va += len(va)
        lines.append(f"  {t['name']:20s} {len(tr) + len(va):5d} total -> {len(tr):5d} train / {len(va):4d} val")
    lines.append(f"  {'TOTAL':20s} {tot_tr + tot_va:5d} total -> {tot_tr:5d} train / {tot_va:4d} val")
    lines.append("  (previous version: 120 total, 100 train / 20 val)")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
    print()
    for t in TEMPLATES[:2]:
        tr, _ = paraphrase_split(t)
        print(f"{t['name']} samples:")
        for s in tr[:4]:
            print("   ", s)
        print()
