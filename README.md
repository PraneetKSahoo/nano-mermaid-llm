# NanoMermaid LLM

A 23M-parameter GPT built from scratch — custom BPE tokenizer, pretrained on TinyStories, then fine-tuned to turn plain-English workflow descriptions into rendered [Mermaid.js](https://mermaid.js.org/) diagrams.

No pretrained weights. No Hugging Face `AutoModel`. Tokenizer, architecture, training loop, evaluation and inference are all in this repo, and the whole thing was pretrained on a 4GB laptop GPU.

```
Input   →  "Do Log Name. If Word Wrong then Lock Door else Open Door."

Output  →  graph TD
               A[Log Name] --> B{Word Wrong}
               B -- Yes --> C[Lock Door]
               B -- No --> D[Open Door]
```

<!-- TODO: replace with a screenshot of app.py -->
![NanoMermaid web UI](assets/demo.png)

---

## Results

Measured on a held-out validation set that varies **both** slot values and sentence phrasings — nothing in it was seen during training.

| metric | score |
|---|---|
| **Exact diagram match** | **0.984** |
| All slots copied correctly | 0.984 |
| — on in-vocabulary labels | 0.995 |
| — on never-seen arbitrary spans | 0.974 |
| Per-slot accuracy | 0.996 |
| Validation loss | 0.0086 |

The `0.974` is the interesting one: those are slot values like `Inallyarchivfi Neass`, random multi-token strings assembled from vocabulary fragments. The model copies them correctly, which means it learned a genuine **span-copy circuit** rather than memorising a lexicon.

### Training curves

<!-- TODO: export from W&B (Workspace → chart → ⋯ → Download PNG) -->
| Pretraining | Fine-tuning |
|---|---|
| ![pretrain loss](assets/pretrain_loss.png) | ![finetune metrics](assets/finetune_metrics.png) |

Pretraining converged from 3.64 → **1.887** validation loss over 3,000 steps with no divergence. Fine-tuning drove all-slots accuracy from 0.521 → **0.984** in 24 epochs.

Sample generation quality during pretraining, showing the language model forming:

| step | output |
|---|---|
| 500 | *"Lily's mom came over to her dress and went to see what was."* |
| 1500 | *"Lily saw a big dog with a serious face. 'Mommy, look! It is so funny!'"* |
| 3000 | *"Timmy's mom told him that the dog was not nice."* |

---

## Model

| | |
|---|---|
| Parameters | 23,344,128 |
| Layers / heads / width | 6 / 8 / 512 |
| Context length | 512 tokens |
| Vocabulary | 8,192 (custom byte-level BPE) |
| Architecture | Pre-LN decoder-only transformer, GELU MLP, weight-tied embeddings |
| Attention | PyTorch SDPA, causal |
| Precision | fp16 autocast + GradScaler |

**Pretraining** — 300k TinyStories documents → 63.5M train tokens. 3,000 steps at batch 16 × 2 gradient accumulation × 512 context = 49.2M tokens seen, cosine LR schedule with 300 warmup steps. ~5 hours on a GTX 1650 Ti (4GB).

**Fine-tuning** — 10 diagram shapes × 3,138 grammar-generated paraphrases, with training pairs sampled **on the fly** so no example is ever repeated. 24 epochs × 625 steps. ~33 minutes on a Colab T4.

---

## What it can and cannot do

**It knows 10 diagram shapes:**

| shape | example |
|---|---|
| 2/3/4-step chain | `First Grab Milk, then Heat Pot, and finally Fill Cup.` |
| decision | `Check if Box Full. Yes leads to Save List, on no Ask Name.` |
| step then decision | `Do Log Name. If Word Wrong then Lock Door else Open Door.` |
| goto labels | `Run Enter Number, then check Path Even: yes jump to B, no jump to C.` |
| three-way branch | `Evaluate Check Card. If Card New use Open Box, if Card Old use Hide Box, if Card Lost use Send Note.` |
| loop until | `Keep doing Count Coin until Bag Full, then Stop Game.` |
| fork | `After Play Song, run both Show Card and Send Letter.` |
| merge | `Both Read Book and Write Note lead to Save Page.` |

**Write node labels in Title Case.** The model locates slot values by capitalisation. `Enter Number` gets copied; `enter number` gives it nothing to copy and it will invent labels instead. The web UI warns you when a prompt has no Title-Case labels.

**Honest limitations.** This is a learned template matcher, not a language understander:

- It classifies your sentence into one of 10 shapes, then copies Title-Case spans into the slots. Anything outside those shapes gets forced into the nearest one.
- Free-form English (`"grab the milk, heat it up"`) fails — turning that into `Grab Milk → Heat Milk` requires rewriting, not copying.
- Out-of-domain input produces plausible-looking **invented** labels rather than an error, which is the more dangerous failure mode.

Within its domain it is reliable. Outside it, it fails confidently.

---

## Quickstart

### Run the pretrained model

```bash
git clone https://github.com/<YOUR_USERNAME>/nano-mermaid-llm.git
cd nano-mermaid-llm
pip install -r requirements.txt
```

Download `best_mermaid_model.pt` and `tokenizer.json` from the [Releases](../../releases) page into:

```
checkpoints/finetune_mermaid/best_mermaid_model.pt
data/tokenizer/tokenizer.json
```

Then:

```bash
python src/generate.py "First Grab Milk, then Heat Pot, and finally Fill Cup."
python app.py          # web UI at http://127.0.0.1:7860
```

### Train from scratch

```bash
python src/tokenizer.py                             # ~10 min   BPE on mixed corpus
python src/dataset_mermaid.py --train-samples 20000 # ~10 sec   validation pairs
python src/dataset.py                               # ~15 min   TinyStories → uint16
python src/train_pretrain.py                        # ~5 hrs    resumable
python src/train_finetune.py                        # ~35 min on a T4
```

Both trainers resume automatically; `--restart` backs up the existing checkpoint and starts fresh.

---

## Repo layout

```
nano-mermaid-llm/
├── src/
│   ├── paths.py              single source of truth for every path
│   ├── model.py              GPT architecture (~200 lines)
│   ├── tokenizer.py          byte-level BPE, mixed-corpus training
│   ├── slots.py              tokenizer-filtered vocabulary + value holdout
│   ├── templates.py          10 diagram shapes, paraphrase grammar
│   ├── dataset.py            TinyStories → memory-mapped token binaries
│   ├── dataset_mermaid.py    on-the-fly instruction pairs, dynamic padding
│   ├── metrics.py            slot exact-match (the metric that matters)
│   ├── train_pretrain.py     cosine LR, AMP, resumable
│   ├── train_finetune.py     selects checkpoints on slot accuracy, not loss
│   └── generate.py           constrained-header inference
├── scripts/
│   ├── diagnose.py           separates copying from phrasing generalisation
│   ├── check_tokens.py       tokenization cost probe
│   ├── debug_raw.py          raw model output, pre-sanitisation
│   └── pick_vocab.py         finds words the tokenizer handles cheaply
├── config/model_config.json
├── app.py                    Gradio UI with live Mermaid rendering
└── assets/                   screenshots and training curves
```

---

## Engineering log

The interesting part of this project was not the architecture — it is nanoGPT-shaped and unremarkable. It was the sequence of bugs, each invisible until something specific exposed it.

**1. Positional embeddings initialised 50× too wide.** The weight-init function had the `nn.Embedding` branch nested inside the `nn.Linear` branch, making it unreachable. Token embeddings were saved by weight tying (shared with `lm_head`, a `Linear`), but positional embeddings kept PyTorch's default `N(0,1)` — 50× wider than the token embeddings they are summed with. The model could see *where* it was far more clearly than *what was there*, directly suppressing the signal a copy circuit needs.

**2. A slot vocabulary small enough to memorise.** With 10 values per slot and 23M parameters against 200k training tokens, guessing cost only `ln(10)` nats — cheaper than learning to attend back to the prompt. Train loss hit 0.0000 by epoch 5, so **14 of 17 epochs ran with no gradient signal at all**. Fixed by generating training pairs on the fly from a 229M-combination space.

**3. A tokenizer trained on its own training vocabulary.** The BPE corpus was TinyStories plus 40k renderings of the 150 slot words — so those words earned dedicated single tokens while ordinary English did not. Training taught the model to copy **2-token** spans; real words like `Scan Barcode` cost **8 tokens**. Fixed by inverting the dependency: filter the vocabulary by what the tokenizer already handles cheaply, rather than training the tokenizer on the vocabulary.

**4. A metric that gave away free points.** Slot accuracy used a bare substring check, but Mermaid node IDs are `A`, `B`, `C` and edge labels are `Yes`/`No` — so a label slot with value `"B"` matched *any* diagram ever produced. Fixed by deriving each slot's expected context from the template pattern (`[B]`, not `B`).

**5. 88% of every batch was padding.** Sequences averaging 44 tokens were padded to 512. Dynamic per-batch padding gave a 10× speedup.

**6. A forced prefix that tokenised differently in isolation.** Constrained decoding forced `"graph TD\n"`, which encodes as `['graph','ĠTD','Ċ']`. But the training target `"graph TD\n    A["` encodes as `['graph','ĠTD','ĊĠĠĠ','ĠA',...]` — the newline **merges** with the indentation into one token. Forcing the bare `'Ċ'` put the model at a token it had never seen in that position, and it skipped the node letter on every single first line. Same characters, different tokens.

**7. Validating against the generator that produced the training data.** Every metric said 0.98+ while hand-typed prompts fell apart. The validation set was drawn from the same template generator as training, so it could not measure the thing that mattered. The diagnostic that finally separated *copying* (cost: 2 points) from *phrasing generalisation* (cost: 47.5 points) is in `scripts/diagnose.py`. The fix was expanding paraphrase coverage 26×, not more training.

**The recurring lesson:** every score above 0.9 in this project was, at some point, measuring something other than what it claimed to. Ad-hoc prompts typed by hand found more real bugs than any validation number.

---

## Acknowledgements

- Architecture follows Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).
- Pretraining data: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (Eldan & Li, 2023).
- Tokenization via HuggingFace [`tokenizers`](https://github.com/huggingface/tokenizers).

## License

MIT — see [LICENSE](LICENSE).
