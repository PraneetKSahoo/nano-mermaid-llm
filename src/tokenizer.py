"""BPE tokenizer trained on a MIXED corpus.

The original tokenizer was trained on TinyStories alone -- a corpus of
deliberately simple children's English. It was then asked to encode things like
`Primary Warehouse`, `stateDiagram-v2`, `A{{...}}` and `-->`, none of which have
merges in a children's-story vocabulary. Those strings shattered into long runs
of near-character-level byte tokens.

That matters a great deal for this task. Copying a slot is not one attention
hop; it is an induction circuit that has to fire correctly across every token of
the string. The more fragments a slot value is split into, the longer the chain
and the more attractive it is for the model to just guess instead.

Mixing the Mermaid corpus into tokenizer training gives the technical terms and
Mermaid punctuation real merges, shortening every slot value to a handful of
tokens and making the copy circuit far easier to learn.

Vocab stays at 8192. Because of weight tying, vocab size directly sets the size
of both the embedding and the output head (8192 x 512 = 4.2M of the model's
23M params), so growing it is not free.
"""
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from datasets import load_dataset

from paths import TOKENIZER_DIR, TOKENIZER_FILE

SPECIAL_TOKENS = [
    "<|pad|>",          # padding so all items in a batch have equal length
    "<|unk|>",          # fallback token for unknown char
    "<|startoftext|>",
    "<|endoftext|>",
    "<|mermaid|>",      # control token for fig generation
    "<|prompt|>",       # user prompt
    "<|response|>",     # target model response
]

VOCAB_SIZE = 8192
TINYSTORIES_DOCS = 25000
MERMAID_PAIRS = 40000


def train_tokenizer(sample_size: int = TINYSTORIES_DOCS, mermaid_pairs: int = MERMAID_PAIRS):
    """Train BPE on TinyStories + the Mermaid corpus and save to data/tokenizer/."""
    from dataset_mermaid import iter_tokenizer_corpus

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading pre-training dataset slice (TinyStories)...")
    dataset = load_dataset("roneneldan/TinyStories", split=f"train[:{sample_size}]")

    def batch_iterator(batch_size=1000):
        # Mermaid corpus first so its merges are established early.
        buf = []
        for line in iter_tokenizer_corpus(num_pairs=mermaid_pairs):
            buf.append(line)
            if len(buf) >= batch_size:
                yield buf
                buf = []
        if buf:
            yield buf
        for i in range(0, len(dataset), batch_size):
            yield dataset[i:i + batch_size]["text"]

    print("Initializing BPE Tokenizer...")
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    # ByteLevel handles arbitrary code symbols and indentation cleanly.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"Training tokenizer ({mermaid_pairs:,} mermaid pairs + {sample_size:,} stories)...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.save(str(TOKENIZER_FILE))

    print(f"Tokenizer saved to '{TOKENIZER_FILE}'")
    print(f"    - Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"    - Special tokens:  {SPECIAL_TOKENS}")
    report_efficiency(tokenizer)


def report_efficiency(tokenizer=None):
    """Print tokens-per-string for the technical strings this project cares about."""
    tokenizer = tokenizer or load_tokenizer()
    probes = [
        "Primary Warehouse", "Compliance Registry", "Quarantine Manifest",
        "Awaiting Reconciled", "Retail Storefront", "stateDiagram-v2",
        "graph TD", "graph TB", "    A[Load Invoice] --> B[Parse Ledger]",
        "    A{Sign Contract} -- Valid --> B[Approve Order]",
        "    [*] --> Awaiting Reviewed",
    ]
    print("\nTokenization efficiency check:")
    for s in probes:
        enc = tokenizer.encode(s)
        print(f"  {len(enc.ids):3d} tokens | {s}")
        print(f"            {enc.tokens}")


def load_tokenizer() -> Tokenizer:
    if not TOKENIZER_FILE.exists():
        raise FileNotFoundError(
            f"Tokenizer missing at {TOKENIZER_FILE}. Run `python src/tokenizer.py` first."
        )
    return Tokenizer.from_file(str(TOKENIZER_FILE))


if __name__ == "__main__":
    train_tokenizer()
