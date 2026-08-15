"""Single source of truth for every path in the project.

Every other module imports from here. This means scripts work regardless of the
current working directory -- you can run `python src/train_finetune.py` from the
repo root, or `python train_finetune.py` from inside src/, and both behave the same.
"""
from pathlib import Path

# src/paths.py -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TOKENIZER_DIR = DATA_DIR / "tokenizer"
TOKENIZER_FILE = TOKENIZER_DIR / "tokenizer.json"

CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.json"

CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
PRETRAIN_DIR = CHECKPOINT_ROOT / "pretrain"
FINETUNE_DIR = CHECKPOINT_ROOT / "finetune_mermaid"

PRETRAIN_BEST = PRETRAIN_DIR / "best_model.pt"
PRETRAIN_LAST = PRETRAIN_DIR / "last_model.pt"
FINETUNE_BEST = FINETUNE_DIR / "best_mermaid_model.pt"

TRAIN_BIN = PROCESSED_DIR / "train.bin"
VAL_BIN = PROCESSED_DIR / "val.bin"
MERMAID_TRAIN_JSON = PROCESSED_DIR / "mermaid_train.json"
MERMAID_VAL_JSON = PROCESSED_DIR / "mermaid_val.json"


def ensure_dirs():
    for d in (PROCESSED_DIR, TOKENIZER_DIR, PRETRAIN_DIR, FINETUNE_DIR):
        d.mkdir(parents=True, exist_ok=True)
