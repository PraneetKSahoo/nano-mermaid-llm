"""TinyStories -> uint16 token binaries -> memory-mapped Dataset.

Changes: uses paths.py, batches tokenizer calls (much faster), and streams token
IDs to disk in chunks instead of building one giant Python list (the old version
held ~25M Python ints in RAM, roughly 800MB, before writing).
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from paths import PROCESSED_DIR, TRAIN_BIN, VAL_BIN
from tokenizer import load_tokenizer

FLUSH_EVERY = 2_000_000  # token IDs buffered before writing to disk


def prepare_pretrain_data(sample_size: int = 100000, val_split: float = 0.05, force: bool = False):
    """Download TinyStories, tokenize with our BPE tokenizer, write uint16 binaries."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if TRAIN_BIN.exists() and VAL_BIN.exists() and not force:
        print("Pre-tokenized dataset binary files already exist.")
        return

    print("Loading raw TinyStories dataset...")
    tokenizer = load_tokenizer()
    dataset = load_dataset("roneneldan/TinyStories", split=f"train[:{sample_size}]")

    split_dataset = dataset.train_test_split(test_size=val_split, seed=42)
    train_data = split_dataset["train"]
    val_data = split_dataset["test"]

    endoftext_id = tokenizer.token_to_id("<|endoftext|>")
    assert endoftext_id is not None, "Tokenizer is missing <|endoftext|>"

    for split_name, data_split, bin_path in [
        ("train", train_data, TRAIN_BIN),
        ("val", val_data, VAL_BIN),
    ]:
        print(f"Tokenizing {split_name} split ({len(data_split):,} documents)...")
        total = 0
        buf = []

        with open(bin_path, "wb") as f:
            for start in range(0, len(data_split), 1000):
                texts = data_split[start:start + 1000]["text"]
                for enc in tokenizer.encode_batch(texts):
                    buf.extend(enc.ids)
                    buf.append(endoftext_id)

                if len(buf) >= FLUSH_EVERY:
                    f.write(np.array(buf, dtype=np.uint16).tobytes())
                    total += len(buf)
                    buf = []

                if start % 20000 == 0 and start:
                    print(f"    ...{start:,}/{len(data_split):,} docs")

            if buf:
                f.write(np.array(buf, dtype=np.uint16).tobytes())
                total += len(buf)

        print(f"Saved {split_name}: {total:,} tokens ({bin_path.stat().st_size/1e6:.2f} MB)")


class PretrainDataset(Dataset):
    """Memory-mapped Dataset for high-speed streaming of token sequences."""

    def __init__(self, bin_path: str, seq_len: int = 512):
        self.bin_path = bin_path
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.total_tokens = len(self.data)

    def __len__(self):
        return (self.total_tokens - 1) // self.seq_len

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1
        chunk = torch.from_numpy(self.data[start_idx:end_idx].astype(np.int64))
        return chunk[:-1], chunk[1:]


def get_pretrain_dataloaders(seq_len: int = 512, batch_size: int = 16, num_workers: int = 0):
    if not TRAIN_BIN.exists() or not VAL_BIN.exists():
        prepare_pretrain_data()

    train_dataset = PretrainDataset(str(TRAIN_BIN), seq_len=seq_len)
    val_dataset = PretrainDataset(str(VAL_BIN), seq_len=seq_len)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=num_workers, drop_last=True,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing dataset tokenization & binary streaming pipeline...")
    prepare_pretrain_data()

    train_loader, val_loader = get_pretrain_dataloaders(seq_len=512, batch_size=8)
    x, y = next(iter(train_loader))

    print("Pretraining DataLoader Verification Passed!")
    print(f"   - Train Batch Input Shape (x):  {x.shape}")
    print(f"   - Train Batch Target Shape (y): {y.shape}")
