import argparse
import math
import time

import torch
from torch.amp import autocast, GradScaler
try:
    import wandb
except ImportError:  # optional dependency
    wandb = None

from paths import PRETRAIN_DIR, PRETRAIN_BEST, PRETRAIN_LAST
from model import load_model_from_config
from dataset import get_pretrain_dataloaders
from tokenizer import load_tokenizer

BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
MAX_STEPS = 3000
WARMUP_STEPS = 300
MAX_LR = 6e-4
MIN_LR = 6e-5
EVAL_INTERVAL = 250
GENERATE_INTERVAL = 500
VAL_STEPS = 20

torch.backends.cudnn.benchmark = True


def _check_vocab(tokenizer, config):
    """A tokenizer larger than the model's vocab produces out-of-range token IDs
    and an immediate CUDA assert. Smaller is harmless (unused embedding rows),
    but usually means the tokenizer and config drifted apart."""
    tv, cv = tokenizer.get_vocab_size(), config["vocab_size"]
    if tv > cv:
        raise ValueError(
            f"Tokenizer vocab ({tv}) exceeds model vocab ({cv}). "
            f"Set vocab_size to {tv} in config/model_config.json and re-pretrain."
        )
    if tv < cv:
        print(f"    ! Tokenizer vocab ({tv}) < model vocab ({cv}); {cv - tv} embedding rows unused.")


def get_lr(step):
    """Cosine learning rate schedule with linear warmup."""
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    if step > MAX_STEPS:
        return MIN_LR
    decay_ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="Once upon a time", max_new_tokens=60):
    model.eval()
    enc = tokenizer.encode(prompt)
    idx = torch.tensor([enc.ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config["max_seq_len"]:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / 0.8
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        if idx_next.item() == tokenizer.token_to_id("<|endoftext|>"):
            break

    decoded = tokenizer.decode(idx[0].cpu().numpy().tolist())
    model.train()
    return decoded


def train(restart: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting pretraining on device: {device}")
    PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)

    model, config = load_model_from_config()
    model.to(device)
    tokenizer = load_tokenizer()

    # Guard against a stale checkpoint / retrained tokenizer mismatch. Without
    # this you get silent garbage rather than an error.
    _check_vocab(tokenizer, config)
    print(f"    - Model Parameters: {model.get_num_params()/1e6:.2f}M")

    train_loader, val_loader = get_pretrain_dataloaders(
        seq_len=config["max_seq_len"], batch_size=BATCH_SIZE
    )
    train_iter = iter(train_loader)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=MAX_LR, betas=(0.9, 0.95), weight_decay=0.1
    )
    scaler = GradScaler("cuda", enabled=(device == "cuda"))

    start_step = 1
    best_val_loss = float("inf")

    if restart:
        for p in (PRETRAIN_BEST, PRETRAIN_LAST):
            if p.exists():
                backup = p.with_name(f"{p.stem}_backup_{time.strftime('%Y%m%d_%H%M%S')}{p.suffix}")
                p.rename(backup)
                print(f"  --restart: moved '{p.name}' -> '{backup.name}'")

    # Resume from LAST, not BEST. Resuming from the best checkpoint rewinds the
    # step counter and the LR schedule to whenever val loss last improved.
    if PRETRAIN_LAST.exists():
        print(f"Resuming training from '{PRETRAIN_LAST}'...")
        ckpt = torch.load(PRETRAIN_LAST, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
        print(f"Resumed from Step {ckpt['step']} (Best Val Loss so far: {best_val_loss:.4f})")

    if start_step > MAX_STEPS:
        print(f"Already completed {MAX_STEPS} steps. Nothing to do.")
        return

    use_wandb = False
    try:
        if wandb is None:
            raise ImportError("wandb not installed")
        wandb.init(project="nano-mermaid-llm", name="pretrain-23M", config=config, mode="offline")
        use_wandb = True
        print("Weights & Biases Logging Enabled")
    except Exception:
        print("WandB not logged in/disabled. Training locally without W&B dashboard.")

    start_time = time.time()
    model.train()

    for step in range(start_step, MAX_STEPS + 1):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(GRAD_ACCUM_STEPS):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with autocast("cuda", enabled=(device == "cuda"), dtype=torch.float16):
                logits, loss = model(x, y)
                loss = loss / GRAD_ACCUM_STEPS

            accum_loss += loss.item()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            model.eval()
            val_loss = 0.0
            seen = 0
            val_iter = iter(val_loader)

            with torch.no_grad():
                for _ in range(VAL_STEPS):
                    try:
                        vx, vy = next(val_iter)
                    except StopIteration:
                        break
                    vx, vy = vx.to(device, non_blocking=True), vy.to(device, non_blocking=True)
                    with autocast("cuda", enabled=(device == "cuda"), dtype=torch.float16):
                        _, v_loss = model(vx, vy)
                    val_loss += v_loss.item()
                    seen += 1

            # Divide by batches actually seen
            val_loss /= max(1, seen)
            elapsed = time.time() - start_time
            print(f"Step {step:4d}/{MAX_STEPS} | Train Loss: {accum_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | LR: {lr:.2e} | Time: {elapsed:.1f}s")

            if use_wandb:
                wandb.log({"step": step, "train_loss": accum_loss, "val_loss": val_loss, "lr": lr})

            payload = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": config,
                "val_loss": val_loss,
                "best_val_loss": min(best_val_loss, val_loss),
            }
            torch.save(payload, PRETRAIN_LAST)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(payload, PRETRAIN_BEST)
                print(f"    Saved new best checkpoint (Val loss: {val_loss:.4f})")

            model.train()

        if step % GENERATE_INTERVAL == 0:
            sample = generate_sample(model, tokenizer, device, prompt="Once upon a time")
            print("\n" + "=" * 50)
            print(f"Sample Generation at Step {step}:")
            print(f'"{sample}"')
            print("=" * 50 + "\n")

    print(f"\nPretraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best weights: {PRETRAIN_BEST}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain the base model on TinyStories.")
    parser.add_argument("--restart", action="store_true",
                        help="Back up existing pretrain checkpoints and start fresh.")
    args = parser.parse_args()
    train(restart=args.restart)
