import argparse
import math
import time

import torch
from torch.amp import autocast, GradScaler
try:
    import wandb
except ImportError:  # optional dependency
    wandb = None

from paths import FINETUNE_DIR, FINETUNE_BEST, PRETRAIN_BEST
from model import load_model_from_config
from tokenizer import load_tokenizer
from dataset_mermaid import get_mermaid_dataloader
from metrics import evaluate_slots, show_examples

BATCH_SIZE = 32
EPOCHS = 30              # "epoch" = SAMPLES_PER_EPOCH fresh samples, not a pass
PATIENCE = 3             # over a fixed set -- training data is now generated on
SAMPLES_PER_EPOCH = 20000  # the fly, so there is nothing to memorise
MAX_LR = 4e-4
MIN_LR = 4e-5
WARMUP_STEPS = 300
EVAL_SAMPLES = 384       # val samples for the slot metric. At n=128 the
                         # stderr was +/-4.4%, so epoch-to-epoch wobble of
                         # 0.48 -> 0.52 was pure noise driving checkpoint
                         # selection. n=384 cuts that to +/-2.5%.

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


def make_lr_fn(total_steps):
    """Cosine decay with linear warmup, same recipe as pretraining.

    The old script held LR flat at 4e-4 for the whole run. A flat LR keeps
    nudging weights around at full step size right up to the last batch, so the
    model never settles into a minimum -- visible in the original log as val loss
    bouncing 1.04 -> 0.99 -> 0.87 -> 1.02 across consecutive epochs.
    """
    def lr_at(step):
        if step < WARMUP_STEPS:
            return MAX_LR * (step + 1) / WARMUP_STEPS
        if step >= total_steps:
            return MIN_LR
        ratio = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return MIN_LR + coeff * (MAX_LR - MIN_LR)
    return lr_at


def finetune(restart: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Mermaid fine-tuning (Batch Size={BATCH_SIZE}) on device: {device}")
    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)

    model, config = load_model_from_config()
    # Move to device BEFORE building the optimizer. optimizer.load_state_dict()
    # casts loaded state to match each parameter's CURRENT device, so if the
    # params are still on CPU at that moment, Adam's exp_avg/exp_avg_sq buffers
    # are created on CPU and a later model.to(device) does not follow them --
    # producing "Expected all tensors to be on the same device" on the first
    # optimizer step of a resumed run.
    model.to(device)
    tokenizer = load_tokenizer()

    _check_vocab(tokenizer, config)

    train_loader, val_loader, val_samples = get_mermaid_dataloader(
        tokenizer, batch_size=BATCH_SIZE, max_seq_len=config["max_seq_len"],
        on_the_fly=True, samples_per_epoch=SAMPLES_PER_EPOCH,
    )
    eval_subset = val_samples[:EVAL_SAMPLES]

    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.01)
    scaler = GradScaler("cuda", enabled=(device == "cuda"))

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * EPOCHS
    lr_at = make_lr_fn(total_steps)
    print(f"    - {steps_per_epoch:,} steps/epoch x {EPOCHS} epochs = {total_steps:,} total steps")

    start_epoch = 1
    best_score = -1.0
    best_val_loss = float("inf")

    if restart and FINETUNE_BEST.exists():
        backup = FINETUNE_BEST.with_name(
            f"{FINETUNE_BEST.stem}_backup_{time.strftime('%Y%m%d_%H%M%S')}{FINETUNE_BEST.suffix}"
        )
        FINETUNE_BEST.rename(backup)
        print(f"  --restart: moved existing checkpoint to '{backup.name}'.")

    if FINETUNE_BEST.exists():
        print(f"Resuming from fine-tune checkpoint '{FINETUNE_BEST}'...")
        ckpt = torch.load(FINETUNE_BEST, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            # Belt and braces: guarantee every optimizer buffer sits on `device`
            # regardless of how the checkpoint was written.
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            scaler.load_state_dict(ckpt.get("scaler_state_dict", scaler.state_dict()))
        start_epoch = ckpt.get("epoch", 0) + 1
        best_score = ckpt.get("sample_acc", -1.0)
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"    -> Epoch {start_epoch} (Best slot accuracy so far: {best_score:.3f})")
    elif PRETRAIN_BEST.exists():
        print(f"Loading pretrained base weights from '{PRETRAIN_BEST}'...")
        ckpt = torch.load(PRETRAIN_BEST, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise FileNotFoundError(f"Missing pretrained checkpoint at {PRETRAIN_BEST}.")

    if start_epoch > EPOCHS:
        print(f"Model has already completed {EPOCHS} epochs. Exiting.")
        return

    use_wandb = False
    try:
        if wandb is None:
            raise ImportError("wandb not installed")
        wandb.init(project="nano-mermaid-llm", name="finetune-mermaid", config=config, mode="offline")
        use_wandb = True
        print("Weights & Biases Logging Enabled.")
    except Exception:
        print("WandB disabled.")

    start_time = time.time()
    epochs_without_improvement = 0
    global_step = (start_epoch - 1) * steps_per_epoch

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            lr = lr_at(global_step)
            for g in optimizer.param_groups:
                g["lr"] = lr

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=(device == "cuda"), dtype=torch.float16):
                logits, loss = model(x, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            global_step += 1

        avg_train_loss = train_loss / len(train_loader)

        # --- validation loss ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device, non_blocking=True), vy.to(device, non_blocking=True)
                with autocast("cuda", enabled=(device == "cuda"), dtype=torch.float16):
                    _, v_loss = model(vx, vy)
                val_loss += v_loss.item()
        avg_val_loss = val_loss / len(val_loader)

        # --- the metric that actually tracks the copy behaviour ---
        m = evaluate_slots(model, tokenizer, eval_subset, device,
                           max_seq_len=config["max_seq_len"])
        elapsed = time.time() - start_time

        print(f"Epoch {epoch:2d}/{EPOCHS} | Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} "
              f"| Slot {m['slot_acc']:.3f} | AllSlots {m['sample_acc']:.3f} "
              f"(pool {m.get('sample_acc_pool', float('nan')):.3f} / "
              f"novel {m.get('sample_acc_novel', float('nan')):.3f}) "
              f"| Exact {m['exact_match']:.3f} | LR {lr:.2e} | {elapsed:.0f}s")

        if use_wandb:
            wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss,
                       "lr": lr, **m})

        if epoch % 5 == 0 or epoch == EPOCHS:
            show_examples(model, tokenizer, eval_subset, device,
                          max_seq_len=config["max_seq_len"], n=2)

        # Select on sample_acc, NOT val loss. Val loss cannot distinguish a model
        # that copies slots from one that guesses them plausibly.
        if m["sample_acc"] > best_score:
            best_score = m["sample_acc"]
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": config,
                "val_loss": avg_val_loss,
                **m,
            }, FINETUNE_BEST)
            print(f"    Saved checkpoint (all-slots accuracy: {best_score:.3f})")
        else:
            epochs_without_improvement += 1
            print(f"    No improvement ({epochs_without_improvement}/{PATIENCE})")
            if epochs_without_improvement >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} — slot accuracy plateaued.")
                break

    print(f"\nDone. Best all-slots accuracy: {best_score:.3f} (val loss {best_val_loss:.4f})")
    if best_score < 0.5:
        print("Slot accuracy below 0.5 means the copy circuit has not formed yet. "
              "Raise EPOCHS, or check that pretraining actually converged.")
    print("Sanity check: train loss should have settled around 0.05-0.30, NOT 0.000. "
          "A train loss of 0.000 with on-the-fly data would mean the sampler is "
          "repeating itself.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the Mermaid model.")
    parser.add_argument("--restart", action="store_true",
                        help="Ignore any existing fine-tune checkpoint and start fresh from the "
                             "pretrained base model (old checkpoint is backed up, not deleted).")
    args = parser.parse_args()
    finetune(restart=args.restart)
