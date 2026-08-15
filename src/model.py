import math
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

from paths import CONFIG_PATH


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using PyTorch SDPA."""

    def __init__(self, config):
        super().__init__()
        assert config["n_embd"] % config["n_head"] == 0

        # K, Q, V projections for all heads in one batched matmul
        self.c_attn = nn.Linear(config["n_embd"], 3 * config["n_embd"], bias=config["bias"])
        self.c_proj = nn.Linear(config["n_embd"], config["n_embd"], bias=config["bias"])

        self.attn_dropout = nn.Dropout(config["dropout"])
        self.resid_dropout = nn.Dropout(config["dropout"])

        self.n_head = config["n_head"]
        self.n_embd = config["n_embd"]
        self.dropout = config["dropout"]

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


# Backwards-compatible alias: old checkpoints were saved with the misspelled
# class name. State dicts key off parameter names, not class names, so this is
# only here so any external import of the old name keeps working.
CasualSelfAttention = CausalSelfAttention


class MLP(nn.Module):
    """Feed-forward network block."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config["n_embd"], 4 * config["n_embd"], bias=config["bias"])
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config["n_embd"], config["n_embd"], bias=config["bias"])
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-LN architecture (LayerNorm before Attention and MLP)."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config["n_embd"])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config["n_embd"])
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class NanoGPT(nn.Module):
    """GPT language model architecture built from scratch."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config["vocab_size"], config["n_embd"]),
            wpe=nn.Embedding(config["max_seq_len"], config["n_embd"]),
            drop=nn.Dropout(config["dropout"]),
            h=nn.ModuleList([TransformerBlock(config) for _ in range(config["n_layer"])]),
            ln_f=nn.LayerNorm(config["n_embd"]),
        ))

        self.lm_head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False)

        # Weight tying (saves params, ties input and output representations)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        # Scaled init for residual projections (GPT-2 recipe)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config["n_layer"]))

    def _init_weights(self, module):
        # FIXED. Previously the nn.Embedding branch was nested inside the
        # nn.Linear branch as an `elif module.bias is not None` sibling, so it
        # was unreachable -- no Embedding could ever also be a Linear. The
        # embeddings therefore kept PyTorch's default N(0, 1), which is 50x too
        # wide, and because wte.weight is TIED to lm_head.weight the output head
        # started with enormous logits. The old std was also 0.2, not 0.02.
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config["max_seq_len"], (
            f"Can't forward seq of length {t}, max_seq_len is {self.config['max_seq_len']}"
        )

        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # Inference: only the last position's logits are needed
            logits = self.lm_head(x[:, -1:, :])
            loss = None

        return logits, loss


def load_model_from_config(config_path=None):
    path = Path(config_path) if config_path else CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model config not found at {path}")
    with open(path, "r") as f:
        config = json.load(f)
    model = NanoGPT(config)
    return model, config


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing model architecture on: {device}")

    model, config = load_model_from_config()
    model.to(device)

    dummy_input = torch.randint(0, config["vocab_size"], (2, 64), device=device)
    dummy_targets = torch.randint(0, config["vocab_size"], (2, 64), device=device)
    logits, loss = model(dummy_input, dummy_targets)

    print("Model Architecture Verification Passed!")
    print(f"   - Total Parameters: {model.get_num_params():,} ({model.get_num_params()/1e6:.2f} Million)")
    print(f"   - Output Logits Shape: {logits.shape}")
    print(f"   - Loss on Dummy Batch: {loss.item():.4f}")
    print(f"   - Expected loss at init: {math.log(config['vocab_size']):.4f}  (ln of vocab size)")
    print(f"   - wte weight std: {model.transformer.wte.weight.std().item():.4f} (should be ~0.02)")
