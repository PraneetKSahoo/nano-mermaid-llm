"""Quick probe: how does your tokenizer split words that are NOT in the slot pools?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from tokenizer import load_tokenizer

t = load_tokenizer()
in_pool = ["Filter Dossier", "Approve Order", "Retail Storefront"]
out_pool = ["Scan Barcode", "Weigh Parcel", "Print Label",
            "Refund Request", "Issue Credit", "Notify Customer"]

for label, group in [("IN the training pools", in_pool), ("NOT in any pool", out_pool)]:
    print(f"\n--- {label} ---")
    for s in group:
        enc = t.encode(" " + s)
        print(f"  {len(enc.ids):2d} tokens | {s:20s} {enc.tokens}")
