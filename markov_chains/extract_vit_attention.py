#!/usr/bin/env python3
"""Extract full softmax attention maps from google/vit-base-patch16-224.

Outputs:
    vit_attention_float16.npy  (118, 12, 12, 197, 197)
    vit_logits_float32.npy     (118, 1000)
    vit_attention_metadata.json

Rows follow image IDs scene_000.png through scene_117.png.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

CHECKPOINT = "google/vit-base-patch16-224"
DEFAULT_IMAGES = Path("/home/maria/ViTExplainabilityExperimental/data/images")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: sibling of images folder)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    image_files = [args.images / f"scene_{i:03d}.png" for i in range(118)]
    missing = [str(p) for p in image_files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} expected images; examples: {missing[:5]}")

    out_dir = args.output or (args.images.parent / "vit_attention")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                  else "cpu")
    else:
        device = args.device
    print(f"Device: {device}")
    print(f"Loading {CHECKPOINT}")

    processor = AutoImageProcessor.from_pretrained(CHECKPOINT)
    model = AutoModelForImageClassification.from_pretrained(
        CHECKPOINT, attn_implementation="eager"
    ).to(device).eval()

    cfg = model.config
    layers = cfg.num_hidden_layers
    heads = cfg.num_attention_heads
    tokens = (cfg.image_size // cfg.patch_size) ** 2 + 1
    n_classes = cfg.num_labels

    attn_path = out_dir / f"vit_attention_{args.dtype}.npy"
    logits_path = out_dir / "vit_logits_float32.npy"
    attn = np.lib.format.open_memmap(
        attn_path, mode="w+", dtype=np.dtype(args.dtype),
        shape=(len(image_files), layers, heads, tokens, tokens)
    )
    logits = np.lib.format.open_memmap(
        logits_path, mode="w+", dtype=np.float32,
        shape=(len(image_files), n_classes)
    )

    max_row_error = 0.0
    with torch.inference_mode():
        for start in range(0, len(image_files), args.batch_size):
            end = min(start + args.batch_size, len(image_files))
            images = []
            for path in image_files[start:end]:
                with Image.open(path) as img:
                    images.append(img.convert("RGB"))
            batch = processor(images=images, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, output_attentions=True, return_dict=True)
            if outputs.attentions is None or len(outputs.attentions) != layers:
                raise RuntimeError("Model did not return all attention matrices; check Transformers version")
            # Tuple of (batch, heads, tokens, tokens), one entry per layer.
            stack = torch.stack(outputs.attentions, dim=1)
            expected = (end - start, layers, heads, tokens, tokens)
            if tuple(stack.shape) != expected:
                raise RuntimeError(f"Unexpected attention shape: {tuple(stack.shape)} vs {expected}")
            if not torch.isfinite(stack).all().item():
                raise RuntimeError(f"Non-finite attention in images {start}..{end - 1}")
            error = (stack.float().sum(dim=-1) - 1.0).abs().max().item()
            max_row_error = max(max_row_error, error)
            if error > 1e-3:
                raise RuntimeError(f"Attention is not row-stochastic (max error={error})")

            attn[start:end] = stack.cpu().numpy().astype(args.dtype, copy=False)
            logits[start:end] = outputs.logits.float().cpu().numpy()
            print(f"Saved images {start:03d}–{end - 1:03d} / 117 | row-sum error {error:.2e}", flush=True)

    attn.flush()
    logits.flush()
    meta = {
        "checkpoint": CHECKPOINT,
        "image_files": [p.name for p in image_files],
        "image_ids": list(range(118)),
        "attention_shape": list(attn.shape),
        "attention_axes": ["image", "layer", "head", "query_token", "key_token"],
        "dtype": args.dtype,
        "cls_token_index": 0,
        "patch_tokens": list(range(1, tokens)),
        "attention_definition": "softmax(Q @ K.T / sqrt(d_head))",
        "row_sum_max_abs_error_before_storage": max_row_error,
        "logits_file": logits_path.name,
        "attention_file": attn_path.name,
        "id2label": {str(k): v for k, v in cfg.id2label.items()},
    }
    (out_dir / "vit_attention_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Re-open to check that the saved file is readable and correctly ordered.
    check = np.load(attn_path, mmap_mode="r")
    assert check.shape == (118, layers, heads, tokens, tokens)
    print(f"\nDONE: {attn_path}")
    print(f"Shape: {check.shape}; dtype: {check.dtype}; size: {attn_path.stat().st_size / 1e9:.2f} GB")
    print(f"Logits: {logits_path}")
    print(f"Metadata: {out_dir / 'vit_attention_metadata.json'}")


if __name__ == "__main__":
    main()
