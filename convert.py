#!/usr/bin/env python3
"""
convert.py – Convert LTX-Video / T5 safetensors weights to GGUF format.

Supports:
  - LTX-Video DiT (ltxv-2b / ltxv-13b)
  - CausalVideoVAE
  - T5-XXL (text encoder)

Usage:
  python3 convert.py --model dit   --input checkpoints/ltxv-2b.safetensors --output models/ltxv-2b-BF16.gguf
  python3 convert.py --model vae   --input checkpoints/ltxv-vae.safetensors --output models/ltxv-vae-BF16.gguf
  python3 convert.py --model t5    --input checkpoints/t5-xxl/ --output models/t5-xxl-BF16.gguf

Requirements:
  pip install gguf safetensors transformers torch
"""

import argparse
import os
import struct
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("gguf package not found. Install with: pip install gguf")

try:
    from safetensors import safe_open
    from safetensors.torch import load_file as st_load
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── Dtype helpers ─────────────────────────────────────────────────────────────

def to_bf16_np(arr: np.ndarray) -> np.ndarray:
    """Convert float32 ndarray to bfloat16 (stored as uint16)."""
    u32 = arr.astype(np.float32).view(np.uint32)
    return (u32 >> 16).astype(np.uint16)


def torch_to_np(t) -> np.ndarray:
    if t.dtype == torch.bfloat16:
        return t.to(torch.float32).numpy()
    return t.numpy()


# ── Safetensors loader ────────────────────────────────────────────────────────

def load_safetensors(path: str) -> Dict[str, np.ndarray]:
    """Load all tensors from a safetensors file as float32 numpy arrays."""
    if not HAS_SAFETENSORS:
        sys.exit("safetensors not installed. Run: pip install safetensors")
    tensors = {}
    if os.path.isdir(path):
        # Load sharded checkpoint.
        import json
        index_file = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(index_file):
            with open(index_file) as f:
                idx = json.load(f)
            shard_files = set(idx["weight_map"].values())
            for shard in sorted(shard_files):
                shard_path = os.path.join(path, shard)
                print(f"  loading shard {shard} ...")
                with safe_open(shard_path, framework="numpy", device="cpu") as f:
                    for k in f.keys():
                        tensors[k] = f.get_tensor(k).astype(np.float32)
        else:
            # Single file in directory.
            for fn in Path(path).glob("*.safetensors"):
                with safe_open(str(fn), framework="numpy", device="cpu") as f:
                    for k in f.keys():
                        tensors[k] = f.get_tensor(k).astype(np.float32)
    else:
        with safe_open(path, framework="numpy", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k).astype(np.float32)
    return tensors


# ── GGUF writer wrapper ───────────────────────────────────────────────────────

class GGUFBuilder:
    def __init__(self, output_path: str, arch: str):
        self.writer = gguf.GGUFWriter(output_path, arch)

    def add_kv(self, key: str, value):
        if isinstance(value, str):
            self.writer.add_string(key, value)
        elif isinstance(value, int):
            self.writer.add_uint32(key, value)
        elif isinstance(value, float):
            self.writer.add_float32(key, value)
        elif isinstance(value, bool):
            self.writer.add_bool(key, value)

    def add_tensor(self, name: str, data: np.ndarray):
        data = data.astype(np.float32)
        self.writer.add_tensor(name, data)

    def write(self):
        self.writer.write_header_to_file()
        self.writer.write_kv_data_to_file()
        self.writer.write_tensors_to_file()
        self.writer.close()


# ── DiT converter ─────────────────────────────────────────────────────────────

def convert_dit(tensors: Dict[str, np.ndarray], output: str, config: dict):
    """Convert LTX-Video DiT weights to GGUF."""
    print(f"Converting DiT → {output}")
    w = GGUFBuilder(output, "ltxv")

    # Metadata.
    w.add_kv("general.architecture",          "ltxv")
    w.add_kv("general.name",                  config.get("name", "LTX-Video DiT"))
    w.add_kv("ltxv.hidden_size",              config.get("hidden_size", 2048))
    w.add_kv("ltxv.num_hidden_layers",        config.get("num_hidden_layers", 28))
    w.add_kv("ltxv.num_attention_heads",      config.get("num_heads", 32))
    w.add_kv("ltxv.in_channels",              config.get("in_channels", 128))
    w.add_kv("ltxv.cross_attention_dim",      config.get("cross_attn_dim", 4096))
    w.add_kv("ltxv.patch_size",               config.get("patch_size", 2))

    # Write tensors preserving original names (expected by ltx_dit.hpp).
    n = 0
    for k, v in tensors.items():
        w.add_tensor(k, v)
        n += 1

    w.write()
    print(f"  wrote {n} tensors to {output}")


# ── VAE converter ─────────────────────────────────────────────────────────────

def convert_vae(tensors: Dict[str, np.ndarray], output: str):
    """Convert LTX-Video VAE weights to GGUF."""
    print(f"Converting VAE → {output}")
    w = GGUFBuilder(output, "ltxv-vae")

    w.add_kv("general.architecture",   "ltxv-vae")
    w.add_kv("general.name",           "LTX-Video CausalVideoVAE")
    w.add_kv("vae.latent_channels",    128)
    w.add_kv("vae.spatial_scale",      8)
    w.add_kv("vae.temporal_scale",     4)

    n = 0
    for k, v in tensors.items():
        # Prefix with "vae." if not already present.
        name = k if k.startswith("vae.") else "vae." + k
        w.add_tensor(name, v)
        n += 1

    w.write()
    print(f"  wrote {n} tensors to {output}")


# ── T5 converter ─────────────────────────────────────────────────────────────

def convert_t5(tensors: Dict[str, np.ndarray], output: str, tokenizer_path: Optional[str] = None):
    """Convert T5-XXL encoder weights to GGUF."""
    print(f"Converting T5 → {output}")
    w = GGUFBuilder(output, "t5")

    # Detect model size from embedding dim.
    emb_key = "encoder.embed_tokens.weight"
    if emb_key in tensors:
        vocab_size, d_model = tensors[emb_key].shape
    else:
        vocab_size, d_model = 32128, 4096

    # Count layers.
    num_layers = 0
    while f"encoder.block.{num_layers}.layer.0.SelfAttention.q.weight" in tensors:
        num_layers += 1
    if num_layers == 0:
        # Alternative naming.
        while f"encoder.block.{num_layers}.layer.0.SelfAttention.q.weight" in tensors:
            num_layers += 1
        num_layers = max(num_layers, 24)

    w.add_kv("general.architecture",      "t5")
    w.add_kv("general.name",              "T5-XXL encoder")
    w.add_kv("t5.block_count",            num_layers)
    w.add_kv("t5.embedding_length",       d_model)
    w.add_kv("t5.feed_forward_length",    d_model * 4 if d_model == 768 else 10240)
    w.add_kv("t5.attention.head_count",   12 if d_model == 768 else 64)
    w.add_kv("t5.vocab_size",             vocab_size)

    # Add tokenizer vocabulary if available.
    if tokenizer_path:
        try:
            from transformers import T5Tokenizer as HFT5Tok
            tok = HFT5Tok.from_pretrained(tokenizer_path)
            vocab = [tok.convert_ids_to_tokens(i) for i in range(tok.vocab_size)]
            w.writer.add_array("tokenizer.ggml.tokens", vocab)
            print(f"  embedded tokenizer ({len(vocab)} tokens)")
        except Exception as e:
            print(f"  warning: could not embed tokenizer: {e}")

    # Remap T5 tensor names to match ltx.cpp conventions.
    remap = {
        "encoder.embed_tokens.weight": "token_emb.weight",
        "encoder.final_layer_norm.weight": "encoder.final_layer_norm.weight",
    }

    n = 0
    for k, v in tensors.items():
        name = remap.get(k, k)
        # Filter to encoder-only tensors.
        if k.startswith("decoder."):
            continue
        if k == "shared.weight":
            # Shared embedding.
            w.add_tensor("token_emb.weight", v)
            n += 1
            continue
        w.add_tensor(name, v)
        n += 1

    w.write()
    print(f"  wrote {n} tensors to {output}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert LTX-Video / T5 weights to GGUF")
    parser.add_argument("--model",  required=True,
        choices=["dit", "vae", "t5"],
        help="Which model to convert")
    parser.add_argument("--input",  required=True,
        help="Input safetensors file or directory")
    parser.add_argument("--output", required=True,
        help="Output GGUF file path")
    parser.add_argument("--tokenizer", default=None,
        help="(T5 only) path to HF tokenizer directory")
    # DiT-specific config overrides.
    parser.add_argument("--hidden-size",    type=int, default=2048)
    parser.add_argument("--num-layers",     type=int, default=28)
    parser.add_argument("--num-heads",      type=int, default=32)
    parser.add_argument("--in-channels",    type=int, default=128)
    parser.add_argument("--cross-attn-dim", type=int, default=4096)
    args = parser.parse_args()

    print(f"loading {args.input} ...")
    tensors = load_safetensors(args.input)
    print(f"  loaded {len(tensors)} tensors")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if args.model == "dit":
        cfg = {
            "hidden_size":    args.hidden_size,
            "num_hidden_layers": args.num_layers,
            "num_heads":      args.num_heads,
            "in_channels":    args.in_channels,
            "cross_attn_dim": args.cross_attn_dim,
        }
        convert_dit(tensors, args.output, cfg)
    elif args.model == "vae":
        convert_vae(tensors, args.output)
    elif args.model == "t5":
        convert_t5(tensors, args.output, args.tokenizer)

    print("done.")


if __name__ == "__main__":
    main()
