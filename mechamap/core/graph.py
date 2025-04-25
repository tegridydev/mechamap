"""
graph
─────
Return a static edge list for a HookedTransformer (TL ≥ 2.0).

We respect the feature-flags that govern which hook points actually exist:

• use_split_qkv_input – if False  → hook_q/k/v_input do **not** work
• use_attn_in         – if False  → hook_attn_in   does **not** work
• use_attn_result_hook or use_attn_result – if False → hook_z/attn_out disabled

The code detects which flags are present on model.cfg and only emits edges
that are guaranteed legal.
"""

from __future__ import annotations
from typing import List, Tuple
from transformer_lens import HookedTransformer

Edge = Tuple[str, str, str]        # (src_node, dst_node, hook_name)


def _cfg_flag(cfg, name: str, default: bool) -> bool:
    """Return cfg.<name> if it exists, else *default*."""
    return getattr(cfg, name, default)


def build(model: HookedTransformer) -> List[Edge]:
    cfg = model.cfg

    split_qkv = _cfg_flag(cfg, "use_split_qkv_input", False)
    allow_in  = _cfg_flag(cfg, "use_attn_in",          False)

    # TL 2.0 → 2.13 used `use_attn_result`
    # TL 2.14+ renamed it to `use_attn_result_hook`
    allow_z   = _cfg_flag(cfg, "use_attn_result_hook",
                _cfg_flag(cfg, "use_attn_result", True))

    edges: List[Edge] = []

    for layer in range(cfg.n_layers):
        rs  = f"L{layer}.ResStart"
        re  = f"L{layer}.ResEnd"
        mlp = f"L{layer}.MLP"

        # ─── Attention path ────────────────────────────────────────────────
        if split_qkv:
            for head in range(cfg.n_heads):
                q = f"L{layer}.H{head}.q"
                k = f"L{layer}.H{head}.k"
                v = f"L{layer}.H{head}.v"
                o = f"L{layer}.H{head}.o"

                edges += [
                    (rs, q, f"blocks.{layer}.hook_q_input"),
                    (rs, k, f"blocks.{layer}.hook_k_input"),
                    (rs, v, f"blocks.{layer}.hook_v_input"),
                ]
                if allow_z:
                    edges += [
                        (q, o, f"blocks.{layer}.attn.hook_z"),
                        (v, o, f"blocks.{layer}.attn.hook_z"),
                    ]
        else:
            # single tensor into attn
            if allow_in:
                attn_in = f"L{layer}.AttnIn"
                edges.append((rs, attn_in, f"blocks.{layer}.hook_attn_in"))

        # ─── MLP path ──────────────────────────────────────────────────────
        edges += [
            (rs,  mlp, f"blocks.{layer}.mlp.hook_pre"),
            (mlp, re,  f"blocks.{layer}.hook_resid_post"),
        ]

        # residual after attention (if z-hook allowed)
        if allow_z:
            attn_out = f"L{layer}.AttnOut"
            edges.append((attn_out, re, f"blocks.{layer}.hook_resid_post"))

    return edges
