"""
patching
────────
Context-manager that masks (multiplies) selected edge activations.

Usage
-----
>>> edges  = [e[2] for e in graph.build(model)]
>>> mask   = torch.ones(len(edges), device=model.cfg.device, requires_grad=True)
>>> mask[17] = 0.0            # ablate one edge
>>> with patch(model, edges, mask):
...     logits = model(text)[0]
"""

from __future__ import annotations
from contextlib import contextmanager
from typing import Sequence, Dict, List

import torch
from transformer_lens import HookedTransformer


@contextmanager
def patch(
    model: HookedTransformer,
    edge_hooks: Sequence[str],   # list of hook-point names
    mask: torch.Tensor,          # 1-D tensor, same length
):
    """
    Multiply each edge’s activation by ``mask[i]`` *in-place*.
    Silent-no-op if a hook-point is disabled in the model’s config.
    """
    if mask.ndim != 1 or len(mask) != len(edge_hooks):
        raise ValueError("mask must be 1-D and match number of edges")

    edge2idx: Dict[str, int] = {h: i for i, h in enumerate(edge_hooks)}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def make_hook(idx: int):
        # mutate activation in-place; no return value needed
        def _hook(act, hook):
            act.mul_(mask[idx])
        return _hook

    # Register hooks; skip ones the model refuses
    for hname in edge_hooks:
        try:
            handle = model.add_hook(hname, make_hook(edge2idx[hname]))
        except AssertionError:
            # Hook point disabled by config (e.g. use_attn_in = False)
            handle = None
        handles.append(handle)

    try:
        yield
    finally:
        # Remove only valid handles
        for h in handles:
            if h is not None:
                h.remove()
