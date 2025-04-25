"""
Internal re-implementation of **Edge Attribution Patching (EAP)**
──────────────────────────────────────────────────────────────────
1. Build full edge list            (core.graph)
2. Attach mask variables to edges  (core.patching)
3. One forward-plus-backward on a corrupt prompt:
      ∂(logit-diff) / ∂mask   → edge-importance
4. Keep top-k % edges as the recovered circuit.

The class is exposed under the historic name **EAPAnalyzer** so legacy
CLI flags keep working (`--analyzer eap`), even though the same logic also
lives in *edge_grad_prune.py*.
"""
from __future__ import annotations

from typing import Dict, Any, List
import numpy as np
import torch

from ..core import graph, patching
from ..config import Config
from .base import Analyzer


# ────────────────────────── helpers ──────────────────────────
def _last_token_slice(logits: torch.Tensor) -> torch.Tensor:
    """
    Return a **(batch, vocab)** tensor containing the logits of the *final*
    sequence position, regardless of Transformer-Lens version.

    * TL ≤ 1.16 → logits.shape == (batch, **seq**, vocab)  
    * TL ≥ 2.14 → logits.shape == (batch, vocab)
    """
    if logits.ndim == 3:          # old behaviour
        return logits[:, -1, :]
    if logits.ndim == 2:          # new behaviour
        return logits
    raise ValueError(f"Unexpected logits shape {logits.shape}")


def _logit_diff(logits: torch.Tensor, target_idx: int) -> torch.Tensor:
    """
    **Metric** = (target-token logit − best-other logit) averaged over batch,
    evaluated at the final sequence position.
    """
    final = _last_token_slice(logits)          # (B, V)
    tgt   = final[:, target_idx]
    other = final.max(dim=-1).values
    return (tgt - other).mean()


def _edge_grad_scores(
    model,
    clean_text: str,
    corrupt_text: str,
    edge_hooks: List[str],
    target_token: int,
) -> np.ndarray:
    """
    Compute |∂(logit-diff)/∂mask_i| for every edge *i* in `edge_hooks`
    using a single forward-/backward-pass on the **corrupt** prompt.
    """
    device = model.cfg.device
    mask   = torch.ones(len(edge_hooks), device=device, requires_grad=True)

    with patching.patch(model, edge_hooks, mask):
        logits = model(corrupt_text)[0]        # Tuple -> logits tensor

    metric = _logit_diff(logits, target_token)
    (-metric).backward()                       # maximise metric
    return mask.grad.detach().abs().cpu().numpy()


# ───────────────────────── Analyzer ──────────────────────────
class EAPAnalyzer(Analyzer):            # re-exported in analyzers/__init__.py
    """Gradient-based edge pruning à la EAP."""
    name = "eap"

    def run(
        self,
        model,
        tokens,        # unused – kept to satisfy Analyzer API
        acts,          # unused – kept to satisfy Analyzer API
        cfg: Config,
    ) -> Dict[str, Any]:

        # 1) enumerate every possible edge
        edges = graph.build(model)                 # [(src, dst, hook_name)]
        edge_hooks = [h for *_ , h in edges]

        # 2) craft (clean, corrupt) prompts
        clean   = cfg.default_text
        corrupt = clean.replace("Paris", "Rome")   # trivial corruption
        target  = model.to_single_token(" Paris")  # assumes leading space

        # 3) importance scores via ∇mask
        scores = _edge_grad_scores(
            model, clean, corrupt, edge_hooks, target
        )

        # 4) keep the top-k % edges
        pct      = (1.0 - cfg.eap_sparsity) * 100
        thresh   = np.percentile(scores, pct)
        keepers  = scores > thresh
        kept_edges = [
            (src, dst) for (src, dst, _), k in zip(edges, keepers) if k
        ]

        return {
            "edges": kept_edges,
            "scores": scores.tolist(),
        }
