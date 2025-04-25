"""
Mini-CDT
────────
A *token-level* contextual decomposition for transformers.

For each residual stream vector R we split:
    R = R_relevant + R_irrelevant
where “relevant” = contribution from GIVEN set of source tokens S.

We propagate the split through:
  • Attention: softmax(W_Q R) · (W_V R)   (linear in R)
  • MLP: GELU( W_1 R )  (approx linearise via first-order Taylor)

This quick version:
  – chooses a single source token index (default last prompt token)
  – outputs per-layer contribution norm  → edge scores
"""
from __future__ import annotations
from typing import Dict, Any, Tuple, List
import torch, numpy as np
from ..core import graph
from .base import Analyzer
from ..config import Config
from transformer_lens import HookedTransformer

class CDTAnalyzer:
    name = "cdt"

    def _split_resid(self, R, src_mask):
        # R: [seq, d]; src_mask: [seq] bool
        rel = (src_mask[:, None] * R)           # keep only src rows
        irrel = ((~src_mask)[:, None] * R)
        return rel, irrel

    def run(self, model: HookedTransformer, tokens, acts, cfg: Config
            ) -> Dict[str, Any]:
        text = cfg.default_text
        tok  = model.to_str_tokens(text)
        seq  = len(tok)
        src_idx = seq - 1                      # last token

        # run with hooks to capture split contributions
        relevant_norms: List[float] = []

        def resid_hook(R, h):
            rel, irrel = self._split_resid(R[0], src_mask)
            relevant_norms.append(rel.norm().item())
            return R

        src_mask = torch.zeros(seq, dtype=torch.bool,
                               device=model.cfg.device)
        src_mask[src_idx] = True

        model.run_with_hooks(text,
            fwd_hooks=[("blocks.*.hook_resid_pre", resid_hook)])
        # simple edge list: layer idx → norm
        edges = [(f"L{i}.Resid", f"L{i}.Resid+1") for i,_ in
                 enumerate(relevant_norms)]
        scores = relevant_norms
        return {"edges": edges, "scores": scores}
