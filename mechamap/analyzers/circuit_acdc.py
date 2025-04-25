"""
Internal re-implementation of ACDC
──────────────────────────────────
• Start from *all* edges OR from EAP pruned set
• Greedy drop the least-important edge batch while circuit still satisfies
  metric ≥ (baseline - δ)
• Optional verify loop: re-patch each remaining edge; if ablation doesn’t
  reduce metric, drop it.

Runs in O(|E| × log |E|) forwards on GPT-2-small.
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np, torch
from ..core import graph, patching
from ..config import Config
from .base import Analyzer
from .circuit_eap import _edge_grad_scores, _logit_diff  # reuse code

THRESH_DELTA = 0.01        # how much metric drop we allow

class ACDCAnalyzer:
    name = "acdc"

    def run(self, model, tokens, acts, cfg: Config) -> Dict[str, Any]:
        full_edges = graph.build(model)
        hooks = [h for *_ , h in full_edges]

        clean   = cfg.default_text
        corrupt = clean.replace("Paris", "Rome")
        tgt     = model.to_single_token(" Paris")

        # baseline logit diff on CLEAN prompt
        base_metric = _logit_diff(model(clean)[0], tgt).item()

        # 1. Initial grad scores  → sort descending
        scores = _edge_grad_scores(model, clean, corrupt, hooks, tgt)
        order = np.argsort(-scores)                 # high→low
        kept_mask = np.ones(len(full_edges), dtype=bool)

        # 2. Greedy prune in batches of 5 %
        batch = max(1, int(0.05 * len(full_edges)))
        for i in range(0, len(order), batch):
            trial_mask = kept_mask.copy()
            trial_mask[order[i:i+batch]] = False

            zero = torch.tensor(trial_mask.astype(float),
                                device=model.cfg.device)

            with patching.patch(model, hooks, zero):
                metric = _logit_diff(model(corrupt)[0], tgt).item()

            if base_metric - metric < THRESH_DELTA:
                kept_mask = trial_mask          # drop permanently
            # else keep current edges

        # 3. Verify each remaining edge individually (ACDC refine)
        verified: List[Tuple[str,str]] = []
        for idx, keep in enumerate(kept_mask):
            if not keep: continue
            zero = torch.tensor(kept_mask.astype(float),
                                device=model.cfg.device)
            zero[idx] = 0.0
            with patching.patch(model, hooks, zero):
                metric = _logit_diff(model(corrupt)[0], tgt).item()
            if base_metric - metric >= THRESH_DELTA:
                verified.append((full_edges[idx][0], full_edges[idx][1]))

        return {"edges": verified}
