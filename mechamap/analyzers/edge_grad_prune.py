"""
EdgeGradientPruneAnalyzer
─────────────────────────
• One forward+backward pass → importance for *all* edges (≈ Auto-Circuit EAP)
• Greedy keep-k% edges, optional iterative refine (ACDC style)

Metric = logit difference between correct target and nearest competitor.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, List
import torch, numpy as np
from ..core import graph, patching, model_loader
from ..config import Config
from .base import Analyzer

def _logit_diff(logits, target_index):
    probs = logits[:, -1]   # [batch, vocab]
    tgt = probs[:, target_index]
    other = probs.max(dim=-1).values
    return (tgt - other).mean()

class EdgeGradientPruneAnalyzer:   # plug-in entry
    name = "egrad"

    def run(self, model, tokens, acts, cfg: Config) -> Dict[str, Any]:
        device = model.cfg.device
        edge_meta = graph.build(model)
        edge_names = [m for *_ , m in edge_meta]           # hook names
        mask = torch.ones(len(edge_names), device=device, requires_grad=True)

        # === choose prompts ===
        text_clean  = cfg.default_text
        text_corrupt = text_clean.replace("Paris", "Rome")  # toy corruption

        # forward clean
        clean_logits = model(text_clean)[0]
        # forward corrupt *with mask vars* so autograd tracks
        with patching.patch(model, edge_names, mask):
            corrupt_logits = model(text_corrupt)[0]

        metric = _logit_diff(corrupt_logits, target_index=model.to_single_token(" Paris"))
        (-metric).backward()        # maximise restoration → minimise -metric
        grads = mask.grad.detach().abs().cpu().numpy()     # importance signal

        # === greedy prune ===
        keep = grads > np.percentile(grads, (1-cfg.eap_sparsity) * 100)
        kept_edges = [e for e, k in zip(edge_meta, keep) if k]

        # optional refine: individually ablate and test causal effect
        verified: List[Tuple[str,str,str]] = []
        for e in kept_edges:
            zero = torch.ones_like(mask)
            zero[edge_meta.index(e)] = 0.0
            with patching.patch(model, edge_names, zero):
                logits = model(text_corrupt)[0]
            if _logit_diff(logits, model.to_single_token(" Paris")) < metric.item():
                verified.append(e)

        out_edges = [(s, t) for s, t, _ in verified]
        return {"edges": out_edges}
