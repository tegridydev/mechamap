import numpy as np, logging
from typing import Dict, Any, Mapping, Tuple, List
from ..config import Config
from .base import Analyzer
log = logging.getLogger(__name__)

class BaselineActivationAnalyzer:  # registered via __init__.py
    name = "baseline"

    def run(
        self,
        model,
        tokens: List[str],
        acts: Mapping[Tuple[int,int], np.ndarray],
        cfg: Config
    ) -> Dict[str, Any]:
        tok2cats = {t: cats for cats in cfg.categories.items()
                    for t in cats[1]}
        results = {}
        for (layer, n), vec in acts.items():
            cats_sum, cats_cnt = {}, {}
            for idx, tok in enumerate(tokens):
                for cat in tok2cats.get(tok, []):
                    cats_sum[cat] = cats_sum.get(cat, 0.) + float(vec[idx])
                    cats_cnt [cat] = cats_cnt .get(cat, 0)  + 1
            cats_avg = {c: s/cats_cnt[c] for c,s in cats_sum.items()
                        if s/cats_cnt[c] >= cfg.threshold}
            if cats_avg:
                top_idx = np.argsort(vec)[::-1][:cfg.top_k]
                results[(layer, n)] = {
                    "categories": cats_avg,
                    "top_tokens": [(tokens[i], float(vec[i])) for i in top_idx]
                }
        log.info("Baseline found %d candidate neurons", len(results))
        return results
