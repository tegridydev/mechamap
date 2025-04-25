from __future__ import annotations
from typing import Protocol, Mapping, Tuple, Dict, Any
import numpy as np

class Analyzer(Protocol):
    name: str
    def run(
        self,
        model,
        tokens: list[str],
        activations: Mapping[Tuple[int,int], np.ndarray],
        cfg
    ) -> Dict[str, Any]: ...
