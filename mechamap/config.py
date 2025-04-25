from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass(frozen=True)
class Config:
    # generic scan parms
    threshold:   float  = 0.20
    top_k:       int    = 5
    limit_neurons: Optional[int] = None
    verbose:     bool   = True

    # ACDC / EAP / CD-T knobs
    acdc_threshold: float = 0.05
    eap_sparsity:   float = 0.10
    cdt_token:      str   = "<OUTPUT>"

    # category → tokens
    categories: Dict[str, List[str]] = field(default_factory=lambda: {
        "location": ["Paris", "London", "Berlin", "Australia", "Canberra"],
        "numeric":  ["0", "1", "10", "3.14", "1000", "2025"],
        "food":     ["pizza", "sushi", "apple", "banana", "bread"],
        "animal":   ["cat", "dog", "lion", "tiger"],
        "color":    ["red", "blue", "green", "yellow"],
    })

    default_text: str = (
        "Paris is the capital of France. I ate sushi in Tokyo. "
        "A dog chased the red car. Pi is 3.14."
    )

    def __post_init__(self):
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold outside [0,1]")
