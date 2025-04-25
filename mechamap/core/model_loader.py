"""
model_loader
────────────
Thin wrapper around Transformer-Lens to load models with logging and
basic OOM handling.
"""

import logging, torch
from transformer_lens import HookedTransformer

log = logging.getLogger(__name__)


def load(model_name: str) -> HookedTransformer:
    """
    Load a pretrained model (e.g. 'gpt2', HF repo ID, or local path).

    Raises
    ------
    torch.cuda.OutOfMemoryError
        If the model can’t fit on the current GPU.
    """
    try:
        model = HookedTransformer.from_pretrained(model_name)
        log.info("Loaded %s into HookedTransformer", model_name)
        return model
    except torch.cuda.OutOfMemoryError:
        log.error("OOM loading %s.  Try a smaller model or CPU.", model_name)
        raise
