"""
activations
───────────
Grab **every** post-MLP activation in a single forward pass.

Returns
-------
Mapping[(layer, neuron)] → np.ndarray[length == seq_len]
"""

from typing import Dict, Tuple, Mapping
import numpy as np
from transformer_lens import HookedTransformer, utils as tl_utils


def grab_post_mlp(
    model: HookedTransformer,
    text: str,
) -> Mapping[Tuple[int, int], np.ndarray]:
    """
    Collect post-MLP activations for all layers / neurons.

    Transformer-Lens 2.x disallows wildcards, so we register one hook per
    layer.  Each inner hook captures its layer index via closure.
    """
    out: Dict[Tuple[int, int], np.ndarray] = {}
    fwd_hooks = []

    for layer in range(model.cfg.n_layers):
        hook_name = f"blocks.{layer}.mlp.hook_post"

        def _make(idx: int):
            def _hook(act, hook):
                arr = tl_utils.to_numpy(act[0])        # (seq, d_mlp)
                for n in range(arr.shape[-1]):
                    out[(idx, n)] = arr[:, n]
            return _hook

        fwd_hooks.append((hook_name, _make(layer)))

    model.run_with_hooks(text, fwd_hooks=fwd_hooks)
    return out
