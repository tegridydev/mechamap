"""
mechamap.analyzers
──────────────────
Central registry of all built-in (and plugin) analyzers.
"""

from importlib import import_module
from pkg_resources import iter_entry_points
import logging

log = logging.getLogger(__name__)

REGISTRY = {}

def _safe_import(module_fqname: str, cls_name: str):
    """
    Import `module_fqname`, fetch `cls_name`, add to REGISTRY.
    On ANY error we log a warning and keep going.
    """
    try:
        mod = import_module(module_fqname)
        cls = getattr(mod, cls_name)
        REGISTRY[cls.name] = cls
    except Exception as e:
        log.warning("Analyzer %s (%s) disabled: %s",
                    cls_name, module_fqname, e)

BASE = "mechamap.analyzers"

# ─── built-ins ─────────────────────────────────────────────────────────
_safe_import(f"{BASE}.baseline_activation", "BaselineActivationAnalyzer")   # → 'baseline'
_safe_import(f"{BASE}.edge_grad_prune",     "EdgeGradientPruneAnalyzer")    # → 'egrad'
_safe_import(f"{BASE}.circuit_eap",        "EAPAnalyzer")                  # → 'eap'
_safe_import(f"{BASE}.circuit_acdc",       "ACDCAnalyzer")                 # → 'acdc'
_safe_import(f"{BASE}.circuit_cdt",        "CDTAnalyzer")                  # → 'cdt'

# ─── third-party plugins (optional) ────────────────────────────────────
for ep in iter_entry_points("mechamap.analyzers"):
    try:
        cls = ep.load()
        REGISTRY[cls.name] = cls
    except Exception as e:
        log.warning("Failed to load plugin %s: %s", ep.name, e)

# convenience helpers
def get(name: str):
    if name not in REGISTRY:
        raise KeyError(
            f"Analyzer '{name}' not found. "
            f"Available: {', '.join(REGISTRY) if REGISTRY else '(none)'}"
        )
    return REGISTRY[name]

def all():
    return list(REGISTRY)
