"""
Placeholder MIB adapter
───────────────────────
We removed the external `mechanistic_interpretability_benchmark` dep.
This stub raises a clear message if someone tries to call it.
"""
def evaluate(*_, **__):
    raise RuntimeError(
        "MIB evaluation requires the official MIB dataset. "
        "Either install it from their repo or remove the CLI call."
    )
