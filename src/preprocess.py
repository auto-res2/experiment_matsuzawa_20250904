"""
src/preprocess.py
-----------------
Dataset / benchmark loaders and any data-related utilities.

This file previously performed *eager* imports of several benchmark creation
helpers that are **not** present in the Avalanche version currently shipped
with the execution environment (v0.6.0 at the time of writing).  Importing an
attribute that does not exist raises an `ImportError` and prevents the whole
package from being imported.

The fix implements **lazy, best-effort imports with graceful degradation**:
• Only the benchmark requested by the caller is imported.
• Multiple possible locations / spellings are consulted to maximise
  compatibility across Avalanche versions.
• If the benchmark is ultimately unavailable an informative error message is
  raised instead of failing at import-time.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable

__all__ = ["get_benchmark"]


# -----------------------------------------------------------------------------
#  INTERNAL HELPERS
# -----------------------------------------------------------------------------

def _resolve_symbol(module_names: list[str], symbol: str) -> Callable | None:
    """Return the first occurrence of *symbol* in the candidate modules.

    Parameters
    ----------
    module_names: list[str]
        Fully-qualified module names that will be imported in order.
    symbol: str
        Name of the attribute we are after.

    Returns
    -------
    object | None
        The attribute if found, *None* otherwise (import and attribute errors
        are swallowed).
    """
    for mod_name in module_names:
        try:
            mod = importlib.import_module(mod_name)
            return getattr(mod, symbol)
        except (ImportError, AttributeError):
            continue
    return None


# -----------------------------------------------------------------------------
#  BENCHMARK FACTORY
# -----------------------------------------------------------------------------

def get_benchmark(name: str, cfg: dict[str, Any] | None = None):
    """Return an Avalanche continual-learning benchmark by *name*.

    Supported benchmark identifiers (case-sensitive):
        "SplitCIFAR100", "SplitMiniImageNet", "SplitTinyImageNet", "PermutedMNIST"

    The function attempts to locate the corresponding creation helper in a
    variety of sub-modules in order to be robust to API changes between
    Avalanche releases.  A *clear* and *actionable* error is raised only if
    the benchmark cannot be located.
    """
    cfg = cfg or {}

    # Mapping from public name → (factory symbol, default kwargs)
    _registry: dict[str, tuple[str, dict[str, Any]]] = {
        "SplitCIFAR100": ("SplitCIFAR100", {"n_experiences": 20}),
        "SplitMiniImageNet": ("SplitMiniImageNet", {"n_experiences": 20}),
        "SplitTinyImageNet": ("SplitTinyImageNet", {"n_experiences": 10}),
        "PermutedMNIST": ("PermutedMNIST", {"n_experiences": 10}),
    }

    if name not in _registry:
        raise ValueError(f"Unknown benchmark name: {name}")

    symbol, default_kwargs = _registry[name]

    # Some classes were renamed in Avalanche (<-> lower-case "n" in Imagenet).  We
    # therefore try a set of plausible fall-back spellings.
    aliases = {
        "SplitMiniImageNet": ["SplitMiniImagenet"],
        "SplitTinyImageNet": ["SplitTinyImagenet"],
    }
    candidate_symbols = [symbol] + aliases.get(name, [])

    # Candidate modules (most specific first)
    modules_to_try = [
        "avalanche.benchmarks.classic",  # official location in recent releases
        "avalanche.benchmarks",          # legacy flat namespace
    ]

    benchmark_cls = None
    for sym in candidate_symbols:
        benchmark_cls = _resolve_symbol(modules_to_try, sym)
        if benchmark_cls is not None:
            break

    if benchmark_cls is None:
        raise ImportError(
            f"Benchmark '{name}' (symbols tried: {candidate_symbols}) was not found in "
            "the installed Avalanche version. Please ensure the correct package "
            "version is installed or adjust the experiment configuration."
        )

    # Merge user overrides (if any) with sensible defaults.
    kwargs = {**default_kwargs, **cfg.get("dataset_kwargs", {})}
    return benchmark_cls(**kwargs)
