"""Shared library for the ECG representation-clustering benchmark.

Backward-compat shim: cohort caches under PTBXL_DIR were pickled before the code was
split into `core/`, so they reference a top-level `data` module. Alias it to `core.data`
(where Recording now lives) on package import, so those .pkl files unpickle from any entry
point. A fresh run that rebuilds the caches does not need this.
"""
import importlib as _importlib
import sys as _sys

_sys.modules.setdefault("data", _importlib.import_module("core.data"))
