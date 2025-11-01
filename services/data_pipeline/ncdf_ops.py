from __future__ import annotations
from typing import Any
import numpy as np
from netCDF4 import Variable
from .types import Float, Array


def pick_slice(var: Variable[Any], t0: int, t1: int, sidx: int) -> Array:
    a = var  # netCDF4.Variable is not generic
    if a.ndim >= 2 and a.shape[0] >= (t1 + 1) and a.shape[1] > sidx:
        return np.asarray(a[t0: t1 + 1, sidx], dtype=Float)
    if a.ndim >= 2 and a.shape[1] >= (t1 + 1) and a.shape[0] > sidx:
        return np.asarray(a[sidx, t0: t1 + 1], dtype=Float)
    if a.ndim == 1 and a.shape[0] >= (t1 + 1):
        return np.asarray(a[t0: t1 + 1], dtype=Float)
    raise IndexError(f"Unsupported shape {a.shape} for slice.")


def norm_fill(a: Array, fill_value: Float, eps: Float) -> Array:
    out = a.astype(Float, copy=True)
    m = np.isfinite(out) & (np.abs(out - fill_value) <= eps)
    out[m] = np.nan
    return out
