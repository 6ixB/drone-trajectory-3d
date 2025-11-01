from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline
from .types import Float, Array, Function


def _to_f64(a: Array) -> Array:
    return np.asarray(a, dtype=Float)


def linear(xs: Array, ys: Array) -> Function:
    x = _to_f64(xs)
    y = _to_f64(ys)
    if x.size != y.size or x.size == 0:
        raise ValueError("xs and ys must be same nonzero length")
    # np.polyfit returns [slope, intercept] for deg=1
    slope, intercept = np.polyfit(x, y, deg=1).astype(Float)

    def f(t: Float) -> Float:
        return slope * t + intercept

    return f


def polynomial(xs: Array, ys: Array, order: int) -> Function:
    if order < 0:
        raise ValueError("order must be >= 0")
    x = _to_f64(xs)
    y = _to_f64(ys)
    if x.size != y.size or x.size == 0:
        raise ValueError("xs and ys must be same nonzero length")
    if order + 1 > x.size:
        raise ValueError("order too high for number of points")

    # polyfit returns highest power first. poly1d handles evaluation efficiently.
    coeffs = np.polyfit(x, y, deg=order).astype(Float)
    p = np.poly1d(coeffs)

    def f(t: Float) -> Float:
        return Float(p(t))

    return f


def cubic_spline(xs: Array, ys: Array) -> Function:
    x = _to_f64(xs)
    y = _to_f64(ys)
    if x.size != y.size or x.size < 2:
        raise ValueError("need at least two points")
    cs = CubicSpline(x, y, bc_type="not-a-knot")

    def f(t: Float) -> Float:
        return Float(cs(t))

    return f
