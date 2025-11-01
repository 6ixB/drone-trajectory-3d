from __future__ import annotations
from typing import Tuple
from services.data_pipeline.types import Float, Array
import numpy as np
from .config import Config


def solve(c: Config) -> Tuple[Tuple[Array, Array, Array], Tuple[Array, Array, Array]]:
    if c.dt <= 0:
        raise ValueError("Dt must be > 0.")
    if c.steps <= 1:
        raise ValueError("Steps must be > 1.")

    # Positions
    xs = np.zeros(shape=c.steps, dtype=Float)
    ys = np.zeros(shape=c.steps, dtype=Float)
    zs = np.zeros(shape=c.steps, dtype=Float)
    xs[0], ys[0], zs[0] = c.x0, c.y0, c.z0

    # Velocities
    bxs = np.zeros(shape=c.steps, dtype=Float)
    bys = np.zeros(shape=c.steps, dtype=Float)
    bzs = np.zeros(shape=c.steps, dtype=Float)
    bxs[0], bys[0], bzs[0] = _get_drone_velocity_components(
        xs[0],
        ys[0],
        zs[0],
        c
    )

    t = c.t0

    for i in range(1, c.steps):
        xs[i], ys[i], zs[i] = _rk4(t, xs[i - 1], ys[i - 1], zs[i - 1], c)
        bxs[i], bys[i], bzs[i] = _get_drone_velocity_components(
            xs[i],
            ys[i],
            zs[i],
            c
        )

        t += c.dt

        if any(np.isnan(v) for v in (xs[i], ys[i], zs[i])):
            raise RuntimeError(f"NaN at step {i} (t={t}).")

    return (xs, ys, zs), (bxs, bys, bzs)


def _get_drone_velocity_components(x: Float, y: Float, z: Float, c: Config) -> Tuple[Float, Float, Float]:
    r = np.sqrt(x * x + y * y + z * z)
    inv = 1.0 / r if r > 0.0 else 0.0

    bx = c.drone_speed * x * inv
    by = c.drone_speed * y * inv
    bz = c.drone_speed * z * inv

    return bx, by, bz


def _derivatives(t: Float, x: Float, y: Float, z: Float, c: Config) -> Tuple[Float, Float, Float]:
    bx, by, bz = _get_drone_velocity_components(x, y, z, c)

    dx = c.wind_accel_x(t) - bx
    dy = c.wind_accel_y(t) - by
    dz = c.wind_accel_z(t) - bz

    return dx, dy, dz


def _rk4(t: Float, x: Float, y: Float, z: Float, c: Config) -> Tuple[Float, Float, Float]:
    h = c.dt

    dx1, dy1, dz1 = _derivatives(t, x, y, z, c)
    dx2, dy2, dz2 = _derivatives(
        t + 0.5 * h,
        x + 0.5 * h * dx1,
        y + 0.5 * h * dy1,
        z + 0.5 * h * dz1,
        c)
    dx3, dy3, dz3 = _derivatives(
        t + 0.5 * h,
        x + 0.5 * h * dx2,
        y + 0.5 * h * dy2,
        z + 0.5 * h * dz2,
        c)
    dx4, dy4, dz4 = _derivatives(
        t + h,
        x + h * dx3,
        y + h * dy3,
        z + h * dz3,
        c)

    xn = x + h / 6.0 * (dx1 + 2 * dx2 + 2 * dx3 + dx4)
    yn = y + h / 6.0 * (dy1 + 2 * dy2 + 2 * dy3 + dy4)
    zn = z + h / 6.0 * (dz1 + 2 * dz2 + 2 * dz3 + dz4)

    return xn, yn, zn
