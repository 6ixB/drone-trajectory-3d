from __future__ import annotations
from dataclasses import dataclass
from typing import Final, Any
from .types import Float, Array
import numpy as np
from netCDF4 import Dataset, Variable
from .utility import pick_slice, norm_fill


@dataclass(frozen=True)
class DatasetLoaderOptions:
    input_path: str
    station_index: int
    start_time: int
    end_time: int = 86399
    time_var: str = "time"
    spd_var: str = "spd"
    u_var: str = "u"
    v_var: str = "v"
    fill_value: Float = Float(1e37)
    fill_epsilon: Float = Float(1e30)


def load_dataset(opt: DatasetLoaderOptions) -> tuple[Array, Array, Array, Array]:
    if not opt.input_path.strip():
        raise ValueError("InputPath required.")
    if opt.start_time < 0 or opt.end_time < 0 or opt.end_time < opt.start_time:
        raise ValueError("Invalid index range.")
    if opt.station_index < 0:
        raise ValueError("StationIndex must be >= 0.")
    if opt.fill_epsilon <= 0:
        raise ValueError("FillEpsilon must be > 0.")

    with Dataset(opt.input_path, "r") as nc:
        time_var: Variable[Any] = nc.variables[opt.time_var]
        time_all: Array = np.asarray(time_var[:], dtype=Float)

        if opt.end_time >= time_all.shape[0]:
            raise IndexError(
                f"EndIndex {opt.end_time} >= time length {time_all.shape[0]}.")

        t0, t1 = opt.start_time, opt.end_time
        ts: Array = time_all[t0: t1 + 1]

        spd: Array = pick_slice(
            nc.variables[opt.spd_var], t0, t1, opt.station_index)
        u:   Array = pick_slice(
            nc.variables[opt.u_var],   t0, t1, opt.station_index)
        v:   Array = pick_slice(
            nc.variables[opt.v_var],   t0, t1, opt.station_index)

    fv: Final[Float] = opt.fill_value
    eps: Final[Float] = opt.fill_epsilon
    spd = norm_fill(spd, fv, eps)
    u = norm_fill(u,   fv, eps)
    v = norm_fill(v,   fv, eps)

    base: Array = spd * spd - u * u - v * v
    base = np.where(np.isnan(spd) | np.isnan(u) | np.isnan(v), np.nan, base)
    ws: Array = np.sqrt(np.maximum(base, 0.0))

    return ts, u, v, ws
