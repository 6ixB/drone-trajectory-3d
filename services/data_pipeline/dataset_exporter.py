from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Any
import numpy as np
import numpy.typing as npt
from netCDF4 import Dataset, Variable
from .types import Float, Array
from .ncdf_ops import pick_slice


@dataclass(frozen=True)
class DatasetExporterOptions:
    input_path: str
    output_directory: str
    output_file_name: str
    station_index: int
    start_time: int
    end_time: int = 86399
    time_var: str = "time"
    spd_var: str = "spd"
    u_var: str = "u"
    v_var: str = "v"
    fill_value: Float = Float(1e37)
    fill_epsilon: Float = Float(1e30)


def save_as_csv(opt: DatasetExporterOptions) -> int:
    if opt.start_time < 0 or opt.end_time < opt.start_time:
        raise ValueError("Invalid index range.")
    if opt.station_index < 0:
        raise ValueError("StationIndex must be >= 0.")
    if opt.fill_epsilon <= 0:
        raise ValueError("FillEpsilon must be > 0.")

    outdir: Path = Path(opt.output_directory)
    outdir.mkdir(parents=True, exist_ok=True)

    fn: str = (
        opt.output_file_name
        if opt.output_file_name.lower().endswith(".csv")
        else opt.output_file_name + ".csv"
    )
    outpath: Path = Path(fn) if Path(fn).is_absolute() else outdir / fn

    with Dataset(opt.input_path, "r") as nc:
        time_var: Variable[Any] = nc.variables[opt.time_var]
        spd_var: Variable[Any] = nc.variables[opt.spd_var]
        u_var: Variable[Any] = nc.variables[opt.u_var]
        v_var: Variable[Any] = nc.variables[opt.v_var]

        time: Array = np.array(time_var[:], dtype=Float)
        if opt.end_time >= time.shape[0]:
            raise IndexError(
                f"EndIndex {opt.end_time} >= time length {time.shape[0]}")
        t0, t1 = opt.start_time, opt.end_time

        spd_raw: Array = pick_slice(
            spd_var, t0, t1, opt.station_index)
        u_raw: Array = pick_slice(u_var, t0, t1, opt.station_index)
        v_raw: Array = pick_slice(v_var, t0, t1, opt.station_index)

        spd: Array = np.asarray(spd_raw, dtype=Float)
        u: Array = np.asarray(u_raw, dtype=Float)
        v: Array = np.asarray(v_raw, dtype=Float)
        tt: Array = time[t0: t1 +
                         1].astype(Float, copy=False)

    fv: Final[Float] = Float(opt.fill_value)
    eps: Final[Float] = Float(opt.fill_epsilon)

    def norm(a: Array) -> Array:
        m: npt.NDArray[np.bool_] = np.isfinite(a) & (np.abs(a - fv) <= eps)
        out: Array = a.astype(Float, copy=True)
        out[m] = np.nan
        return out

    spd = norm(spd)
    u = norm(u)
    v = norm(v)

    data: Array = np.column_stack([tt, spd, u, v])

    with open(outpath, "w", buffering=1 << 20) as f:
        f.write("time,spd,u,v\n")
        np.savetxt(f, data, fmt="%.15g", delimiter=",")

    return int(data.shape[0])
