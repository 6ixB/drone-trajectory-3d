from dataclasses import dataclass
from data.types import Float, Function


@dataclass(frozen=True)
class Config:
    wind_accel_x: Function
    wind_accel_y: Function
    wind_accel_z: Function
    drone_speed: Float
    dt: Float
    steps: int
    x0: Float
    y0: Float
    z0: Float
    t0: Float = Float(0.0)
