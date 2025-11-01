from enum import Enum
from services.data_pipeline.types import Float, Function
import numpy as np


class TrigFuncType(Enum):
    SIN = "sin"
    COS = "cos"


def linear(a: Float) -> Function:
    return lambda t: a * t


def trig(a: Float, alpha: Float, trig_func_type: TrigFuncType) -> Function:
    if trig_func_type is TrigFuncType.SIN:
        return lambda t: a * np.sin(alpha * t)
    if trig_func_type is TrigFuncType.COS:
        return lambda t: a * np.cos(alpha * t)
    raise ValueError("Invalid TrigFuncType")


def exp(a: Float, alpha: Float, k: Float, trig_func_type: TrigFuncType) -> Function:
    if trig_func_type is TrigFuncType.SIN:
        return lambda t: a * np.exp(k * t) * np.sin(alpha * t)
    if trig_func_type is TrigFuncType.COS:
        return lambda t: a * np.exp(k * t) * np.cos(alpha * t)
    raise ValueError("Invalid TrigFuncType")
