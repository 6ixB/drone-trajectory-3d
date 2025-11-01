from typing import Callable
import numpy as np
import numpy.typing as npt


Float = np.float64
Array = npt.NDArray[Float]
Function = Callable[[Float], Float]
