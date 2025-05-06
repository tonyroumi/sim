from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt

@dataclass
class Obs:
    """Observation data structure"""

    lin_vel: npt.NDArray[np.float32]
    gyro: npt.NDArray[np.float32]
    gravity: npt.NDArray[np.float32]
    joint_angles: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(12, dtype=np.float32)
    )
    joint_vel: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(12, dtype=np.float32)
    )
