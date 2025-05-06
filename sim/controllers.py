from src.utils.array_utils import ArrayType

class PositionController:
    """A class for controlling the position of a robot's joints."""

    def step(self, q: ArrayType, q_dot: ArrayType, a: ArrayType):
        """Advances the system state by one time step using the provided acceleration.

        Args:
            q (ArrayType): The current state vector of the system.
            q_dot (ArrayType): The current velocity vector of the system.
            a (ArrayType): The acceleration vector to be applied.

        Returns:
            ArrayType: The acceleration vector `a`.
        """
        return a