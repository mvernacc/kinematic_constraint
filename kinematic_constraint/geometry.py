"""3d geometry helpers."""

from numpy.typing import NDArray
import numpy as np

Vec3 = tuple[float | int, float | int, float | int] | NDArray


def unit(x: Vec3) -> NDArray:
    x_ = np.array(x, dtype=np.float64)
    return x_ / np.linalg.norm(x_)


def check_len_3(x: Vec3, name: str = "vector"):
    if isinstance(x, np.ndarray) and x.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {x.shape}")
    elif len(x) != 3:
        raise ValueError(f"{name} must have length 3, got {len(x)}")


def angle_between_vectors(a: NDArray, b: NDArray) -> float:
    """The angle between two 3-vectors in radians."""
    return float(np.arccos(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))


class Line3:
    def __init__(self, point: Vec3, direction: Vec3) -> None:
        """A line in 3d space, represented by a point and a direction."""
        self.point = point
        self.direction = direction

    @property
    def point(self) -> NDArray:
        """A point the line passes through, with shape `(3,)` and units of length."""
        return self._point

    @point.setter
    def point(self, value: Vec3):
        check_len_3(value, "point")
        self._point = np.array(value, dtype=np.float64)

    @property
    def direction(self) -> NDArray:
        """The direction of the line, with shape `(3,)` and dimensionless unit length."""
        return self._direction

    @direction.setter
    def direction(self, value: Vec3):
        check_len_3(value, "direction")
        self._direction = unit(np.array(value, dtype=np.float64))

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(point=({self.point[0]}, {self.point[1]}, {self.point[2]}), direction=({self.direction[0]}, {self.direction[1]}, {self.direction[2]}))"

    def shortest_distance_to_point(self, p: Vec3) -> float:
        # https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        p = np.array(p, dtype=np.float64)
        # The norm of `line.direction`` is 1, so we don't need to divide by it.
        return float(np.linalg.norm(np.cross(self.direction, self.point - p)))

    def shortest_distance_to_line(self, other: "Line3") -> float:
        """Calculate the shortest distance between two 3d lines."""
        d1xd2 = np.cross(self.direction, other.direction)
        d1xd2_norm = np.linalg.norm(d1xd2)
        if d1xd2_norm < 1e-9:
            # Lines are almost parallel, avoid dividing by zero.
            # N.B. the norm of the directions is 1.
            return float(np.linalg.norm(np.cross(self.direction, (other.point - self.point))))
        return float(np.linalg.norm(d1xd2 @ (other.point - self.point)) / d1xd2_norm)

    def parallel_or_intersecting(
        self,
        other: "Line3",
        angle_tol: float = 1e-12,
        distance_tol: float = 1e-12,
    ) -> bool:
        a = angle_between_vectors(self.direction, other.direction)
        if a <= angle_tol or abs(a - np.pi) <= angle_tol:
            return True  # The lines are parallel (to within the tolerance)
        if self.shortest_distance_to_line(other) <= distance_tol:
            return True  # The lines intersect (to within the tolerance)
        return False

    def coincident(
        self,
        other: "Line3",
        angle_tol: float = 1e-12,
        distance_tol: float = 1e-12,
    ) -> bool:
        a = angle_between_vectors(self.direction, other.direction)
        if a >= angle_tol and abs(a - np.pi) >= angle_tol:
            return False
        return self.shortest_distance_to_point(other.point) <= distance_tol

    def closest_points(self, other: "Line3") -> tuple[NDArray, NDArray]:
        """Calculate the points where two 3d lines are closest to each other.

        Returns two arrays:
        * The point on line `self` closest to line `other`.
        * The point on line `other` closest to line `self`.

        Both are of shape `(3,)` and in the same length units as the `point`s of the lines.

        If the lines are parallel, returns an arbitrary pair of points on the lines.
        """
        d1xd2 = np.cross(self.direction, other.direction)
        d1xd2_mag2 = d1xd2 @ d1xd2
        if d1xd2_mag2 < 1e-12:
            ta = 0.0
            # Formula from https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
            # The magnitude of b.direction is 1, so we don't need to divide by it.
            tb = -(other.point - self.point) @ other.direction
        else:
            # Formula from https://math.stackexchange.com/a/4764188
            ta = np.cross((other.point - self.point), other.direction) @ d1xd2 / d1xd2_mag2
            tb = np.cross((other.point - self.point), self.direction) @ d1xd2 / d1xd2_mag2
        closest_a = self.point + ta * self.direction
        closest_b = other.point + tb * other.direction
        return closest_a, closest_b
