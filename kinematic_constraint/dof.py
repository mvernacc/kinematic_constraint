"""Models for kinematic constraint and degrees of freedom of a 3d rigid body."""

import copy
from dataclasses import dataclass

from numpy.typing import NDArray
import numpy as np
from scipy.linalg import null_space

from .geometry import Line3, unit, check_len_3, Vec3, angle_between_vectors
from .linalg import orthogonal_subspace


class Constraint(Line3):
    """A constraint line which connects to a rigid body at a point, and prevents the connection
    point from moving along the direction of the constraint.

    `point` is where the constraint connects to the object.

    `direction` is a unit vector from the remote end of the constraint to the connection point.

    This definition of a constraint is from Section 1.3 of Blanding [1].

    References:
        [1] D. Blanding, Exact Constraint: Machine Design Using Kinematic Processing.
            New York: American Society of Mechanical Engineers, 1999.
    """

    pass


class Rotation(Line3):
    """An axis about which a body can rotate.

    `point` is a point through which the line of rotation passes.

    `direction` is a unit vector along the axis of rotation.
    """

    pass


class DoF:
    def __init__(
        self, translation: Vec3 | None, rotation: Rotation | None, pitch: float | None = None
    ):
        """A degree of freedom of a rigid body.

        This may be a pure translation, a pure rotation, or a rotation and translation that are
        coupled in a helical motion.
        """
        if translation is not None:
            check_len_3(translation, "translation")
            self._translation = unit(np.array(translation, dtype=np.float64))
        else:
            self._translation = None
        self._rotation = rotation
        if pitch is not None and (translation is None or rotation is None):
            raise ValueError("Both translation and rotation must be provided if pitch is provided.")
        self._pitch = pitch

    @property
    def translation(self) -> NDArray | None:
        return self._translation

    @property
    def rotation(self) -> Rotation | None:
        return self._rotation

    @property
    def pitch(self) -> float | None:
        """The ratio of translation to rotation for a coupled DoF, in units of length per radian."""
        return self._pitch

    def to_screw(self) -> NDArray:
        """Create a screw-like 6 vector that represents this degree of freedom.

        The screw is:
        $$
        \\vec{dm} = [\\vec{r} \\, d\\theta, \\quad \\vec{t} \\, ds - \\vec{r} \\times \\vec{p} \\, d\\theta]
        $$

        The magnitude of the 6-vector is arbitrarily set to 1.
        """
        r = np.zeros(3) if self.rotation is None else self.rotation.direction
        p = np.zeros(3) if self.rotation is None else self.rotation.point
        t = np.zeros(3) if self.translation is None else self.translation
        pitch = 1.0 if self.pitch is None else self.pitch
        screw = np.concatenate((r, pitch * t - np.cross(r, p)))
        return screw / np.linalg.norm(screw)

    def __str__(self) -> str:
        translation_str = (
            "None"
            if self.translation is None
            else f"({self.translation[0]}, {self.translation[1]}, {self.translation[2]})"
        )
        return f"{self.__class__.__name__}(translation={translation_str}, rotation={self.rotation}, pitch={self.pitch})"


def get_translation_linear_operator(constraints: list[Constraint]) -> NDArray:
    """Get a linear operator which maps translations of the body -> changes in the constraint lengths.

    The sign convention for length changes is that motion in the constraint direction is a positive
    length change.

    The null space of this operator represents the translational degrees of freedom of the body,
    if any exist.
    """
    return np.array([unit(cst.direction) for cst in constraints])


def get_rotation_translation_linear_operator(constraints: list[Constraint]) -> NDArray:
    """Get a linear operator which maps differential rotations and translations of a body
    -> changes in the constraint lengths.

    The domain (input) for this operator is screw-like 6-vectors of the form
    [r dtheta, t ds - (r x p) dtheta]

    where
    * r is the direction of rotation (unit 3-vector),
    * p is the point about which the body is rotated,
    * t is the direction of translation (unit 3-vector),
    * dtheta is the differential rotation angle,
    * ds is the differential translation length

    The range (output) for this operator is the differential length change of each constraint
    due to the motion. The sign convention for length changes is that motion in the constraint
    direction is a positive length change.

    This linearization is only valid for very small rotations, i.e. dtheta << 1.

    The null space of this operator represents the degrees of freedom of the body, if any exist.

    Args:
        constraints: Constraints applied to the body.

    Returns:
        The $A$ matrix of shape `(len(constraints), 6)`
    """
    return np.array(
        [
            np.concatenate((np.cross(cst.point, unit(cst.direction)), unit(cst.direction)))
            for cst in constraints
        ]
    )


def calc_dofs_basis(constraints: list[Constraint], simplify: bool = True) -> list[DoF]:
    """Calculate a basis for the translational and rotational degrees of freedom of a constrained rigid body.

    This function models a rigid body supported by point-contact "constraint lines",
    as defined in Blanding [1].

    References:
        [1] D. Blanding, Exact Constraint: Machine Design Using Kinematic Processing.
            New York: American Society of Mechanical Engineers, 1999.
    """
    n_constraints = len(constraints)
    if n_constraints == 0:
        return [
            DoF(translation=(1, 0, 0), rotation=None),
            DoF(translation=(0, 1, 0), rotation=None),
            DoF(translation=(0, 0, 1), rotation=None),
            DoF(translation=None, rotation=Rotation(point=(0, 0, 0), direction=(1, 0, 0))),
            DoF(translation=None, rotation=Rotation(point=(0, 0, 0), direction=(0, 1, 0))),
            DoF(translation=None, rotation=Rotation(point=(0, 0, 0), direction=(0, 0, 1))),
        ]

    if len(constraints) == 1 or all(
        constraints[0].coincident(constraints[i]) for i in range(1, len(constraints))
    ):
        # Special case: either there is only one constraint, or all the constraints are coincident.
        # In this case, there are 3 rotation DoFs and 2 translation DoFs.
        # The rotation DoFs are ambiguous, as they could be any three orthogonal lines which
        # all intersect the one constraint line.
        # Just choose an intuitive set as a special case.
        translation_basis = orthogonal_subspace(
            [(1, 0, 0), (0, 1, 0), (0, 0, 1)], constraints[0].direction
        )
        return [
            DoF(translation_basis[:, 0], None),
            DoF(translation_basis[:, 1], None),
            DoF(
                translation=None, rotation=Rotation(point=constraints[0].point, direction=(1, 0, 0))
            ),
            DoF(
                translation=None, rotation=Rotation(point=constraints[0].point, direction=(0, 1, 0))
            ),
            DoF(
                translation=None, rotation=Rotation(point=constraints[0].point, direction=(0, 0, 1))
            ),
        ]

    linop_rt = get_rotation_translation_linear_operator(constraints)

    # Calculate an orthonormal basis for the degrees of freedom as the null space of
    # the linear operator that maps screw-like 6-vectors -> constraint length changes.
    # The null space contains the differential motion 6-vectors which
    # result in zero constraint length change, meaning those motions
    # are allowed by the constraints.
    basis = null_space(linop_rt)

    assert basis.shape[0] == 6
    n_dof = basis.shape[1]
    # Each constraint removes at most 1 degree of freedom.
    assert n_dof >= 6 - n_constraints

    # Interpret the screw-like 6-vectors of the basis into easier-to-read DoF objects.
    # First, choose negligibly small values of length and angle. These are used below instead
    # of comparing to zero to allow for numerical error in the calculation of the basis vectors.
    angle_epsilon = 1e-9  # [radian]
    max_dist = _max_distance_between_points(constraints)
    length_epsilon = 1e-9 if max_dist == 0.0 else 1e-9 * max_dist  # [length units]
    dofs = []
    for i in range(n_dof):
        col = basis[:, i]

        translation = None
        rotation = None
        pitch = None
        if np.linalg.norm(col[:3]) < angle_epsilon:
            # Pure translation
            translation = col[3:]
        else:
            direction = unit(col[:3])
            point, residual = _calc_rotation_point(constraints, direction)
            rotation = Rotation(point=point, direction=direction)
            if residual > length_epsilon:
                # `_calc_rotation_point` could not find a pure rotation about this axis
                # which gives zero length change for all constraints.
                # The remainder represents that a translation is irreducibly coupled
                # to this rotational degree of freedom.
                remainder = col[3:] + np.cross(col[:3], point)
                # We expect the remainder to be parallel to the direction of rotation,
                # i.e. the coupled rotation and translation is a helical motion.
                if angle_between_vectors(remainder, direction) > angle_epsilon:
                    raise RuntimeError(
                        f"For column {i}, remainder {remainder} is not parallel to rotation direction {direction}."
                        " This is unexpected."
                    )
                translation = remainder
                # pitch is ds/dtheta. The magnitude of remainder is ds, and the magnitude of col[:3] is  dtheta.
                pitch = float(np.linalg.norm(remainder) / np.linalg.norm(col[:3]))
        dofs.append(DoF(translation=translation, rotation=rotation, pitch=pitch))

    if simplify:
        dofs = simplify_dofs(dofs, distance_tol=length_epsilon)
    return dofs


def constraints_allow_dof(constraints: list[Constraint], dof: DoF) -> bool:
    """Check if a set of constraints allow (infinitesimal) motion of along a degree of freedom."""
    linop_rt = get_rotation_translation_linear_operator(constraints)
    dlengths = linop_rt @ dof.to_screw()

    # Calculate a length scale that is small compared to the distance between the constraint points.
    max_dist = _max_distance_between_points(constraints)
    length_epsilon = 1e-9 if max_dist == 0.0 else 1e-9 * max_dist  # [length units]

    # Compare the `dlengths` to zero while allowing for some numerical error.
    return bool(np.all(np.abs(dlengths) < length_epsilon))


def _calc_rotation_point(constraints: list[Constraint], axis: Vec3) -> tuple[NDArray, float]:
    """Given a set of constraints and a rotation axis,
    solve for the point about which the rotation will not change the length
    of any constraint.

    Returns:
        * point
        * residual, the 2-norm of the constraint length changes for a rotation about
          point p. A non-zero residual indicates that a pure rotation could not be found.
    """
    A_p = np.array([np.cross(unit(cst.direction), axis) for cst in constraints])
    b_p = np.array([np.cross(cst.point, unit(cst.direction)) @ axis for cst in constraints])

    p, _, _, _ = np.linalg.lstsq(A_p, b_p)

    # Calculate the residual to check if we got a good solution.
    # Surprisingly the `residual` returned by `np.linalg.lstsq` is sometimes
    # empty even though there is a significant residual!
    residual = np.linalg.norm(b_p - A_p @ p)

    return p, float(residual)


def _max_distance_between_points(constraints: list[Constraint]) -> float:
    max_dist = 0.0
    for i in range(len(constraints)):
        for j in range(i + 1, len(constraints)):
            dist = float(np.linalg.norm(constraints[i].point - constraints[j].point))
            if dist >= max_dist:
                max_dist = dist
    return max_dist


def simplify_dofs(dofs: list[DoF], distance_tol: float = 1e-9) -> list[DoF]:
    """Convert one set of degrees of freedom into an equivalent set,
    which a human may find more intuitive.
    """
    new_dofs = copy.deepcopy(dofs)
    use_common_point_for_intersecting_lines(
        [dof.rotation for dof in new_dofs if dof.rotation is not None], tol=distance_tol
    )
    # Round
    n_digits_length = int(np.ceil(np.log10(1 / distance_tol)))
    n_digits_unit = 9
    for i, dof in enumerate(new_dofs):
        t = dof.translation
        translation = None if t is None else np.round(t, n_digits_unit)
        rotation = None
        if dof.rotation is not None:
            r = dof.rotation.direction
            p = dof.rotation.point
            rotation = Rotation(np.round(p, n_digits_length), np.round(r, n_digits_unit))
        pitch = None if dof.pitch is None else round(dof.pitch, n_digits_length)
        new_dofs[i] = DoF(translation, rotation, pitch)

    return new_dofs


@dataclass
class _PointAndIndexSet:
    point: NDArray
    indexes: set[int]


def use_common_point_for_intersecting_lines(lines: list[Line3], tol: float = 1e-9):
    """For any sub-set of lines which intersect, set their `point`s to be the intersection point.

    Modifies the argument in-place.
    """
    # (intersection point, indexes of lines which intersect that point)
    intersections: list[_PointAndIndexSet] = []

    # Find all points where two lines intersect.
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            pa, pb = lines[i].closest_points(lines[j])
            if np.linalg.norm(pa - pb) < tol:
                # The lines intersect to within the tolerance.
                intersections.append(_PointAndIndexSet((pa + pb) / 2, set([i, j])))

    if len(intersections) == 0:
        return

    # Simplify the list of intersections by grouping close points together.
    while len(intersections) > 1:
        # Find the pair of intersections which are closest to each other.
        closest_dist = np.inf
        closest_pair = None
        for i in range(len(intersections)):
            for j in range(i + 1, len(intersections)):
                dist = np.linalg.norm(intersections[i].point - intersections[j].point)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_pair = (i, j)
        if closest_dist > tol:
            break
        assert closest_pair is not None
        # Merge the closest pair of intersections.
        a = intersections.pop(closest_pair[0])
        b = intersections.pop(closest_pair[1] - 1)  # we changed the indexes with the previous pop
        intersections.append(_PointAndIndexSet((a.point + b.point) / 2, a.indexes.union(b.indexes)))

    for intersection in intersections:
        for i in intersection.indexes:
            lines[i].point = intersection.point
