"""Unit tests for the dof module."""

import numpy as np
from numpy.typing import NDArray
import pytest
from pytest import approx

from kinematic_constraint.dof import (
    DoF,
    Constraint,
    Rotation,
    get_rotation_translation_linear_operator,
    get_translation_linear_operator,
    use_common_point_for_intersecting_lines,
    calc_dofs_basis,
    constraints_allow_dof,
)
from kinematic_constraint.linalg import basis_contains_vector
from kinematic_constraint.geometry import Line3, angle_between_vectors


class TestGetTranslationLinearOperator:
    def test_translation_one_constraint(self):
        linop = get_translation_linear_operator([Constraint((0, 0, 0), (1, 0, 0))])
        assert linop @ np.array([1, 0, 0]) == approx(1.0)
        assert linop @ np.array([0, 1, 0]) == approx(0.0)
        assert linop @ np.array([0, 0, 1]) == approx(0.0)


class TestGetRotationLinearOperator:
    def test_three_constraints_thru_origin(self):
        linop = get_rotation_translation_linear_operator(
            [
                Constraint((1, 0, 0), (1, 0, 0)),
                Constraint((0, 1, 0), (0, 1, 0)),
                Constraint((0, 0, 1), (0, 0, 1)),
            ]
        )

        # No length changes for rotation about the x axis through the origin.
        r = [1, 0, 0]
        p = [0, 0, 0]
        assert linop @ np.concatenate((r, -np.cross(r, p))) == approx(np.zeros(3))
        # No length changes for rotation about the y axis through the origin.
        r = [0, 1, 0]
        p = [0, 0, 0]
        assert linop @ np.concatenate((r, -np.cross(r, p))) == approx(np.zeros(3))
        # No length change for rotation about the z axis through the origin.
        r = [0, 0, 1]
        p = [0, 0, 0]
        assert linop @ np.concatenate((r, -np.cross(r, p))) == approx(np.zeros(3))

        # This rotation should change the length of only the z-aligned constraint
        # because it passes through the contact point of the x-aligned constraint
        # and is parallel to the y-aligned constraint.
        r = [0, 1, 0]
        p = [1, 0, 0]
        dl = linop @ np.concatenate((r, -np.cross(r, p)))
        assert dl[0] == approx(0)
        assert dl[1] == approx(0)
        assert dl[2] > 1e-3


class TestUseCommonPointForIntersectingLines:
    def test_triangle(self):
        # These three lines for a triangle in the x,y plane.
        # They intersect at:
        #  a, b: (0, 0, 0)
        #  a, c: (1, 0, 0)
        #  b, c: (0, 1, 0)
        a = Line3((0, 0, 0), (1, 0, 0))
        b = Line3((0, 1, 0), (0, 1, 0))
        c = Line3((2, -1, 0), (-1, 1, 0))

        use_common_point_for_intersecting_lines([a, b, c])

        # Each line's point should be set to one of it's
        # two intersection points, but it does not matter which one.
        assert a.point == approx(np.zeros(3)) or a.point == approx(np.array([1, 0, 0]))
        assert b.point == approx(np.zeros(3)) or b.point == approx(np.array([0, 1, 0]))
        assert c.point == approx(np.array([1, 0, 0])) or c.point == approx(np.array([0, 1, 0]))

    def test_three_thru_origin(self):
        # These three lines all intersect at the origin.
        a = Line3((2, 0, 0), (1, 0, 0))
        b = Line3((0, 2, 0), (0, 1, 0))
        c = Line3((0, 0, 2), (0, 0, 1))

        use_common_point_for_intersecting_lines([a, b, c])

        assert a.point == approx(np.zeros(3))
        assert b.point == approx(np.zeros(3))
        assert c.point == approx(np.zeros(3))


class TestCalcDofs:
    def test_three_constraints_thru_origin(self):
        dofs = calc_dofs_basis(
            [
                Constraint((1, 0, 0), (1, 0, 0)),
                Constraint((0, 1, 0), (0, 1, 0)),
                Constraint((0, 0, 1), (0, 0, 1)),
            ]
        )
        print(dofs)
        assert len(dofs) == 3
        for dof in dofs:
            assert dof.translation is None
            assert dof.rotation is not None
            assert dof.rotation.point == approx(np.zeros(3))

    def test_three_constraints_thru_111(self):
        constraints = [
            Constraint((1, 1, 1), (1, 0, 0)),
            Constraint((1, 1, 1), (0, 1, 0)),
            Constraint((1, 1, 1), (0, 0, 1)),
        ]
        dofs = calc_dofs_basis(constraints)
        print(dofs)
        assert len(dofs) == 3
        for dof in dofs:
            assert dof.translation is None
            assert dof.rotation is not None
            # Axes of rotation will intersect or be parallel to all constraints.
            for cst in constraints:
                assert cst.shortest_distance_to_line(dof.rotation) < 1e-9

    @pytest.mark.parametrize(
        ["offset"],
        [
            (np.array([0.0, 0.0, 0.0]),),
            (np.array([0.1, 0.2, 0.3]),),
        ],
    )
    @pytest.mark.parametrize(
        ["constraints", "correct_dofs"],
        [
            # 3 R, 3 T
            (
                [],
                [
                    DoF((1, 0, 0), None),
                    DoF((0, 1, 0), None),
                    DoF((0, 0, 1), None),
                    DoF(None, Rotation((0, 0, 0), (1, 0, 0))),
                    DoF(None, Rotation((0, 0, 0), (0, 1, 0))),
                    DoF(None, Rotation((0, 0, 0), (0, 0, 1))),
                ],
            ),
            # 3 R, 2 T
            (
                [
                    Constraint((1, 0, 0), (-1, 0, 0)),
                ],
                [
                    DoF((0, 1, 0), None),
                    DoF((0, 0, 1), None),
                    DoF(None, Rotation((0, 0, 0), (1, 0, 0))),
                    DoF(None, Rotation((0, 0, 0), (0, 1, 0))),
                    DoF(None, Rotation((0, 0, 0), (0, 0, 1))),
                ],
            ),
            # 3 R, 1 T
            (
                [
                    Constraint((1, 0, 0), (-1, 0, 0)),
                    Constraint((0, -1, 0), (0, 1, 0)),
                ],
                [
                    DoF((0, 0, 1), None),
                    DoF(None, Rotation((0, 0, 0), (1, 0, 0))),
                    DoF(None, Rotation((0, 0, 0), (0, 1, 0))),
                    DoF(None, Rotation((0, 0, 0), (0, 0, 1))),
                ],
            ),
            # 3 R, 0 T
            (
                [
                    Constraint((1, 0, 0), (-1, 0, 0)),
                    Constraint((0, -1, 0), (0, 1, 0)),
                    Constraint((0, 0, 1), (0, 0, -1)),
                ],
                [
                    DoF(None, Rotation((0, 0, 0), (1, 0, 0))),
                    DoF(None, Rotation((0, 0, 0), (0, 1, 0))),
                    DoF(None, Rotation((0, 0, 0), (0, 0, 1))),
                ],
            ),
            # 2R, 2 T
            (
                [
                    Constraint((1, 0, -1), (-1, 0, 0)),
                    Constraint((1, 0, 1), (-1, 0, 0)),
                ],
                [
                    DoF((0, 1, 0), None),
                    DoF((0, 0, 1), None),
                    DoF(None, Rotation((0, 0, 0), (1, 0, 0))),
                    DoF(None, Rotation((0, 0, 0), (0, 0, 1))),
                ],
            ),
            # 2 R, 1 T, top variant
            (
                [
                    Constraint((1, 0, 1), (-1, 0, 0)),
                    Constraint((1, 0, -1), (-1, 0, 0)),
                    Constraint((0, -1, 0), (0, 1, 0)),
                ],
                [
                    DoF((0, 0, 1), None),
                    DoF(None, Rotation((0, 0, 0), (1, 0, 0))),
                    DoF(None, Rotation((0, 0, 0), (0, 0, 1))),
                ],
            ),
            # 2 R, 1 T, bottom variant
            (
                [
                    Constraint((1, 1, 0), (-1, 0, 0)),
                    Constraint((1, -1, 0), (-1, 0, 0)),
                    Constraint((0, -1, 0), (0, 1, 0)),
                ],
                [
                    DoF((0, 0, 1), None),
                    DoF(None, Rotation((0, 0, 0), (1, 0, 0))),
                    DoF(None, Rotation((0, 0, 0), (0, 1, 0))),
                ],
            ),
            # 2 R, 0T
            (
                [
                    Constraint((0, 0, 1), (0, 0, -1)),
                    Constraint((1, 0, 1), (-1, 0, 0)),
                    Constraint((1, 0, -1), (-1, 0, 0)),
                    Constraint((0, -1, 0), (0, 1, 0)),
                ],
                [
                    DoF(None, Rotation((0, 0, 0), (1, 0, 0))),
                    DoF(None, Rotation((0, 0, 0), (0, 0, 1))),
                ],
            ),
            # 1 R, 2 T
            (
                [
                    Constraint((1, 1, 1), (-1, 0, 0)),
                    Constraint((1, 1, -1), (-1, 0, 0)),
                    Constraint((1, -1, -1), (-1, 0, 0)),
                ],
                [
                    DoF((0, 1, 0), None),
                    DoF((0, 0, 1), None),
                    DoF(None, Rotation((0, 0, 0), (1, 0, 0))),
                ],
            ),
            # 1 R, 1 T, top variant
            (
                [
                    Constraint((1, 1, 1), (-1, 0, 0)),
                    Constraint((1, 1, -1), (-1, 0, 0)),
                    Constraint((1, -1, -1), (-1, 0, 0)),
                    Constraint((0, -1, 0), (0, 1, 0)),
                ],
                [
                    DoF((0, 0, 1), None),
                    DoF(None, Rotation((0, 0, 0), (1, 0, 0))),
                ],
            ),
            # 1 R, 1 T, middle variant
            (
                [
                    Constraint((1, 0, 1), (-1, 0, 0)),
                    Constraint((1, 0, -1), (-1, 0, 0)),
                    Constraint((1, -1, 0), (0, 1, 0)),
                    Constraint((-1, -1, 0), (0, 1, 0)),
                ],
                [
                    DoF((0, 0, 1), None),
                    DoF(None, Rotation((0, 0, 0), (1, 0, 0))),
                ],
            ),
            # 1 R, 1 T, bottom variant
            (
                [
                    Constraint((1, 0, 1), (-1, 0, 0)),
                    Constraint((1, 0, -1), (-1, 0, 0)),
                    Constraint((0, -1, 1), (0, 1, 0)),
                    Constraint((0, -1, -1), (0, 1, 0)),
                ],
                [
                    DoF((0, 0, 1), None),
                    DoF(None, Rotation((0, 0, 0), (0, 0, 1))),
                ],
            ),
            # 1 R, 0 T, top variant
            (
                [
                    Constraint((0, 0, 1), (0, 0, -1)),
                    Constraint((1, 1, 1), (-1, 0, 0)),
                    Constraint((1, 1, -1), (-1, 0, 0)),
                    Constraint((1, -1, -1), (-1, 0, 0)),
                    Constraint((0, -1, 0), (0, 1, 0)),
                ],
                [
                    DoF(None, Rotation((0, 0, 0), (1, 0, 0))),
                ],
            ),
            # 1 R, 0 T, middle variant
            (
                [
                    Constraint((0, 0, 1), (0, 0, -1)),
                    Constraint((1, 0, 1), (-1, 0, 0)),
                    Constraint((1, 0, -1), (-1, 0, 0)),
                    Constraint((1, -1, 0), (0, 1, 0)),
                    Constraint((-1, -1, 0), (0, 1, 0)),
                ],
                [
                    DoF(None, Rotation((0, 0, 0), (1, 0, 0))),
                ],
            ),
            # 1 R, 0 T, bottom variant
            (
                [
                    Constraint((0, 0, 1), (0, 0, -1)),
                    Constraint((1, 0, 1), (-1, 0, 0)),
                    Constraint((1, 0, -1), (-1, 0, 0)),
                    Constraint((0, -1, 1), (0, 1, 0)),
                    Constraint((0, -1, -1), (0, 1, 0)),
                ],
                [
                    DoF(None, Rotation((0, 0, 0), (0, 0, 1))),
                ],
            ),
            # 0 R, 1 T
            (
                [
                    Constraint((1, 1, 1), (-1, 0, 0)),
                    Constraint((1, 1, -1), (-1, 0, 0)),
                    Constraint((1, -1, -1), (-1, 0, 0)),
                    Constraint((0, -1, -1), (0, 1, 0)),
                    Constraint((0, -1, 1), (0, 1, 0)),
                ],
                [DoF((0, 0, 1), None)],
            ),
            # 0 R, 0 T, top variant
            (
                [
                    Constraint((0, 0, 1), (0, 0, -1)),
                    Constraint((1, 1, 1), (-1, 0, 0)),
                    Constraint((1, 1, -1), (-1, 0, 0)),
                    Constraint((1, -1, -1), (-1, 0, 0)),
                    Constraint((0, -1, -1), (0, 1, 0)),
                    Constraint((0, -1, 1), (0, 1, 0)),
                ],
                [],
            ),
            # 0 R, 0 T, bottom variant
            (
                [
                    Constraint((0, 1, 1), (0, 0, -1)),
                    Constraint((1, 0, 1), (-1, 0, 0)),
                    Constraint((1, 0, -1), (-1, 0, 0)),
                    Constraint((1, -1, 0), (0, 1, 0)),
                    Constraint((-1, -1, 0), (0, 1, 0)),
                    Constraint((0, -1, 1), (0, 0, -1)),
                ],
                [],
            ),
        ],
    )
    def test_hale_2_21(
        self, constraints: list[Constraint], correct_dofs: list[DoF], offset: NDArray
    ):
        """Test on the example cases shown in Figure 2-21 of Hale [1].

        References:
            [1] Layton C. Hale, "Principles and techniques for designing precision machines,"
                 Thesis, Massachusetts Institute of Technology, 1999. Accessed: Jun. 28, 2022.
                 [Online]. Available: https://dspace.mit.edu/handle/1721.1/9414
        """
        correct_translations = [
            dof.translation for dof in correct_dofs if dof.translation is not None
        ]
        correct_rotations = [dof.rotation for dof in correct_dofs if dof.rotation is not None]
        # In these examples, none of the constraints are redundant, so number of dofs + number of constraints is 6.
        # Check that we got this right when writing down the correct DoFs.
        assert len(correct_dofs) + len(constraints) == 6

        # Apply the offset
        for cst in constraints:
            cst.point += offset
        for correct_rotation in correct_rotations:
            correct_rotation.point += offset

        for constraint in constraints:
            print(str(constraint))

        dofs = calc_dofs_basis(constraints)
        for dof in dofs:
            print(dof)

        assert len(dofs) == len(correct_dofs)
        for dof in dofs:
            assert dof.translation is None or dof.rotation is None
        translations = [dof.translation for dof in dofs if dof.translation is not None]
        rotations = [dof.rotation for dof in dofs if dof.rotation is not None]

        assert len(translations) == len(correct_translations)
        assert len(rotations) == len(correct_rotations)

        # There should be exactly one translation equal to each correct translation.
        for correct_translation in correct_translations:
            equal_count = 0
            for translation in translations:
                if translation == approx(correct_translation):
                    equal_count += 1
            assert equal_count == 1

        # "The axes of a body's rotational degrees of freedom will each intersect
        # all constraints applied to the body"
        # -- Hale, Section 2.6, Statement 4.
        # Being parallel to a constraint line counts as intersecting it "at infinity".
        for rotation in rotations:
            for constraint in constraints:
                assert constraint.parallel_or_intersecting(rotation)
        # For good measure, check that we go this right when writing down the correct DoFs.
        for rotation in correct_rotations:
            for constraint in constraints:
                print(f"angle = {angle_between_vectors(constraint.direction, rotation.direction)}")
                print(f"dist  = {constraint.shortest_distance_to_line(rotation)}")
                assert constraint.parallel_or_intersecting(rotation)

        for rotation in rotations:
            if len(constraints) == 0:
                # Although the rotations could be about any point, the origin is the most intuitive.
                assert rotation.point == approx(np.zeros(3))
            elif len(constraints) == 1:
                # Although the rotations could be about any point along the single constraint line,
                # the specified constraint point is most intuitive.
                assert rotation.point == approx(constraints[0].point)
            elif len(correct_translations) == 0 and len(correct_rotations) > 1:
                # All the rotations should be through the offset point.
                assert rotation.point == approx(offset)
            # If there are translation degrees of freedom, there will be many valid centers
            # for some of the rotations, so don't test the centers.
            #
            # If there are no translation dofs and more than one rotation dof,
            # the simplification should move the center points of all the rotation dofs
            # to the point of intersection of their lines, which for these examples should
            # be the offset point.

        # The basis of rotation directions should contain each correct rotation direction.
        rotation_direction_basis = [rotation.direction for rotation in rotations]
        for correct_rotation in correct_rotations:
            assert basis_contains_vector(rotation_direction_basis, correct_rotation.direction)

    def test_helical_dof(self):
        """Test on a system of constraints with a single degree of freedom, which is a coupled
        rotation and translation, i.e. a helical motion.

        The arrangement of constraints is similar to that shown in Figure 2-25 of Hale.
        However, the 3 constraints shown in Figure 2-25 leave 3 degrees of freedom,
        which may be any three rotations whose lines are generators of a hyperboloid,
        as shown in Figure 6.4.12 of Blanding.

        Instead, the test here adds two additional constraints, whose line of constraint
        intersects the z axis. This removes all degrees of freedom except one, because
        there are now five non-redundant constraints. The remaining degree of freedom
        is a helical motion about the z-axis.
        """
        # Setup
        constraints = []
        x0 = 1.0
        y0 = 0.1
        correct_pitch = 0.1
        for theta in [0.0, 2 / 3 * np.pi, 4 / 3 * np.pi]:
            constraints.append(
                Constraint(
                    point=(
                        x0 * np.cos(theta) - y0 * np.sin(theta),
                        x0 * np.sin(theta) + y0 * np.cos(theta),
                        0,
                    ),
                    direction=(np.cos(theta), np.sin(theta), 1),
                )
            )
        for theta in [0.0, 2 / 3 * np.pi]:
            x = x0 * np.cos(theta) - y0 * np.sin(theta)
            y = x0 * np.sin(theta) + y0 * np.cos(theta)
            constraints.append(Constraint((x, y, 0), (-x, -y, 0)))

        # Check the setup
        # The constraints should allow a helical motion about the z axis.
        assert constraints_allow_dof(
            constraints,
            DoF(
                translation=(0, 0, 1), rotation=Rotation((0, 0, 0), (0, 0, 1)), pitch=correct_pitch
            ),
        )
        # The constraints should not allow pure translation about the z axis, nor pure rotation the about z axis.
        assert not constraints_allow_dof(constraints, DoF(translation=(0, 0, 1), rotation=None))
        assert not constraints_allow_dof(
            constraints, DoF(translation=None, rotation=Rotation((0, 0, 0), (0, 0, 1)))
        )

        # Action
        dofs = calc_dofs_basis(constraints)

        # Verification
        assert len(dofs) == 1
        dof = dofs[0]
        assert dof.translation == approx(np.array([0, 0, 1]))
        assert dof.rotation is not None
        assert dof.rotation.direction == approx(np.array([0, 0, 1]))
        assert dof.rotation.point == approx(np.zeros(3))
        assert dof.pitch == approx(correct_pitch)
