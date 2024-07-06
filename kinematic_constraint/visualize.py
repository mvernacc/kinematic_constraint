"""Visualize 3d constraints and degrees of freedom."""

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .dof import Constraint, DoF

CONSTRAINT_LINEWIDTH = 2.0
CONSTRAINT_EXTENDED_LINEWIDTH = 0.5
CONSTRAINT_MARKERSIZE = 10.0
DOF_LINEWIDTH = 1.0
DOF_EXTENDED_LINEWIDTH = 0.5
DOF_MARKERSIZE = 10.0
DOF_TRANSLATION_LENGTH = 0.5


def draw_constraint_three_view(
    top_xy: Axes,
    front_xz: Axes,
    right_yz: Axes,
    constraint: Constraint,
):
    for ax, i, j in ((top_xy, 0, 1), (front_xz, 0, 2), (right_yz, 1, 2)):
        if not (abs(constraint.direction[i]) < 1e-9 and abs(constraint.direction[j]) < 1e-9):
            ax.axline(
                (constraint.point[i], constraint.point[j]),
                (
                    constraint.point[i] + constraint.direction[i],
                    constraint.point[j] + constraint.direction[j],
                ),
                linestyle="--",
                linewidth=CONSTRAINT_EXTENDED_LINEWIDTH,
                color="gray",
            )
        ax.plot(
            [
                constraint.point[i],
                constraint.point[i] - constraint.direction[i],
            ],
            [
                constraint.point[j],
                constraint.point[j] - constraint.direction[j],
            ],
            linewidth=CONSTRAINT_LINEWIDTH,
            color="black",
        )
        ax.plot(
            [constraint.point[i]],
            [constraint.point[j]],
            markersize=CONSTRAINT_MARKERSIZE,
            marker=".",
            color="black",
        )


def draw_constraint_3d(ax: Axes3D, constraint: Constraint):
    ax.plot(
        [constraint.point[0], constraint.point[0] - constraint.direction[0]],
        [constraint.point[1], constraint.point[1] - constraint.direction[1]],
        [constraint.point[2], constraint.point[2] - constraint.direction[2]],
        linewidth=CONSTRAINT_LINEWIDTH,
        color="black",
    )
    ax.plot(
        [constraint.point[0]],
        [constraint.point[1]],
        [constraint.point[2]],
        markersize=CONSTRAINT_MARKERSIZE,
        marker=".",
        color="black",
    )


def draw_dof_three_view(
    top_xy: Axes,
    front_xz: Axes,
    right_yz: Axes,
    dof: DoF,
    color: str = "gray",
):
    if dof.translation is not None:
        for ax, i, j in ((top_xy, 0, 1), (front_xz, 0, 2), (right_yz, 1, 2)):
            if abs(dof.translation[i]) < 1e-9 and abs(dof.translation[j]) < 1e-9:
                ax.plot([0], [0], marker=".", color=color, markersize=DOF_MARKERSIZE)
            else:
                ax.arrow(
                    0.0,
                    0.0,
                    DOF_TRANSLATION_LENGTH * dof.translation[i],
                    DOF_TRANSLATION_LENGTH * dof.translation[j],
                    color=color,
                    head_width=0.05,
                    length_includes_head=True,
                )
    if dof.rotation is not None:
        for ax, i, j in ((top_xy, 0, 1), (front_xz, 0, 2), (right_yz, 1, 2)):
            ax.plot(
                [dof.rotation.point[i], dof.rotation.point[i] + dof.rotation.direction[i]],
                [dof.rotation.point[j], dof.rotation.point[j] + dof.rotation.direction[j]],
                linewidth=DOF_LINEWIDTH,
                markersize=DOF_MARKERSIZE,
                marker="+",
                color=color,
            )
            if not (
                abs(dof.rotation.direction[i]) < 1e-9 and abs(dof.rotation.direction[j]) < 1e-9
            ):
                ax.axline(
                    (dof.rotation.point[i], dof.rotation.point[j]),
                    (
                        dof.rotation.point[i] + dof.rotation.direction[i],
                        dof.rotation.point[j] + dof.rotation.direction[j],
                    ),
                    linestyle="--",
                    linewidth=DOF_EXTENDED_LINEWIDTH,
                    color=color,
                )


def draw_dof_3d(
    ax: Axes3D,
    dof: DoF,
    color: str = "gray",
):
    if dof.translation is not None:
        ax.quiver(
            0.0,
            0.0,
            0.0,
            DOF_TRANSLATION_LENGTH * dof.translation[0],
            DOF_TRANSLATION_LENGTH * dof.translation[1],
            DOF_TRANSLATION_LENGTH * dof.translation[2],
            color=color,
        )
    if dof.rotation is not None:
        ax.plot(
            [dof.rotation.point[0], dof.rotation.point[0] + dof.rotation.direction[0]],
            [dof.rotation.point[1], dof.rotation.point[1] + dof.rotation.direction[1]],
            [dof.rotation.point[2], dof.rotation.point[2] + dof.rotation.direction[2]],
            linewidth=DOF_LINEWIDTH,
            markersize=DOF_MARKERSIZE,
            marker="+",
            color=color,
        )


def draw(constraints: list[Constraint], dofs: list[DoF]) -> Figure:
    """Draw a set of constraints and degrees of freedom.

    Returns:
        A figure with a 3d orthographic view, top view, front view, and right view.
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(15, 15))
    axes[0, 1].remove()  # type: ignore
    axes[0, 1] = fig.add_subplot(2, 2, 2, projection="3d")  # type: ignore
    top_xy, ortho = axes[0]  # type: ignore
    front_xz, right_yz = axes[1]  # type: ignore
    ortho.set_proj_type("ortho")
    for cst in constraints:
        draw_constraint_three_view(top_xy, front_xz, right_yz, cst)
        draw_constraint_3d(ortho, cst)
    for i, dof in enumerate(dofs):
        color = f"C{i}"
        draw_dof_three_view(
            top_xy,
            front_xz,
            right_yz,
            dof,
            color=color,
        )
        draw_dof_3d(ortho, dof, color=color)
    top_xy.set_title("Top")
    top_xy.set_xlabel("$x$")
    top_xy.set_ylabel("$y$")
    front_xz.set_title("Front")
    front_xz.set_xlabel("$x$")
    front_xz.set_ylabel("$z$")
    right_yz.set_title("Right")
    right_yz.set_xlabel("$y$")
    right_yz.set_ylabel("$z$")
    ortho.set_xlabel("$x$")
    ortho.set_ylabel("$y$")
    ortho.set_zlabel("$z$")

    for ax in (top_xy, front_xz, right_yz):
        ax.set_aspect("equal")
    ortho.set_aspect("equal")

    fig.tight_layout()
    fig.subplots_adjust(right=0.94)

    return fig
