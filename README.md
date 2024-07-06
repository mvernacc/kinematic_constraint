# Kinematic Constraint

Solve for rigid body degrees of freedom with the theory of kinematic constraint (aka exact constraint).

For example, the cube below is connected to four constraints (show by thick black lines with dots).
This body has one translational degree of freedom along $z$ (shown by a thin arrow),
and one rotational degree of freedom around $x$ (shown by a thin axis with "+" marks)

![A cube with 4 constraints and 2 degrees of freedom, from Hale 1999](figures/hale1999_1r1t.png)

This python script calculates and visualizes its degrees of freedom:

```pycon
>>> from matplotlib import pyplot as plt
>>> from kinematic_constraint import Constraint, calc_dofs, draw
>>> constraints = [
...     Constraint(point=(1, 1, 1), direction=(-1, 0, 0)),
...     Constraint(point=(1, 1, -1), direction=(-1, 0, 0)),
...     Constraint(point=(1, -1, -1), direction=(-1, 0, 0)),
...     Constraint(point=(0, -1, 0), direction=(0, 1, 0)),
... ]
>>> dofs = calc_dofs(constraints)
>>> for dof in dofs:
...     print(dof)
DoF(translation=None, rotation=Rotation(point=(-0.0, 0.0, -0.0), direction=(-1.0, 0.0, 0.0)), pitch=None)
DoF(translation=(0.0, 0.0, 1.0), rotation=None, pitch=None)

>>> draw(constraints, dofs)
>>> plt.show()  # doctest: +SKIP

```

## Motivation

The theory of exact constraint is a powerful tool for the design of precision machines / mechanisms.
However, it can be difficult to apply.
Blanding 1999 and Hale 1999 (see [references](#references)) describe a geometric / pictorial method for determining
the degrees of freedom from the applied constraints
(e.g. "The axes of a body's rotational degrees of freedom will each intersect all constraints applied to the body").
This "think real hard about the line diagram" approach is difficult to apply,
particularly for complicated arrangements of non-orthogonal constraints.
Missing a degree of freedom at the design stage may not be noticed until the mechanism is built (and doesn't work), and thus can have expensive consequences
(the author has done this once or twice 😰).

This software can automatically detect the degrees of freedom, and visualize them,
with only a few lines of python.
Hopefully, this makes the method of exact constraint easier to use in mechanical design.


## How it works

The degrees of freedom are calculated from the null space of a constraint matrix, as described in [math.md](math.md).

The method is tested on every example from Figure 2-21 of Hale 1999:

![Hale 1999, Figure 2-21](figures/hale1999_fig_2-21.png)

The method also handles helical degrees of freedom, e.g. like this
arrangement from Figure 2-25 of Hale 1999:

![Hale 1999, Figure 2-25](figures/hale1999_fig_2-25.png)


## References

[1] D. Blanding, Exact Constraint: Machine Design Using Kinematic Processing.
New York: American Society of Mechanical Engineers, 1999.

[2] Layton C. Hale, "Principles and techniques for designing precision machines,"
Thesis, Massachusetts Institute of Technology, 1999. Accessed: Jun. 28, 2022.
[Online]. Available: https://dspace.mit.edu/handle/1721.1/9414

