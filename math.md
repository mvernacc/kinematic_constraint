This note describes a method to find the degrees of freedom of a constrained 3d rigid body
using the [null space](https://en.wikipedia.org/wiki/Kernel_(linear_algebra)) of a constraint matrix.
The constraints are modeled as rigid lines connected to points on the body,
as described in the literature of kinematic / exact constraint (e.g. Blanding 1999, Hale 1999).

The note first applies the method to only translation degrees of freedom,
which is less useful but easier to explain.
The second section applies the method to rotation and translation degrees of freedom.

## Translation only

Suppose we have a set of $n$ constraints which connect to a body at connection points $\vec{c_i}$
in directions $\vec{d_i}$.
We want to find the set of translation directions which are allowed by these constraints.

A translation is allowed by the constraints if a infinitesimal motion $ds$
in the translation direction $\vec{t}$ would not change the length of any constraint,
i.e. would not move the connection point along the constraint direction.
Thus, we seek a $n \times 3$ linear operator $A$ which maps translations to constraint length changes:

$$
\vec{dl} = A (\vec{t} \, ds)
$$

The null space of $A$ is the allowed translation directions.

For constraint $i$, its length change is,

$$
dl_i = \vec{d_i} \cdot (\vec{t} \, ds)
$$

Thus the rows of $A$ are simply the directions $\vec{d_i}$ of each constraint.

## Translation and rotation

Now we wish to find the allowed set of rotations and translations.
Again, we will do this by (1) constructing a linear operator which maps differential motions to the differential length change of the constraints, and (2) finding the null space of that linear operator.
However, the domain of this linear operator will be [screw](https://en.wikipedia.org/wiki/Screw_theory)-like 6-vectors which represent combined rotations and translations.

To derive $A$, we start with a formula for the constraint length change $dl_i$ due to a rotation of
$d\theta$ radians about an axis $\vec{r}$ through a point $\vec{p}$,
coupled with a translation $ds$ in direction $\vec{t}$.
Again, we have a set of $n$ constraints which connect to a body at connection points $\vec{c_i}$
in directions $\vec{d_i}$.
The vectors $\vec{r}$, $\vec{t}$ and $\vec{d_i}$ are dimensionless unit vectors,
whereas $\vec{p}$ and $\vec{c_i}$ have units of length.

Let $\vec{dc_i}$ be instantaneous change in the location of connection point $i$ due to this motion.
The location change due to rotation is:
$$
(\vec{r} \times (\vec{c_i} - \vec{p})) \, d\theta
$$

and the location change due to translation is:
$$
\vec{t} \, ds
$$

Combined, the location change is:

$$
\begin{align}
\vec{dc_i} &= (\vec{r} \times (\vec{c_i} - \vec{p})) \, d\theta + \vec{t} \, ds \\
&= (\vec{r} \times \vec{c_i} - \vec{r} \times \vec{p}) \, d\theta + \vec{t} \, ds
\end{align}
$$

The constraint length change is the component of $\vec{dc_i}$ in the constraint direction $\vec{d_i}$:

$$
\begin{align}
dl_i &= \vec{dc_i} \cdot \vec{d_i} \\
&= \left( (\vec{r} \times \vec{c_i} - \vec{r} \times \vec{p}) \, d\theta + \vec{t} \, ds \right) \cdot \vec{d_i} \\
&= (\vec{r} \times \vec{c_i} \cdot \vec{d_i}) \, d\theta
   - (\vec{r} \times \vec{p} \cdot \vec{d_i}) \, d\theta
   + (\vec{t} \cdot \vec{d_i}) \, ds
\end{align}
$$

Apply the vector triple product identity to the first term:

$$
dl_i = (\vec{c_i} \times \vec{d_i} \cdot \vec{r}) \, d\theta
   - (\vec{r} \times \vec{p} \cdot \vec{d_i}) \, d\theta
   + (\vec{t} \cdot \vec{d_i}) \, ds
$$

Rearrange the second and third terms:

$$
dl_i = (\vec{c_i} \times \vec{d_i}) \cdot \vec{r} \, d\theta
+ \vec{d_i} \cdot \left( \vec{t} \, ds - (\vec{r} \times \vec{p}) \, d\theta \right)
$$

Now, define a [screw](https://en.wikipedia.org/wiki/Screw_theory)-like 6-vector $\vec{dm}$
for the combined rotation and translation motion:

$$
\vec{dm} = [\vec{r} \, d\theta, \quad \vec{t} \, ds - (\vec{r} \times \vec{p}) \, d\theta]
$$

Note that several combinations of $\vec{r}, \vec{p}, \vec{t}$ can produce the same $\vec{dm}$.
First, any component of the translation direction $\vec{t}$
which is perpendicular to the rotation axis $\vec{r}$ could be
"compensated for" by choosing a different rotation point $\vec{p}$.
Thus, without loss of generality in the motions we can represent, assume that $\vec{t} = \vec{r}$,
i.e. that the translation is along the axis of rotation
([Chasles' theorem](https://en.wikipedia.org/wiki/Chasles%27_theorem_(kinematics))).
Second, the line of rotation is unchanged if we move the rotation point
along the direction of rotation, i.e. if we add $a \vec{r}$ to $\vec{p}$ for any scalar $a$.

Finally, we can define a $n \times 6$ linear operator $A$ which maps screws $\vec{dm}$ to the
length change $dl_i$ of each constraint:

$$
\vec{dl} = A \cdot \vec{dm}
$$

The rows of $A$ are:

$$
A_{i,:} = [\vec{c_i} \times \vec{d_i}, \quad \vec{d_i}]
$$

The null space of $A$ is the set of rotation and translation degrees of freedom which are allowed by the constraints.

### Interpreting the linear operator and its null space basis

Many linear algebra libraries provide a function to calculate an orthonormal basis for the null space of a matrix, e.g. [`scipy.linalg.null_space`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.null_space.html).
However, simply displaying this basis matrix to the user will not help them understand the
system's degrees of freedom.
We want to do several interpretation tasks:

1. Convert each basis vector into a translation and/or rotation line.
2. Check if a user-supplied rotation + translation is allowed by the constraints, i.e. is within the null space.

Let $M$ be the matrix of orthonormal basis vectors of the null space of $A$, and let the unit 6-vector $\bar{m}$ be a particular column of $M$.

#### Convert a basis vector to a rotation and/or translation

If $\bar{m}_{1:3}$ are zero, then $\bar{m}$ represents a pure translation, and $\bar{m}_{4:6}$ are the translation direction.

If $\bar{m}_{1:3}$ are non-zero, then $\vec{r} = \mathrm{unit}(\bar{m}_{1:3})$.
Next, we need to extract $\vec{p}$, and possibly $\vec{t}$ from
$\bar{m}_{4:6} = \vec{t} \, ds - \vec{r} \times \vec{p} \, d\theta$.
This is more complicated.
To do so, we set up a linear system of equations, that can be solved for $\vec{p}$.
Given $\vec{r}$, we wish to find the $\vec{p}$ such that $dl_i/d\theta$ is zero for every constraint:

$$
0 = \frac{dl_i}{d\theta} = \vec{c_i} \times \vec{d_i} \cdot \vec{r} - \vec{r} \times \vec{p} \cdot \vec{d_i}  \quad \forall i
$$

Rearranging and using the vector triple product identity gives:

$$
\vec{c_i} \times \vec{d_i} \cdot \vec{r} = (\vec{d_i} \times \vec{r}) \cdot \vec{p} \quad \forall i
$$

This is a linear system of equations:

$$
\vec{b_p} = A_p \cdot \vec{p}
$$

The number of equations is the number of constraints, which may not be 3, so the system of equations
may be under-, well, or over-determined.
Thus, we find a least-squares solution for $\vec{p}$, e.g. using [`numpy.linalg.lstsq`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html#numpy.linalg.lstsq).
If the least-squares solution has zero residual, $\bar{m}$ represents a pure rotation.

Otherwise, $\bar{m}$ represents a coupled rotation and translation.
In this case, we find the "remainder" of $\bar{m}_{4:6}$:
$$
\vec{m_{remain}} = \bar{m}_{4:6} + \bar{m}_{1:3} \times \vec{p}
$$

The remainder is in the direction of the translation, which should be parallel to $\vec{r}$,
i.e. a helical motion. The pitch of the helix is:
$$
\frac{ds}{d\theta} = \frac{|| \vec{t} \, ds ||}{|| \vec{r} \, d\theta ||} = \frac{|| \vec{m_{remain}} ||}{|| \bar{m}_{1:3} ||}
$$ 

#### Check if a rotation and translation is allowed

Convert the given rotation and translation into a screw 6-vector:
$$
\vec{dm} = [\vec{r} \, d\theta, \quad \vec{t} \, ds - (\vec{r} \times \vec{p}) \, d\theta]
$$

Multiply the constraint matrix $A$ by the screw to get the constraint length changes.
If all the $dl$s are zero (to within some tolerance), then the motion is allowed by the constraints.

$$
|| A \cdot \vec{dm} || = || \vec{dl} || = 0 \rightarrow \vec{dm} \text{ is a valid degree of freedom}
$$
