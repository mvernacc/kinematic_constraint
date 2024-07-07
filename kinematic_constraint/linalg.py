"""Linear algebra helpers."""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .geometry import Vec3


def basis_contains_vector(basis: Sequence[Vec3], vector: Vec3) -> bool:
    """Check if `vector` lies in the space spanned by the vectors in `basis`."""
    A = np.stack([*basis, vector])
    return np.linalg.matrix_rank(A) <= len(basis)


def orthogonal_subspace(basis: Sequence[Vec3], vector: Vec3) -> NDArray:
    """Within the space spanned by the vectors of `basis`, create a basis for a new
    sub-space, which is orthogonal to `vector`.
    """
    A = np.stack([vector, *basis], axis=-1)
    Q, R = np.linalg.qr(A)
    return Q[:, 1:]
