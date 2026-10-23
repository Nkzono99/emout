"""Relocate EMSES electric-field components to cell centres."""

from typing import Literal

import numpy as np


def relocated_electric_field(
    ef: np.ndarray,
    axis: int,
    btype: Literal["periodic", "dirichlet", "neumann"],
    *,
    periodic_endpoint: Literal["legacy", "duplicate"] = "legacy",
):
    """Relocate the electric field to cell-centre positions.

    Parameters
    ----------
    ef : np.ndarray
        Electric field array.
    axis : int
        Target axis.
    btype : Literal["periodic", "dirichlet", "neumann"]
        Boundary condition type.
    periodic_endpoint : str, optional
        'duplicate' treats the last input as a periodic copy of the first
        and averages the two adjacent samples at the seam. 'legacy'
        preserves the historical boundary formula.

    Returns
    -------
    np.ndarray
        Relocated electric field array.
    """

    if periodic_endpoint not in ("legacy", "duplicate"):
        raise ValueError("periodic_endpoint must be 'legacy' or 'duplicate'")

    def slc(a, b=None):
        """Build a slice tuple for the target axis.

        Parameters
        ----------
        a : object
            Start-side value.
        b : object, optional
            End-side value.

        Returns
        -------
        tuple
            Slice tuple addressing the target axis.
        """
        s = slice(a, b) if b else a
        slices = tuple(s if i == axis else slice(None, None) for i in range(3))

        return slices

    # Relocated electric field buffer
    ref = np.zeros_like(ef)

    ref[slc(1, -1)] = 0.5 * (ef[slc(None, -2)] + ef[slc(1, -1)])

    if btype == "periodic":
        neighbour = 0 if periodic_endpoint == "duplicate" else 1
        ref[slc(0)] = 0.5 * (ef[slc(-2)] + ef[slc(neighbour)])
        ref[slc(-1)] = ref[slc(0)]
    elif btype == "neumann":
        ref[slc(0)] = 0
        ref[slc(-1)] = 0
    else:
        ref[slc(0)] = ef[slc(1)]
        ref[slc(-1)] = ef[slc(-2)]

    return ref
