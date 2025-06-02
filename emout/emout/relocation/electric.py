from typing import Literal

import numpy as np


def relocated_electric_field(
    ef: np.ndarray, axis: int, btype: Literal["periodic", "dirichlet", "neumann"]
):
    def slc(a, b=None):
        s = slice(a, b) if b else a
        slices = tuple(s if i == axis else slice(None, None) for i in range(3))

        return slices

    # Relocated electric field buffer
    ref = np.zeros_like(ef)

    ref[slc(1, -1)] = 0.5 * (ef[slc(None, -2)] + ef[slc(1, -1)])

    if btype == "periodic":
        ref[slc(0)] = 0.5 * (ef[slc(-2)] + ef[slc(1)])
        ref[slc(-1)] = 0.5 * (ef[slc(-2)] + ef[slc(1)])
    elif btype == "neumann":
        ref[slc(0)] = 0
        ref[slc(-1)] = 0
    else:
        ref[slc(0)] = ef[slc(1)]
        ref[slc(-1)] = ef[slc(-2)]

    return ref
