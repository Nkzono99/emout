from typing import Literal

import numpy as np


def relocated_magnetic_field(
    bf: np.array, axis: int, btype: Literal["periodic", "dirichlet", "neumann"]
):
    def slc(s1, s2=slice(None, None)):
        axis1 = (axis + 1) % 3
        axis2 = (axis + 2) % 3

        slices = [None, None, None]

        slices[axis] = slice(None, None)
        slices[axis1] = s1
        slices[axis2] = s2
        slices = tuple(slices)

        return slices

    # Relocated electric field buffer
    rbf = np.zeros_like(bf)

    # xy平面に1グリッド覆うように拡張する
    bfe = np.empty(
        np.array(bf.shape) + np.array([0 if i == axis else 1 for i in range(3)])
    )
    bfe[slc(slice(1, -1), slice(1, -1))] = bf[slc(slice(None, -1), slice(None, -1))]
    if btype in "periodic":
        bfe[slc(slice(1, -1), 0)] = bfe[slc(slice(1, -1), -2)]
        bfe[slc(slice(1, -1), -1)] = bfe[slc(slice(1, -1), 1)]
    elif btype in "dirichlet":
        bfe[slc(slice(1, -1), 0)] = -bfe[slc(slice(1, -1), 1)]
        bfe[slc(slice(1, -1), -1)] = -bfe[slc(slice(1, -1), -2)]
    else:  # if btype in "neumann":
        bfe[slc(slice(1, -1), 0)] = bfe[slc(slice(1, -1), 1)]
        bfe[slc(slice(1, -1), -1)] = bfe[slc(slice(1, -1), -2)]

    if btype in "periodic":
        bfe[slc(0)] = bfe[slc(-2)]
        bfe[slc(-1)] = bfe[slc(1)]
    elif btype in "dirichlet":
        bfe[slc(0)] = -bfe[slc(1)]
        bfe[slc(-1)] = -bfe[slc(-2)]
    else:  # if btype in "neumann":
        bfe[slc(0)] = bfe[slc(1)]
        bfe[slc(-1)] = bfe[slc(-2)]

    rbf[:, :, :] = 0.25 * (
        bfe[slc(slice(None, -1), slice(None, -1))]
        + bfe[slc(slice(1, None), slice(None, -1))]
        + bfe[slc(slice(None, -1), slice(1, None))]
        + bfe[slc(slice(1, None), slice(1, None))]
    )

    return rbf
