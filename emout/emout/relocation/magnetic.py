from typing import Literal, Tuple

import numpy as np


Btype = Literal["periodic", "dirichlet", "neumann"]

def relocated_magnetic_field(
    bf: np.array, axis: int, btypes: Tuple[Btype, Btype] 
):
    axis1 = (axis + 1) % 3
    axis2 = (axis + 2) % 3

    def slc(s1, s2=slice(None, None)):
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
    if btypes[0] == "periodic":
        bfe[slc(slice(1, -1), 0)] = bfe[slc(slice(1, -1), -2)]
        bfe[slc(slice(1, -1), -1)] = bfe[slc(slice(1, -1), 1)]
    elif btypes[0] == "dirichlet":
        bfe[slc(slice(1, -1), 0)] = -bfe[slc(slice(1, -1), 1)]
        bfe[slc(slice(1, -1), -1)] = -bfe[slc(slice(1, -1), -2)]
    else:  # if btypes[0] == "neumann":
        bfe[slc(slice(1, -1), 0)] = bfe[slc(slice(1, -1), 1)]
        bfe[slc(slice(1, -1), -1)] = bfe[slc(slice(1, -1), -2)]

    if btypes[1] == "periodic":
        bfe[slc(0)] = bfe[slc(-2)]
        bfe[slc(-1)] = bfe[slc(1)]
    elif btypes[1] == "dirichlet":
        bfe[slc(0)] = -bfe[slc(1)]
        bfe[slc(-1)] = -bfe[slc(-2)]
    else:  # if btypes[1] == "neumann":
        bfe[slc(0)] = bfe[slc(1)]
        bfe[slc(-1)] = bfe[slc(-2)]

    rbf[:, :, :] = 0.25 * (
        bfe[slc(slice(None, -1), slice(None, -1))]
        + bfe[slc(slice(1, None), slice(None, -1))]
        + bfe[slc(slice(None, -1), slice(1, None))]
        + bfe[slc(slice(1, None), slice(1, None))]
    )

    return rbf
