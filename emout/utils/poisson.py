"""Poisson-equation solver with configurable boundary conditions.

Provides the :func:`poisson` entry point and boundary-condition classes
(:class:`PeriodicPoissonBoundary`, :class:`DirichletPoissonBoundary`,
:class:`NeumannPoissonBoundary`) for 1-D / 2-D / 3-D grids.
"""

from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Callable, List, Tuple

import numpy as np
import scipy.constants as cn
import scipy.fft


def poisson(
    rho: np.ndarray,
    dx: float,
    boundary_types: List[str] = None,
    boundary_values: Tuple[Tuple[float]] = None,
    btypes: str = None,
    epsilon_0=cn.epsilon_0,
):
    """Solve Poisson's equation with FFT.

    Parameters
    ----------
    rho : np.ndarray
        3-dimentional array of the charge density [C/m^3]. The shape is (nz+1, ny+1, nx+1).
    boundary_types : List[str] of {'periodic', 'dirichlet', 'neumann'},
        the boundary condition types, by default ['periodic', 'periodic', 'periodic']
    boundary_values : List[Tuple[float]]
        the boundary values [(x-lower, x-upper), (y-lower, y-upper), (z-lower, z-upper)],
        by default [(0., 0.), (0., 0.), (0., 0.)]
    btypes : str
        string consisting of prefixes of boundary conditions, by default None.
        If this is set, it takes precedence over boundary_types.
    dx : float, optional
        the grid width [m], by default 1.0
    epsilon_0 : _type_, optional
        the electric constant (vacuum permittivity) [F/m], by default cn.epsilon_0

    Returns
    -------
    np.ndarray
        3-dimentional of the potential [V].

    """
    if boundary_types is None:
        boundary_types = ["periodic", "periodic", "periodic"]
    if boundary_values is None:
        boundary_values = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]

    # If a boundary condition is specified in abbreviated form by btypes, revert to the original notation.
    if btypes:
        btypes_dict = {
            "p": "periodic",
            "d": "dirichlet",
            "n": "neumann",
        }
        boundary_types = [btypes_dict[btype] for btype in btypes]

    POISSON_BOUNDARIES = {
        "periodic": PeriodicPoissonBoundary,
        "dirichlet": DirichletPoissonBoundary,
        "neumann": NeumannPoissonBoundary,
    }

    # [x-boundary, y-boundary, z-boundary]
    boundaries: List[PoissonBoundary] = [
        POISSON_BOUNDARIES[_type](2 - i, boundary_values[i]) for i, _type in enumerate(boundary_types)
    ]

    rho_target = rho[tuple(boundary.get_target_slice() for boundary in reversed(boundaries))].copy()

    # Poisson's equation: dphi/dx^2 = -rho/epsilon_0
    rho_target = -rho_target / epsilon_0 * dx * dx

    # Transpose boundary values.
    for boundary in boundaries:
        boundary.transpose_boundary_values(rho_target, dx)

    # Create a FFT-solver with 3d data.
    forwards = [boundary.fft_forward for boundary in boundaries]
    backwards = [boundary.fft_backward for boundary in boundaries]
    fft3d = FFT3d(forwards, backwards)

    # FFT forward.
    rhok = fft3d.forward(rho_target)

    # Caluculate a modified wave number.
    modified_wave_number = np.zeros_like(rhok, dtype=float)
    nz, ny, nx = np.array(rho.shape) - 1

    for kx in range(rhok.shape[2]):
        modified_wave_number[:, :, kx] += boundaries[0].modified_wave_number(kx, nx)
    for ky in range(rhok.shape[1]):
        modified_wave_number[:, ky, :] += boundaries[1].modified_wave_number(ky, ny)
    for kz in range(rhok.shape[0]):
        modified_wave_number[kz, :, :] += boundaries[2].modified_wave_number(kz, nz)

    # Solve the equation in the wavenumber domain.  The zero mode becomes
    # singular when every axis is periodic/neumann; keep it masked here and
    # fix the reference level explicitly below.
    phik = np.zeros_like(rhok)
    np.divide(rhok, modified_wave_number, out=phik, where=modified_wave_number != 0.0)

    # When all boundary conditions are periodic|neumann boundaries,
    # there is no reference for the potential and it is not uniquely determined,
    # so the average is set to zero.
    if all([_type in ("periodic", "neumann") for _type in boundary_types]):
        phik[0, 0, 0] = 0.0

    # FFT backward
    _phi = fft3d.backward(phik)
    _phi = np.real_if_close(_phi, tol=1000)
    if np.iscomplexobj(_phi):
        _phi = _phi.real

    # Create an array of the same shape as the input rho array.
    phi = np.zeros(rho.shape, dtype=np.result_type(_phi, np.float64))
    phi[tuple(boundary.get_target_slice() for boundary in reversed(boundaries))] = _phi

    # In the above, the operation was performed on the array excluding the boundary values,
    # so the boundary values are substituted here.
    for boundary in boundaries:
        boundary.correct_boundary_values(phi)

    return phi


class FFT3d:
    """Composite 3-D FFT that applies per-axis forward/backward transforms."""

    def __init__(
        self,
        forwards: List[Callable[[np.ndarray], np.ndarray]],
        backwards: List[Callable[[np.ndarray], np.ndarray]],
    ):
        """Initialize the instance.

        Parameters
        ----------
        forwards : List[Callable[[np.ndarray], np.ndarray]]
            Forward FFT callable for each axis.
        backwards : List[Callable[[np.ndarray], np.ndarray]]
            Backward FFT callable for each axis.
        """
        self.__forwards = forwards
        self.__backwards = backwards

    def forward(self, data3d: np.ndarray) -> np.ndarray:
        """Apply the forward transform.

        Parameters
        ----------
        data3d : np.ndarray
            3-D input data.

        Returns
        -------
        np.ndarray
            Transformed data in frequency space.
        """
        result3d = data3d

        result3d = self.__forwards[2](result3d, axis=0, norm="ortho")
        result3d = self.__forwards[1](result3d, axis=1, norm="ortho")
        result3d = self.__forwards[0](result3d, axis=2, norm="ortho")

        return result3d

    def backward(self, data3d: np.ndarray) -> np.ndarray:
        """Apply the backward (inverse) transform.

        Parameters
        ----------
        data3d : np.ndarray
            3-D input data in frequency space.

        Returns
        -------
        np.ndarray
            Transformed data in real space.
        """
        result3d = data3d

        result3d = self.__backwards[2](result3d, axis=0, norm="ortho")
        result3d = self.__backwards[1](result3d, axis=1, norm="ortho")
        result3d = self.__backwards[0](result3d, axis=2, norm="ortho")

        return result3d


class PoissonBoundary(metaclass=ABCMeta):
    """Abstract base for Poisson boundary conditions."""

    def __init__(self, axis: int, boundary_values: Tuple[float] = (0.0, 0.0)):
        """Initialize the instance.

        Parameters
        ----------
        axis : int
            Target axis index.
        boundary_values : Tuple[float], optional
            Boundary values ``(lower, upper)`` for the target axis.
        """
        self.__axis = axis
        self.__boundary_values = boundary_values

    @property
    def axis(self) -> int:
        """Return the target axis index.

        Returns
        -------
        int
            Target axis index.
        """
        return self.__axis

    @property
    def boundary_values(self) -> Tuple[float]:
        """Return the boundary values ``(lower, upper)``.

        Returns
        -------
        Tuple[float]
            Boundary value pair.
        """
        return self.__boundary_values

    @property
    @abstractmethod
    def fft_forward(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return the forward FFT function for this boundary condition.

        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            Forward FFT callable.
        """
        pass

    @property
    @abstractmethod
    def fft_backward(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return the backward FFT function for this boundary condition.

        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            Backward FFT callable.
        """
        pass

    @abstractmethod
    def get_target_slice(self) -> slice:
        """Return the slice selecting the interior grid points for this axis.

        Returns
        -------
        slice
            Slice corresponding to the target axis.
        """
        pass

    @abstractmethod
    def modified_wave_number(self, k: int, n: int) -> float:
        """Compute the modified wave number for this boundary condition.

        Parameters
        ----------
        k : int
            Wavenumber index.
        n : int
            Number of grid points.
        Returns
        -------
        float
            Modified wave number value.
        """
        pass

    @abstractmethod
    def transpose_boundary_values(self, rho_target: np.ndarray, dx: float) -> None:
        """Transpose (fold) boundary values into the right-hand side array.

        Parameters
        ----------
        rho_target : np.ndarray
            Charge density array for the target region.
        dx : float
            Grid spacing.
        Returns
        -------
        None
            No return value.
        """
        pass

    @abstractmethod
    def correct_boundary_values(self, phi: np.ndarray) -> None:
        """Correct boundary values in the potential array.

        Parameters
        ----------
        phi : np.ndarray
            Potential array.
        Returns
        -------
        None
            No return value.
        """
        pass

    def _get_slices_at(self, index_axis) -> Tuple[slice]:
        """Return a tuple of slices that selects a boundary plane at the given index.

        Parameters
        ----------
        index_axis : object
            Fixed index along ``self.axis`` (e.g. ``0`` or ``-1``).
        Returns
        -------
        Tuple[slice]
            Indexing tuple for the 3-D array.
        """
        return tuple(index_axis if i == self.axis else slice(None) for i in range(3))


class PeriodicPoissonBoundary(PoissonBoundary):
    """Periodic boundary condition for the Poisson solver."""

    @property
    def fft_forward(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return the forward FFT function for periodic boundaries.

        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            Forward FFT callable.
        """
        return scipy.fft.fft

    @property
    def fft_backward(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return the backward FFT function for periodic boundaries.

        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            Backward FFT callable.
        """
        return scipy.fft.ifft

    def get_target_slice(self) -> slice:
        """Return the target slice for periodic boundaries.

        Returns
        -------
        slice
            Slice excluding the last (periodic duplicate) point.
        """
        return slice(0, -1)

    def modified_wave_number(self, k: int, n: int) -> float:
        """Compute the modified wave number for periodic boundaries.

        Parameters
        ----------
        k : int
            Wavenumber index.
        n : int
            Number of grid points.
        Returns
        -------
        float
            Modified wave number value.
        """
        if k <= int(n / 2):
            wn = 2.0 * np.sin(np.pi * k / float(n))
        else:
            wn = 2.0 * np.sin(np.pi * (n - k) / float(n))
        wn = -wn * wn

        return wn

    def transpose_boundary_values(self, rho_target: np.ndarray, dx: float) -> None:
        """Transpose boundary values into the right-hand side (no-op for periodic).

        Parameters
        ----------
        rho_target : np.ndarray
            Charge density array for the target region.
        dx : float
            Grid spacing.
        Returns
        -------
        None
            No return value.
        """
        pass

    def correct_boundary_values(self, phi: np.ndarray) -> None:
        """Correct boundary values by copying the periodic duplicate.

        Parameters
        ----------
        phi : np.ndarray
            Potential array.
        Returns
        -------
        None
            No return value.
        """
        phi[self._get_slices_at(-1)] = phi[self._get_slices_at(0)]


class DirichletPoissonBoundary(PoissonBoundary):
    """Fixed-value (Dirichlet) boundary condition for the Poisson solver."""

    @property
    def fft_forward(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return the forward FFT function for Dirichlet boundaries.

        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            Forward DST callable.
        """
        return partial(scipy.fft.dst, type=1)

    @property
    def fft_backward(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return the backward FFT function for Dirichlet boundaries.

        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            Backward IDST callable.
        """
        return partial(scipy.fft.idst, type=1)

    def get_target_slice(self) -> slice:
        """Return the target slice for Dirichlet boundaries.

        Returns
        -------
        slice
            Slice excluding both boundary points.
        """
        return slice(1, -1)

    def modified_wave_number(self, k: int, n: int) -> float:
        """Compute the modified wave number for Dirichlet boundaries.

        Parameters
        ----------
        k : int
            Wavenumber index.
        n : int
            Number of grid points.
        Returns
        -------
        float
            Modified wave number value.
        """
        wn = 2.0 * (np.cos(np.pi * (k + 1) / float(n + 1)) - 1.0)

        return wn

    def transpose_boundary_values(self, rho_target: np.ndarray, dx: float) -> None:
        """Transpose boundary values into the right-hand side for Dirichlet conditions.

        Parameters
        ----------
        rho_target : np.ndarray
            Charge density array for the target region.
        dx : float
            Grid spacing.
        Returns
        -------
        None
            No return value.
        """
        rho_target[self._get_slices_at(0)] = rho_target[self._get_slices_at(0)] - self.boundary_values[0]
        rho_target[self._get_slices_at(-1)] = rho_target[self._get_slices_at(-1)] - self.boundary_values[1]

    def correct_boundary_values(self, phi: np.ndarray) -> None:
        """Correct boundary values by assigning the fixed Dirichlet values.

        Parameters
        ----------
        phi : np.ndarray
            Potential array.
        Returns
        -------
        None
            No return value.
        """
        phi[self._get_slices_at(0)] = self.boundary_values[0]
        phi[self._get_slices_at(-1)] = self.boundary_values[1]


class NeumannPoissonBoundary(PoissonBoundary):
    """Zero-gradient (Neumann) boundary condition for the Poisson solver."""

    @property
    def fft_forward(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return the forward FFT function for Neumann boundaries.

        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            Forward DCT callable.
        """
        return partial(scipy.fft.dct, type=1, orthogonalize=False)

    @property
    def fft_backward(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return the backward FFT function for Neumann boundaries.

        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            Backward IDCT callable.
        """
        return partial(scipy.fft.idct, type=1, orthogonalize=False)

    def get_target_slice(self) -> slice:
        """Return the target slice for Neumann boundaries.

        Returns
        -------
        slice
            Slice including all points.
        """
        return slice(None, None)

    def modified_wave_number(self, k: int, n: int) -> float:
        """Compute the modified wave number for Neumann boundaries.

        Parameters
        ----------
        k : int
            Wavenumber index.
        n : int
            Number of grid points.
        Returns
        -------
        float
            Modified wave number value.
        """
        wn = 2.0 * (np.cos(np.pi * (k) / float(n)) - 1.0)
        return wn

    def transpose_boundary_values(self, rho_target: np.ndarray, dx: float) -> None:
        """Transpose boundary values into the right-hand side for Neumann conditions.

        Parameters
        ----------
        rho_target : np.ndarray
            Charge density array for the target region.
        dx : float
            Grid spacing.
        Returns
        -------
        None
            No return value.
        """
        rho_target[self._get_slices_at(0)] = rho_target[self._get_slices_at(0)] - self.boundary_values[0] * dx
        rho_target[self._get_slices_at(-1)] = rho_target[self._get_slices_at(-1)] + self.boundary_values[1] * dx

    def correct_boundary_values(self, phi: np.ndarray) -> None:
        """Correct boundary values for Neumann conditions (no-op).

        Parameters
        ----------
        phi : np.ndarray
            Potential array.
        Returns
        -------
        None
            No return value.
        """
        pass
