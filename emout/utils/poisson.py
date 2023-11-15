from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Callable, List, Tuple

import numpy as np
import scipy.constants as cn
import scipy.fft


def poisson(rho: np.ndarray,
            dx: float,
            boundary_types: List[str] = ['periodic', 'periodic', 'periodic'],
            boundary_values: Tuple[Tuple[float]] = [
                (0., 0.), (0., 0.), (0., 0.)],
            btypes: str = None,
            epsilon_0=cn.epsilon_0):
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
    # If a boundary condition is specified in abbreviated form by btypes, revert to the original notation.
    if btypes:
        btypes_dict = {
            'p': 'periodic',
            'd': 'dirichlet',
            'n': 'neumann',
        }
        boundary_types = [btypes_dict[btype] for btype in btypes]

    POISSON_BOUNDARIES = {
        'periodic': PeriodicPoissonBoundary,
        'dirichlet': DirichletPoissonBoundary,
        'neumann': NeumannPoissonBoundary,
    }

    # [x-boundary, y-boundary, z-boundary]
    boundaries: List[PoissonBoundary] = [POISSON_BOUNDARIES[_type](2-i, boundary_values[i])
                                         for i, _type in enumerate(boundary_types)]

    rho_target = rho[tuple(boundary.get_target_slice()
                           for boundary in reversed(boundaries))].copy()

    # Poisson's equation: dphi/dx^2 = -rho/epsilon_0
    rho_target = -rho_target / epsilon_0 * dx*dx

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
        modified_wave_number[:, :, kx] \
            += boundaries[0].modified_wave_number(kx, nx)
    for ky in range(rhok.shape[1]):
        modified_wave_number[:, ky, :] \
            += boundaries[1].modified_wave_number(ky, ny)
    for kz in range(rhok.shape[0]):
        modified_wave_number[kz, :, :] \
            += boundaries[2].modified_wave_number(kz, nz)

    # Solve the equation in the wavenumber domain
    phik = rhok/modified_wave_number

    # When all boundary conditions are periodic|neumann boundaries,
    # there is no reference for the potential and it is not uniquely determined,
    # so the average is set to zero.
    if all([_type in ('periodic', 'neumann') for _type in boundary_types]):
        phik[0, 0, 0] = 0.

    # FFT backward
    _phi = fft3d.backward(phik)

    # Create an array of the same shape as the input rho array.
    phi = np.zeros_like(rho)
    phi[tuple(boundary.get_target_slice()
              for boundary in reversed(boundaries))] = _phi

    # In the above, the operation was performed on the array excluding the boundary values,
    # so the boundary values are substituted here.
    for boundary in boundaries:
        boundary.correct_boundary_values(phi)

    return phi


class FFT3d:
    def __init__(self,
                 forwards: List[Callable[[np.ndarray], np.ndarray]],
                 backwards: List[Callable[[np.ndarray], np.ndarray]]):
        self.__forwards = forwards
        self.__backwards = backwards

    def forward(self, data3d: np.ndarray) -> np.ndarray:
        result3d = data3d

        result3d = self.__forwards[2](result3d, axis=0, norm='ortho')
        result3d = self.__forwards[1](result3d, axis=1, norm='ortho')
        result3d = self.__forwards[0](result3d, axis=2, norm='ortho')

        return result3d

    def backward(self, data3d: np.ndarray) -> np.ndarray:
        result3d = data3d

        result3d = self.__backwards[2](result3d, axis=0, norm='ortho')
        result3d = self.__backwards[1](result3d, axis=1, norm='ortho')
        result3d = self.__backwards[0](result3d, axis=2, norm='ortho')

        return result3d


class PoissonBoundary(metaclass=ABCMeta):
    def __init__(self, axis: int, boundary_values: Tuple[float] = (0., 0.)):
        self.__axis = axis
        self.__boundary_values = boundary_values

    @property
    def axis(self) -> int:
        return self.__axis

    @property
    def boundary_values(self) -> Tuple[float]:
        return self.__boundary_values

    @property
    @abstractmethod
    def fft_forward(self) -> Callable[[np.ndarray], np.ndarray]:
        pass

    @property
    @abstractmethod
    def fft_backward(self) -> Callable[[np.ndarray], np.ndarray]:
        pass

    @abstractmethod
    def get_target_slice(self) -> slice:
        pass

    @abstractmethod
    def modified_wave_number(self, k: int, n: int) -> float:
        pass

    @abstractmethod
    def transpose_boundary_values(self, rho_target: np.ndarray, dx: float) -> None:
        pass

    @abstractmethod
    def correct_boundary_values(self, phi: np.ndarray) -> None:
        pass

    def _get_slices_at(self, index_axis) -> Tuple[slice]:
        return tuple(index_axis if i == self.axis else slice(None) for i in range(3))


class PeriodicPoissonBoundary(PoissonBoundary):
    @property
    def fft_forward(self) -> Callable[[np.ndarray], np.ndarray]:
        return scipy.fft.fft

    @property
    def fft_backward(self) -> Callable[[np.ndarray], np.ndarray]:
        return scipy.fft.ifft

    def get_target_slice(self) -> slice:
        return slice(0, -1)

    def modified_wave_number(self, k: int, n: int) -> float:
        if (k <= int(n/2)):
            wn = 2.0*np.sin(np.pi*k/float(n))
        else:
            wn = 2.0*np.sin(np.pi*(n - k)/float(n))
        wn = -wn*wn

        return wn

    def transpose_boundary_values(self, rho_target: np.ndarray, dx: float) -> None:
        pass

    def correct_boundary_values(self, phi: np.ndarray) -> None:
        phi[self._get_slices_at(-1)] = phi[self._get_slices_at(0)]


class DirichletPoissonBoundary(PoissonBoundary):
    @property
    def fft_forward(self) -> Callable[[np.ndarray], np.ndarray]:
        return partial(scipy.fft.dst, type=1)

    @property
    def fft_backward(self) -> Callable[[np.ndarray], np.ndarray]:
        return partial(scipy.fft.idst, type=1)

    def get_target_slice(self) -> slice:
        return slice(1, -1)

    def modified_wave_number(self, k: int, n: int) -> float:
        wn = 2.0*(np.cos(np.pi*(k + 1)/float(n + 1)) - 1.0)

        return wn

    def transpose_boundary_values(self, rho_target: np.ndarray, dx: float) -> None:
        rho_target[self._get_slices_at(0)] = \
            rho_target[self._get_slices_at(0)] - self.boundary_values[0]
        rho_target[self._get_slices_at(-1)] = \
            rho_target[self._get_slices_at(-1)] - self.boundary_values[1]

    def correct_boundary_values(self, phi: np.ndarray) -> None:
        phi[self._get_slices_at(0)] = self.boundary_values[0]
        phi[self._get_slices_at(-1)] = self.boundary_values[1]


class NeumannPoissonBoundary(PoissonBoundary):
    @property
    def fft_forward(self) -> Callable[[np.ndarray], np.ndarray]:
        return partial(scipy.fft.dct, type=1, orthogonalize=False)

    @property
    def fft_backward(self) -> Callable[[np.ndarray], np.ndarray]:
        return partial(scipy.fft.idct, type=1, orthogonalize=False)

    def get_target_slice(self) -> slice:
        return slice(None, None)

    def modified_wave_number(self, k: int, n: int) -> float:
        wn = 2.0*(np.cos(np.pi*(k)/float(n)) - 1.0)
        return wn

    def transpose_boundary_values(self, rho_target: np.ndarray, dx: float) -> None:
        rho_target[self._get_slices_at(0)] = \
            rho_target[self._get_slices_at(0)] - self.boundary_values[0]*dx
        rho_target[self._get_slices_at(-1)] = \
            rho_target[self._get_slices_at(-1)] + self.boundary_values[1]*dx

    def correct_boundary_values(self, phi: np.ndarray) -> None:
        pass
