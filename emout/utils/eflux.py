"""
Energy-flux computation and pitch-angle classification library.
Energies are handled in eV.
"""

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.constants import e as e_charge


def get_indices_in_pitch_range(
    velocities: np.ndarray,
    B: np.ndarray,
    a_deg: float,
    b_deg: float,
    direction: str = 'both',
) -> np.ndarray:
    """
    Return the indices of particles whose pitch angle falls within [a_deg, b_deg].

    Parameters
    ----------
    velocities : np.ndarray, shape (N, 3)
        Velocity vectors for each particle (m/s). N is the number of samples.
    B : np.ndarray, shape (3,)
        Magnetic field vector (T) or direction vector. Must be non-zero.
    a_deg : float
        Lower bound of the pitch angle (degrees). 0 <= a_deg < b_deg <= 180.
    b_deg : float
        Upper bound of the pitch angle (degrees).
    direction : str, default='both'
        Sign direction of the pitch angle:
          - 'both': all particles regardless of the sign of v dot B
          - 'pos' : only particles with v dot B > 0 (parallel to B)
          - 'neg' : only particles with v dot B < 0 (anti-parallel to B)

    Returns
    -------
    idx : np.ndarray
        Index array of particles satisfying the angle range and direction condition.
    """

    if not (0.0 <= a_deg < b_deg <= 180.0):
        raise ValueError(f"Invalid a_deg={a_deg}, b_deg={b_deg}. Required: 0 <= a < b <= 180")
    if direction not in ('both', 'pos', 'neg'):
        raise ValueError(f"direction='{direction}' must be one of 'both', 'pos', 'neg'.")

    a_rad = np.deg2rad(a_deg)
    b_rad = np.deg2rad(b_deg)
    cos_a = np.cos(a_rad)
    cos_b = np.cos(b_rad)

    speeds = np.linalg.norm(velocities, axis=1)
    B_norm = np.linalg.norm(B)
    if B_norm == 0:
        raise ValueError("Magnetic field vector B has zero magnitude.")

    dot_vB = velocities.dot(B)
    cos_theta = np.zeros_like(dot_vB)
    nz = speeds > 0
    cos_theta[nz] = dot_vB[nz] / (speeds[nz] * B_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    mask_angle = (cos_theta >= cos_b) & (cos_theta <= cos_a)
    if direction == 'pos':
        mask_dir = dot_vB > 0
    elif direction == 'neg':
        mask_dir = dot_vB < 0
    else:
        mask_dir = np.ones_like(dot_vB, dtype=bool)

    idx = np.where(mask_angle & mask_dir)[0]
    return idx


def compute_energy_flux_histogram(
    velocities: np.ndarray,
    probs: np.ndarray,
    mass: float,
    energy_bins: Union[int, np.ndarray],
):
    """Compute an energy-flux histogram.

    Parameters
    ----------
    velocities : np.ndarray
        Particle velocity array.
    probs : np.ndarray
        Weight (probability / contribution) array for each particle.
    mass : float
        Particle mass [kg].
    energy_bins : Union[int, np.ndarray]
        Histogram bin specification. An integer sets the number of bins;
        an array sets the bin edges explicitly.
    Returns
    -------
    object
        Tuple of ``(hist, bin_edges)``.
    """
    speeds = np.linalg.norm(velocities, axis=1)
    energies_J = 0.5 * mass * speeds**2
    energies_eV = energies_J / e_charge

    if energy_bins is None:
        energy_bins = 30

    if isinstance(energy_bins, int):
        _, bin_edges = np.histogram(energies_eV, bins=energy_bins)
        bins = bin_edges
    else:
        bins = energy_bins.copy()

    energy_flux = energies_eV * speeds * probs

    E_cls = energies_eV
    w_cls = energy_flux
    hist, bin_edges = np.histogram(E_cls, bins=bins, weights=w_cls)

    return hist, bin_edges


def compute_energy_flux_histograms(
    velocities: np.ndarray,
    probs: np.ndarray,
    B: np.ndarray,
    mass: float,
    energy_bins: Union[int, np.ndarray],
    pitch_ranges: Union[list[tuple[float, float, str]], None] = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Compute energy x energy-flux histograms for user-specified pitch-angle
    ranges and directions, from velocity vectors and a probability array.
    Energies are computed in eV.

    Parameters
    ----------
    velocities : np.ndarray, shape (N, 3)
        Velocity vectors for each particle (m/s). N is the number of samples.
    probs : np.ndarray, shape (N,)
        Probability or weight corresponding to each velocity vector.
    B : np.ndarray, shape (3,)
        Magnetic field vector (T) or direction vector.
    mass : float
        Particle mass (kg).
    energy_bins : int or np.ndarray, shape (M+1,)
        - int: let numpy.histogram auto-generate bin edges.
        - np.ndarray: use these bin edges directly.
    pitch_ranges : list of (a_deg, b_deg, direction) | None
        List specifying pitch-angle ranges and directions. Each tuple has
        the form ``(a_deg, b_deg, direction)`` where direction is one of
        ``'both'``, ``'pos'``, ``'neg'``.
        If ``None``, the default 6-class decomposition is used.

    Returns
    -------
    histograms : dict[str, (hist, bin_edges)]
        Keys have the format ``f"{a_deg:02d}-{b_deg:02d}_{direction}"``.
        Values are ``(hist, bin_edges)`` tuples where:
        - hist: total energy flux (eV x v x prob) per bin, shape=(M,)
        - bin_edges: bin boundaries in eV, shape=(M+1,)
    """

    N = velocities.shape[0]
    if probs.shape[0] != N:
        raise ValueError("`velocities` and `probs` must have the same length.")

    # Compute speeds and energies (eV)
    speeds = np.linalg.norm(velocities, axis=1)
    energies_J = 0.5 * mass * speeds**2
    energies_eV = energies_J / e_charge

    # If energy_bins is int, let numpy.histogram auto-generate bin_edges
    if isinstance(energy_bins, int):
        _, bin_edges = np.histogram(energies_eV, bins=energy_bins)
        bins = bin_edges
    else:
        bins = energy_bins.copy()
    M = len(bins) - 1

    # Energy-flux weight per particle = eV x speed x prob
    energy_flux = energies_eV * speeds * probs

    if pitch_ranges is None:
        pitch_ranges = [
            (0.0, 30.0, 'pos'),
            (0.0, 30.0, 'neg'),
            (30.0, 60.0, 'pos'),
            (30.0, 60.0, 'neg'),
            (60.0, 180.0, 'pos'),
            (60.0, 180.0, 'neg'),
        ]

    histograms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for (a_deg, b_deg, direction) in pitch_ranges:
        idx = get_indices_in_pitch_range(
            velocities=velocities,
            B=B,
            a_deg=a_deg,
            b_deg=b_deg,
            direction=direction
        )
        if idx.size > 0:
            E_cls = energies_eV[idx]
            w_cls = energy_flux[idx]
            hist, bin_edges = np.histogram(E_cls, bins=bins, weights=w_cls)
        else:
            hist = np.zeros(M, dtype=float)
            bin_edges = bins.copy()

        key = f"{int(a_deg):02d}-{int(b_deg):02d}_{direction}"
        histograms[key] = (hist, bin_edges)

    return histograms


def plot_energy_fluxes(
    velocities_list: list[np.ndarray],
    x: np.ndarray,
    mass: float,
    energy_bins: Union[int, np.ndarray],
    use_probs: bool = False,
    probs_list: Union[list[np.ndarray], None] = None,
    cmap: str = 'viridis',
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a 2-D heatmap (x vs Energy, colour scale = energy flux) from
    velocity vector lists across multiple series. The energy flux for each
    series is histogrammed as the sum of eV x v x (prob).

    Parameters
    ----------
    velocities_list : list[np.ndarray]
        List of length T, each element a velocity array of shape (NT, 3).
    x : np.ndarray, shape (T,)
        x-axis values corresponding to each velocity list.
    mass : float
        Particle mass (kg).
    energy_bins : int or np.ndarray, shape (M+1,)
        - int: let numpy.histogram auto-generate bin edges across all series.
        - np.ndarray: use these bin edges directly.
    use_probs : bool, default=False
        If True, read per-series probabilities from *probs_list* and
        include them in the energy-flux weights. If False, assume
        probs = np.ones(NT) (weight = eV x v only).
    probs_list : list[np.ndarray] | None, default=None
        List of length T, each element a probability array of shape (NT,).
        Required when use_probs=True.
    cmap : str, default='viridis'
        Matplotlib colourmap name.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The created Figure and Axes.
    """

    T = len(velocities_list)
    if x.shape[0] != T:
        raise ValueError("`x` and `velocities_list` must have the same length.")

    if use_probs:
        if probs_list is None or len(probs_list) != T:
            raise ValueError("When use_probs=True, probs_list must be a list of length T.")

    # Collect energies_eV across all series to obtain bin_edges
    all_energies_eV = []
    for v_arr in velocities_list:
        speeds = np.linalg.norm(v_arr, axis=1)
        energies_eV = (0.5 * mass * speeds**2) / e_charge
        all_energies_eV.append(energies_eV)
    all_energies_eV = np.concatenate(all_energies_eV)

    if isinstance(energy_bins, int):
        _, bin_edges = np.histogram(all_energies_eV, bins=energy_bins)
        bins = bin_edges
    else:
        bins = energy_bins.copy()
    M = len(bins) - 1

    energy_centers = 0.5 * (bins[:-1] + bins[1:])
    E_map = np.zeros((M, T), dtype=float)

    for j in range(T):
        v_arr = velocities_list[j]
        speeds = np.linalg.norm(v_arr, axis=1)
        energies_eV = (0.5 * mass * speeds**2) / e_charge

        if use_probs:
            probs = probs_list[j]
            if probs.shape[0] != v_arr.shape[0]:
                raise ValueError(f"probs_list[{j}] and velocities_list[{j}] must have the same length.")
        else:
            probs = np.ones_like(speeds)

        weights = energies_eV * speeds * probs
        hist, _ = np.histogram(energies_eV, bins=bins, weights=weights)
        E_map[:, j] = hist

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        E_map,
        origin='lower',
        aspect='auto',
        extent=[x[0], x[-1], energy_centers[0], energy_centers[-1]],
        norm=LogNorm(vmin=E_map[E_map > 0].min(), vmax=E_map.max()),
        cmap=cmap,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Energy Flux [eV·(m/s)·prob] (log scale)')

    ax.set_xlabel('x')
    ax.set_ylabel('Energy [eV]')
    ax.set_title('x vs Energy-Flux Map')

    plt.tight_layout()
    return fig, ax


def plot_energy_flux(
    velocities: np.ndarray,
    probs: np.ndarray,
    B: np.ndarray,
    mass: float,
    energy_bins: Union[int, np.ndarray],
    pitch_ranges: Union[list[tuple[float, float, str]], None] = None,
    cmap: str = 'plasma',
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the energy x energy-flux distribution for all particles and for
    each specified pitch-angle range, overlaid on a single axis.
    Energies are computed in eV.

    Parameters
    ----------
    velocities : np.ndarray, shape (N, 3)
        Velocity vectors for each particle (m/s). N is the number of samples.
    probs : np.ndarray, shape (N,)
        Probability or weight corresponding to each velocity vector.
    B : np.ndarray, shape (3,)
        Magnetic field vector (T) or direction vector.
    mass : float
        Particle mass (kg).
    energy_bins : int or np.ndarray, shape (M+1,)
        - int: let numpy.histogram auto-generate bin edges.
        - np.ndarray: use these bin edges directly.
    pitch_ranges : list of (a_deg, b_deg, direction) | None
        Pitch-angle ranges and direction specifier list. None uses the
        default 6-class decomposition.
    cmap : str, default='plasma'
        Matplotlib colourmap name.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The created Figure and Axes.
    """
    speeds = np.linalg.norm(velocities, axis=1)
    energies_eV = (0.5 * mass * speeds**2) / e_charge
    energy_flux = energies_eV * speeds * probs

    if isinstance(energy_bins, int):
        _, bin_edges = np.histogram(energies_eV, bins=energy_bins)
        bins = bin_edges
    else:
        bins = energy_bins.copy()
    M = len(bins) - 1
    centers = 0.5 * (bins[:-1] + bins[1:])

    total_hist, _ = np.histogram(energies_eV, bins=bins, weights=energy_flux)

    hists = compute_energy_flux_histograms(
        velocities=velocities,
        probs=probs,
        B=B,
        mass=mass,
        energy_bins=bins,
        pitch_ranges=pitch_ranges,
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(centers, total_hist, where='mid', label='All', color='black', linewidth=1.5)

    keys = sorted(hists.keys())
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(keys)))

    for color, key in zip(colors, keys):
        hist, _ = hists[key]
        ax.step(centers, hist, where='mid', label=key, color=color)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Energy [eV]')
    ax.set_ylabel('Energy  x  Flux  x  prob (arb.)')
    ax.set_title('Energy-Flux Distribution')
    ax.legend(fontsize='small', loc='upper right', ncol=2)
    plt.tight_layout()

    return fig, ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Sample data for 50 series
    T = 50
    x = np.linspace(0.0, 5.0, T)
    velocities_list = []
    probs_list = []

    np.random.seed(0)
    for _ in range(T):
        N_t = 200
        vels = np.random.normal(loc=0.0, scale=1e6, size=(N_t, 3))
        velocities_list.append(vels)
        probs_list.append(np.random.rand(N_t))

    m_e = 9.10938356e-31
    B = np.array([0.0, 0.0, 5e-9])

    # Test with integer energy_bins
    energy_bins_int = 30

    # 1) Test get_indices_in_pitch_range
    sample_vels = velocities_list[0]
    idx_20_50_pos = get_indices_in_pitch_range(
        velocities=sample_vels, B=B, a_deg=20.0, b_deg=50.0, direction='pos'
    )
    print("20-50 deg parallel particle count:", idx_20_50_pos.size)

    # 2) Test compute_energy_flux_histograms (int bins)
    hists = compute_energy_flux_histograms(
        velocities=sample_vels,
        probs=probs_list[0],
        B=B,
        mass=m_e,
        energy_bins=energy_bins_int,
    )
    print("Histogram keys:", list(hists.keys()))

    # 3) Test x vs Energy-Flux Map (int bins)
    fig1, ax1 = plot_energy_fluxes(
        velocities_list=velocities_list,
        x=x,
        mass=m_e,
        energy_bins=energy_bins_int,
        use_probs=True,
        probs_list=probs_list,
        cmap='plasma'
    )
    plt.show(fig1)

    # 4) Test Energy-Flux Distribution (int bins)
    fig2, ax2 = plot_energy_flux(
        velocities=sample_vels,
        probs=probs_list[0],
        B=B,
        mass=m_e,
        energy_bins=energy_bins_int,
    )
    plt.show(fig2)
