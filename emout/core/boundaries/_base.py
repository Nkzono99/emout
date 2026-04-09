"""Boundary base class for MPIEMSES finbound mode."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Set, Type

import numpy as np

from emout.plot.surface_cut import CompositeMeshSurface, MeshSurface3D


class Boundary:
    """Base class for a single finbound sub-boundary.

    Concrete subclasses know which ``&ptcond`` parameters to consult and how
    to construct their :class:`MeshSurface3D` representation. They must
    implement :meth:`_build_params` and :meth:`_build_mesh`, and they should
    declare :attr:`mesh_class` so :class:`BoundaryCollection.mesh` can filter
    broadcast keyword arguments to the kwargs the underlying mesh class
    actually accepts.

    Parameters
    ----------
    inp
        The :class:`emout.utils.InpFile` holding ``data.inp``'s namelist.
    unit
        The :class:`emout.utils.Units` instance for SI conversion (may be
        ``None`` if the simulation has no unit conversion key).
    index
        Zero-indexed position of this boundary inside ``boundary_types``.
    btype
        The MPIEMSES type string (``"sphere"``, ``"cylinderz"``, …). Kept so
        classes handling several axis variants can infer the axis letter.
    """

    #: The :class:`MeshSurface3D` subclass that this boundary builds. Used by
    #: :meth:`BoundaryCollection.mesh` to filter broadcast keyword arguments
    #: so that, for example, ``data.boundaries.mesh(theta_range=...)`` only
    #: forwards ``theta_range`` to boundaries whose mesh class actually
    #: accepts it. Set to ``None`` to opt out of filtering.
    mesh_class: Optional[Type[MeshSurface3D]] = None

    def __init__(self, inp, unit, index: int, btype: str):
        self.inp = inp
        self.unit = unit
        self.index = index
        self.fortran_index = index + 1
        self.btype = btype

    def __repr__(self) -> str:
        return f"<{type(self).__name__} index={self.index} btype={self.btype!r}>"

    # -- inp helpers ---------------------------------------------------------

    def _ptcond(self):
        return self.inp.nml["ptcond"]

    def _to_si_length(self, value):
        if self.unit is None:
            raise ValueError(
                "use_si=True requires a unit conversion key; this simulation "
                "has no data.unit."
            )
        return self.unit.length.reverse(value)

    # -- to be implemented by subclasses ------------------------------------

    def _build_params(self, use_si: bool) -> Dict[str, Any]:
        raise NotImplementedError

    def _build_mesh(self, params: Mapping[str, Any]) -> MeshSurface3D:
        raise NotImplementedError

    # -- public API ----------------------------------------------------------

    def mesh(self, *, use_si: bool = True, **overrides) -> MeshSurface3D:
        """Build the :class:`MeshSurface3D` for this boundary.

        Parameters auto-detected from ``data.inp`` may be overridden by
        passing keyword arguments. Overrides are interpreted in the same unit
        system as the returned mesh: SI metres by default
        (``use_si=True``), or simulation grid units when ``use_si=False``.
        """
        params = self._build_params(use_si=use_si)
        params.update(overrides)
        return self._build_mesh(params)

    # -- composition ---------------------------------------------------------

    def __add__(self, other):
        """Combine boundaries (or a boundary and another collection/mesh).

        ``boundary1 + boundary2`` and ``boundary + collection`` both return a
        :class:`BoundaryCollection` whose ``mesh()`` builds the composite,
        so the user can pick ``use_si``/``per`` at render time::

            mesh = (data.boundaries[0] + data.boundaries[2]).mesh(use_si=True)

        ``boundary + mesh_surface`` falls back to building this boundary's
        mesh in grid units and concatenating with the existing
        :class:`CompositeMeshSurface` ``+`` operator.
        """
        from ._collection import BoundaryCollection

        if isinstance(other, Boundary):
            unit = self.unit if self.unit is not None else other.unit
            return BoundaryCollection.from_boundaries([self, other], unit=unit)
        if isinstance(other, BoundaryCollection):
            unit = self.unit if self.unit is not None else other.unit
            return BoundaryCollection.from_boundaries([self, *other], unit=unit)
        if isinstance(other, MeshSurface3D):
            return self.mesh() + other
        return NotImplemented

    def __radd__(self, other):
        from ._collection import BoundaryCollection

        if other == 0:  # enables sum([b1, b2, ...])
            return BoundaryCollection.from_boundaries([self], unit=self.unit)
        if isinstance(other, MeshSurface3D):
            return other + self.mesh()
        return NotImplemented

    def render(self, *, use_si: bool = True, **style_kwargs):
        """Build the mesh and wrap it in a ``RenderItem`` for ``plot_surfaces``.

        Convenience that mirrors :meth:`MeshSurface3D.render`. ``use_si``
        defaults to ``True`` (SI metres) and is forwarded to :meth:`mesh`;
        remaining keyword arguments are passed through to ``RenderItem``
        (``style``, ``solid_color``, ``alpha``, …). For finer
        mesh-construction control, chain explicitly:
        ``boundary.mesh(ntheta=96).render(style="solid")``.
        """
        return self.mesh(use_si=use_si).render(**style_kwargs)


