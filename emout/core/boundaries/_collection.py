"""BoundaryCollection and type dispatch for MPIEMSES boundaries."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple


from emout.plot.surface_cut import (
    CompositeMeshSurface,
    MeshSurface3D,
)

from ._base import Boundary
from ._helpers import _accepted_kwargs, _safe_attr
from ._complex_types import (
    CircleBoundary,
    CuboidBoundary,
    CylinderBoundary,
    DiskBoundary,
    PlaneWithCircleBoundary,
    RectangleBoundary,
    SphereBoundary,
)
from ._legacy_types import (
    CylinderHoleBoundary,
    FlatSurfaceBoundary,
    RectangleHoleBoundary,
)


# ---------------------------------------------------------------------------
# Type → class dispatch
# ---------------------------------------------------------------------------


_BOUNDARY_CLASS_MAP: Dict[str, type] = {
    # Finbound "complex" mode primitives (docs Parameters.md §C).
    "sphere": SphereBoundary,
    "cuboid": CuboidBoundary,
    "rectangle": RectangleBoundary,
    "cylinderx": CylinderBoundary,
    "cylindery": CylinderBoundary,
    "cylinderz": CylinderBoundary,
    "open-cylinderx": CylinderBoundary,
    "open-cylindery": CylinderBoundary,
    "open-cylinderz": CylinderBoundary,
    "diskx": DiskBoundary,
    "disky": DiskBoundary,
    "diskz": DiskBoundary,
    "circlex": CircleBoundary,
    "circley": CircleBoundary,
    "circlez": CircleBoundary,
    "plane-with-circlex": PlaneWithCircleBoundary,
    "plane-with-circley": PlaneWithCircleBoundary,
    "plane-with-circlez": PlaneWithCircleBoundary,
    # Legacy single-body modes that also appear in `boundary_types(*)` in
    # MPIEMSES3D src/physics/collision/surfaces.F90. Each pulls global
    # scalars (`zssurf`, `xlrechole`, …) rather than indexed arrays.
    "flat-surface": FlatSurfaceBoundary,
    "rectangle-hole": RectangleHoleBoundary,
    "cylinder-hole": CylinderHoleBoundary,
}


#: The set of `boundary_type` values that, when set *without* `complex`,
#: describe a single-body simulation. These are the legacy finbound-extension
#: modes handled directly via :class:`BoundaryCollection`.
_LEGACY_SINGLE_BODY_TYPES: Tuple[str, ...] = (
    "flat-surface",
    "rectangle-hole",
    "cylinder-hole",
)


SUPPORTED_BOUNDARY_TYPES: Tuple[str, ...] = tuple(sorted(_BOUNDARY_CLASS_MAP))


# ---------------------------------------------------------------------------
# BoundaryCollection
# ---------------------------------------------------------------------------


def _offset_mesh(mesh_surface, offsets):
    """Return a copy of *mesh_surface* with vertices shifted by *offsets*.

    Parameters
    ----------
    mesh_surface : MeshSurface3D
        Source mesh.
    offsets : tuple of (float or str)
        ``(x_offset, y_offset, z_offset)``.

    Returns
    -------
    MeshSurface3D
        Shifted mesh (a lightweight wrapper).
    """
    from emout.utils.util import apply_offset

    V, F = mesh_surface.mesh()
    V = V.copy()
    for axis_idx in range(min(3, len(offsets))):
        if offsets[axis_idx] is not None:
            V[:, axis_idx] = apply_offset(V[:, axis_idx].copy(), offsets[axis_idx])

    class _OffsetMesh(MeshSurface3D):
        def mesh(self):
            return V, F

    return _OffsetMesh()


class BoundaryCollection:
    """Collection of boundaries discovered in ``data.inp``'s finbound config.

    Supports ``len``, iteration, and integer indexing. ``mesh()`` returns a
    composite :class:`MeshSurface3D` concatenating every boundary, with
    optional per-index or common keyword overrides.

    Boundaries of unsupported types (or legacy single-body modes outside
    ``boundary_type = 'complex'``) are silently skipped. A
    :attr:`skipped` list records the reason per skipped slot for debugging.
    """

    def __init__(self, inp, unit, remote_open_kwargs: Optional[Mapping[str, Any]] = None):
        self.inp = inp
        self.unit = unit
        self._emout_open_kwargs = None if remote_open_kwargs is None else dict(remote_open_kwargs)
        self.skipped: List[Tuple[int, str, str]] = []
        self._boundaries = self._build()

    @classmethod
    def from_boundaries(
        cls,
        boundaries: "Iterable[Boundary]",
        unit=None,
    ) -> "BoundaryCollection":
        """Build a collection from an explicit iterable of boundaries.

        Skips the namelist parsing path used by ``__init__``. Useful when
        composing boundaries with ``+`` (``boundary1 + boundary2``) or
        constructing ad-hoc subsets for plotting and tests.

        Parameters
        ----------
        boundaries
            Iterable of :class:`Boundary` instances. ``BoundaryCollection``
            children are flattened.
        unit
            Optional :class:`emout.utils.Units` for SI conversion. Falls back
            to the first non-``None`` ``unit`` found on the input boundaries.
        """
        flat: List[Boundary] = []
        for item in boundaries:
            if isinstance(item, BoundaryCollection):
                flat.extend(item)
            else:
                flat.append(item)
        if unit is None:
            for b in flat:
                if getattr(b, "unit", None) is not None:
                    unit = b.unit
                    break
        instance = cls.__new__(cls)
        instance.inp = None
        instance.unit = unit
        instance._emout_open_kwargs = None
        instance.skipped = []
        instance._boundaries = flat
        return instance

    # -- construction --------------------------------------------------------

    def _build(self) -> List[Boundary]:
        if self.inp is None:
            return []

        btype = _safe_attr(self.inp, "boundary_type")
        if btype is None:
            return []

        if btype == "complex":
            return self._build_complex()

        # Legacy single-body mode: one boundary whose global scalars
        # (``zssurf``, ``xlrechole``, …) live directly in ``&ptcond``.
        if btype in _LEGACY_SINGLE_BODY_TYPES:
            cls = _BOUNDARY_CLASS_MAP[btype]
            return [cls(self.inp, self.unit, 0, btype)]

        self.skipped.append((0, btype, "unsupported top-level boundary_type"))
        return []

    def _build_complex(self) -> List[Boundary]:
        btypes_raw = _safe_attr(self.inp, "boundary_types")
        if btypes_raw is None:
            return []
        if isinstance(btypes_raw, str):
            btypes_raw = [btypes_raw]

        built: List[Boundary] = []
        for ib, raw in enumerate(btypes_raw):
            if raw is None:
                self.skipped.append((ib, "", "slot is None"))
                continue
            if not isinstance(raw, str):
                self.skipped.append((ib, repr(raw), "non-string boundary type"))
                continue
            name = raw.strip()
            if not name:
                self.skipped.append((ib, raw, "empty boundary type"))
                continue
            cls = _BOUNDARY_CLASS_MAP.get(name)
            if cls is None:
                self.skipped.append((ib, name, "unsupported boundary type"))
                continue
            built.append(cls(self.inp, self.unit, ib, name))
        return built

    # -- container protocol --------------------------------------------------

    def __len__(self) -> int:
        return len(self._boundaries)

    def __iter__(self) -> Iterator[Boundary]:
        return iter(self._boundaries)

    def __getitem__(self, idx):
        return self._boundaries[idx]

    def __bool__(self) -> bool:
        return bool(self._boundaries)

    def __repr__(self) -> str:
        types = ", ".join(b.btype for b in self._boundaries)
        return f"<BoundaryCollection [{types}]>"

    # -- composition ---------------------------------------------------------

    def __add__(self, other):
        if isinstance(other, BoundaryCollection):
            unit = self.unit if self.unit is not None else other.unit
            return type(self).from_boundaries([*self, *other], unit=unit)
        if isinstance(other, Boundary):
            unit = self.unit if self.unit is not None else other.unit
            return type(self).from_boundaries([*self, other], unit=unit)
        if isinstance(other, MeshSurface3D):
            return self.mesh() + other
        return NotImplemented

    def __radd__(self, other):
        if other == 0:  # sum([...]) compatibility
            return self
        if isinstance(other, MeshSurface3D):
            return other + self.mesh()
        return NotImplemented

    def render(
        self,
        *,
        use_si: bool = True,
        per: Optional[Mapping[int, Mapping[str, Any]]] = None,
        **style_kwargs,
    ):
        """Build the composite mesh and wrap it in a ``RenderItem``.

        ``use_si`` defaults to ``True`` (SI metres). ``use_si`` and ``per``
        are forwarded to :meth:`mesh`; remaining keyword arguments are
        forwarded to ``RenderItem`` (``style``, ``solid_color``, ``alpha``,
        …).
        """
        return self.mesh(use_si=use_si, per=per).render(**style_kwargs)

    def plot(
        self,
        *,
        ax=None,
        use_si: bool = True,
        offsets=None,
        per: Optional[Mapping[int, Mapping[str, Any]]] = None,
        style: str = "solid",
        solid_color="0.7",
        alpha: float = 0.6,
        **kwargs,
    ):
        """Plot boundary meshes in 3-D.

        Use this to inspect boundary geometry without a field overlay.
        To overlay on field data, use ``Data3d.plot_surfaces(data.boundaries)``.

        Parameters
        ----------
        ax : Axes3D, optional
            Target 3-D axes. Created if ``None``.
        use_si : bool, default True
            Convert to SI metres.
        offsets : tuple of (float or str), optional
            ``(x_offset, y_offset, z_offset)`` applied to mesh vertices.
            Accepts ``"left"``, ``"center"``, ``"right"`` or numeric values.
        per : dict, optional
            Per-boundary mesh overrides.
        style, solid_color, alpha
            Rendering style forwarded to :class:`RenderItem`.
        """
        from emout.distributed.remote_figure import (
            is_recording,
            record_boundary_plot,
            request_session,
        )
        import matplotlib.pyplot as plt
        from emout.plot.surface_cut import plot_surfaces as _plot_surfaces, RenderItem

        if is_recording():
            emout_kw = getattr(self, "_emout_open_kwargs", None)
            request_session(emout_kw)
            record_boundary_plot(
                {
                    "use_si": use_si,
                    "offsets": offsets,
                    "per": per,
                    "style": style,
                    "solid_color": solid_color,
                    "alpha": alpha,
                    **kwargs,
                },
                emout_kwargs=emout_kw,
            )
            return None

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        composite = self.mesh(use_si=use_si, per=per)

        if offsets is not None:
            composite = _offset_mesh(composite, offsets)

        item = RenderItem(composite, style=style, solid_color=solid_color, alpha=alpha, **kwargs)
        _plot_surfaces(ax, field=None, surfaces=item)
        return ax

    # -- composite mesh ------------------------------------------------------

    def mesh(
        self,
        *,
        use_si: bool = True,
        per: Optional[Mapping[int, Mapping[str, Any]]] = None,
        **common_overrides,
    ) -> MeshSurface3D:
        """Return the composite mesh of all recognised boundaries.

        Parameters
        ----------
        use_si : bool, default True
            Convert geometry to SI metres via ``data.unit.length.reverse``.
            Pass ``use_si=False`` to keep simulation grid units instead.
        per : dict, optional
            Mapping from boundary index (0-based) to a dict of overrides
            passed to that boundary's ``mesh()`` call. Per-index entries are
            merged on top of (and override) the broadcast ``common_overrides``
            and **bypass the kwarg filter** — they go straight through to
            ``boundary.mesh()``.
        **common_overrides
            Overrides broadcast to every boundary. Each boundary only sees
            the subset of these kwargs that its underlying ``mesh_class``
            ``__init__`` actually accepts (introspected via
            :func:`inspect.signature`). For example::

                data.boundaries.mesh(theta_range=[0, np.pi])

            forwards ``theta_range`` to the sphere/cylinder/disk/circle/
            plane-with-circle entries that support it and silently drops it
            for cuboid/rectangle/flat-surface entries.
        """
        per = per or {}
        children: List[MeshSurface3D] = []
        for boundary in self._boundaries:
            accepted = _accepted_kwargs(getattr(type(boundary), "mesh_class", None))
            if accepted is None:
                # No introspection possible — broadcast everything verbatim.
                extra: Dict[str, Any] = dict(common_overrides)
            else:
                extra = {k: v for k, v in common_overrides.items() if k in accepted}
            # Per-boundary overrides always win, and bypass the filter so the
            # caller can target a specific entry with kwargs the broadcast
            # filter would otherwise drop.
            extra.update(per.get(boundary.index, {}))
            children.append(boundary.mesh(use_si=use_si, **extra))
        return CompositeMeshSurface(children)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


__all__ = [
    "Boundary",
    "BoundaryCollection",
    "SphereBoundary",
    "CuboidBoundary",
    "RectangleBoundary",
    "CircleBoundary",
    "CylinderBoundary",
    "DiskBoundary",
    "PlaneWithCircleBoundary",
    "FlatSurfaceBoundary",
    "RectangleHoleBoundary",
    "CylinderHoleBoundary",
    "SUPPORTED_BOUNDARY_TYPES",
]
