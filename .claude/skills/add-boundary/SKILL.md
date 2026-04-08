---
name: add-boundary
description: Add a new MPIEMSES finbound boundary type to emout/emout/boundaries.py so data.boundaries[i] handles it. Use when the user asks for support of a new boundary_types(*) entry (e.g. hyperboloid-hole, plane-with-circle-hole, an extension type) or a legacy single-body boundary_type. Covers parameter reading, use_si conversion, _BOUNDARY_CLASS_MAP registration, and the testing pattern.
---

# add-boundary

Teach `data.boundaries` about a new MPIEMSES finbound (or legacy) boundary type.

## Prerequisites

- Read the relevant section of `/home/b/b36291/large0/Github/MPIEMSES3D/docs/Parameters.md` (or its English twin) for the new type's parameters.
- Confirm the actual Fortran implementation in `/home/b/b36291/large0/Github/MPIEMSES3D/src/physics/collision/objects.F90` (complex mode) or `surfaces.F90` (legacy `*-hole`). The docs sometimes lag the source; the source is the ground truth for which parameters are read and how indices map.
- If the new boundary needs a mesh shape that does not yet exist in `emout/plot/surface_cut/mesh.py`, run the `add-mesh-surface` skill first.
- Delegate deep investigation to the `finbound-investigator` agent if the parameter layout is unclear.

## Boundary class skeleton

Place the new class in `emout/emout/boundaries.py` near other boundaries of a similar shape (closed solids, planes, legacy pits, …).

```python
class NewBoundary(Boundary):
    """MPIEMSES `new-type` → :class:`NewShapeMeshSurface`."""

    @property
    def axis_letter(self) -> str:  # only if the type is axis-variant (e.g. newtypex/y/z)
        return self.btype[-1]

    def _build_params(self, use_si: bool) -> Dict[str, Any]:
        pt = self._ptcond()
        # For finbound complex mode, read indexed arrays via ib_fortran:
        origin = _get_vector(pt, "new_origin", self.fortran_index)
        radius = _get_scalar(pt, "new_radius", self.fortran_index)
        if origin is None or radius is None:
            raise ValueError(
                f"new_origin/new_radius not set for boundary index {self.index}"
            )
        origin = np.asarray(origin, dtype=np.float64)
        radius = float(radius)
        if use_si:
            origin = self._to_si_length(origin)
            radius = self._to_si_length(radius)
        return {"center": origin, "radius": radius}

    def _build_mesh(self, params: Mapping[str, Any]) -> MeshSurface3D:
        return NewShapeMeshSurface(**params)
```

Conventions to preserve:

- **All reads go through `_get_scalar` / `_get_vector`.** They handle f90nml's `start_index` correctly for both dense and sparse namelist arrays.
- **`_build_params` returns grid or SI depending on `use_si`.** Do not apply unit conversion in `_build_mesh` — overrides are passed through `mesh()` and should win without re-converting.
- **Missing parameters raise `ValueError` with a specific message.** Do not silently build a degenerate mesh.
- **Do not rename geometry kwargs.** The keys in the returned dict must exactly match the corresponding mesh class's `__init__` keyword arguments, because `_build_mesh` unpacks with `**params`.

## Registration

1. Add the new class name to `__all__` at the bottom of `boundaries.py`.
2. Register in `_BOUNDARY_CLASS_MAP`. For an axis-variant type, register all three axis letters pointing at the same class:
   ```python
   _BOUNDARY_CLASS_MAP = {
       ...
       "new-typex": NewBoundary,
       "new-typey": NewBoundary,
       "new-typez": NewBoundary,
   }
   ```
3. If the new type is also a legacy single-body mode (i.e. it can appear as `boundary_type = '<type>'` outside `complex`), append its name to `_LEGACY_SINGLE_BODY_TYPES`. `BoundaryCollection._build` already dispatches to `_BOUNDARY_CLASS_MAP[btype]` for legacy mode.
4. `SUPPORTED_BOUNDARY_TYPES` is auto-derived from `_BOUNDARY_CLASS_MAP` — no manual edit needed.

## Legacy `*-hole` conventions (if relevant)

Legacy `*-hole` boundaries read global scalars, not indexed arrays. Use `_safe_attr(self.inp, "zssurf")` and `_rectangle_hole_bounds(self.inp)` (already defined). Be aware of:

- `xl = xlrechole(1)`, `xu = xurechole(1)`, `yl = ylrechole(1)`, `yu = yurechole(1)`, `zl = zlrechole(2)`, `zu = zssurf`. The `zlrechole(2)` (Fortran index 2) is load-bearing and matches the Fortran source in `surfaces.F90`. Do not "normalize" it to index 1.
- Multiple legacy-named entries in `boundary_types(*)` will all share the same scalars — this matches the Fortran behavior. Document the limitation in a comment if you add a type that works this way.

## Test pattern

Add tests to `tests/test_boundaries.py`. The existing tests build an `InpFile` from a temp file using `_make_inp` (or inline `path.write_text(...)`). Template:

```python
def test_new_boundary_complex_mode(tmp_path: Path, unit: Units):
    inp = _make_inp(
        tmp_path,
        """\
&ptcond
    boundary_type = 'complex'
    boundary_types(1) = 'new-typez'
    new_origin(:, 1) = 10.0, 20.0, 30.0
    new_radius(1) = 2.0
/
""",
    )
    coll = BoundaryCollection(inp, unit)
    assert len(coll) == 1
    assert isinstance(coll[0], NewBoundary)

    mesh_obj = coll[0].mesh()
    assert isinstance(mesh_obj, NewShapeMeshSurface)
    assert np.allclose(mesh_obj.center, [10.0, 20.0, 30.0])
```

Always include at least:

- A complex-mode test that reads the parameters correctly.
- A `use_si=True` test that verifies the grid→SI conversion (remember that `dx=0.1` in the test fixture makes the conversion factor 10×).
- An override test: pass a resolution kwarg and verify it reaches the mesh class.
- If the type is axis-variant, cover at least one non-default axis.
- If the type is also a legacy single-body mode, add a second test with `boundary_type = '<type>'` at the top level.

Run the narrower suite first for fast iteration:

```bash
pytest tests/test_boundaries.py -q
```

Then the full baseline via the `run-tests` skill.

## Announcement

After landing, update `AGENTS.md §8` and/or `CLAUDE.md §8` only if the new type introduces a new *convention* (e.g. a new helper function, a new parameter idiom). Trivial additions that follow the existing pattern do not need doc updates — `SUPPORTED_BOUNDARY_TYPES` is the runtime source of truth.
