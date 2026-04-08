---
name: add-mesh-surface
description: Add a new MeshSurface3D subclass to emout/plot/surface_cut/mesh.py. Use when the user asks for a new explicit mesh shape (e.g. a torus, cone, partial sphere) or when a new MPIEMSES finbound boundary type needs a dedicated mesh class. Covers the shared helper conventions, __all__ plumbing, and testing pattern.
---

# add-mesh-surface

Drop a new `MeshSurface3D` subclass into `emout/plot/surface_cut/mesh.py` without breaking the conventions the existing classes rely on.

## Before you write any code

1. `Read` `emout/plot/surface_cut/mesh.py` top to bottom. It is one file. Look at 2–3 existing classes that are structurally closest to what you want:
   - Closed solid with caps → `CylinderMeshSurface`, `DiskMeshSurface`, `SphereMeshSurface`.
   - Flat panel → `RectangleMeshSurface`, `CircleMeshSurface`, `PlaneWithCircleMeshSurface`.
   - Slab with a hole → `HollowCylinderMeshSurface`.
   - Cuboid-faced solid → `BoxMeshSurface`.
2. Decide which shared helper you will build on. Do **not** reimplement triangulation if a helper exists. Helpers, roughly from simplest to most flexible:
   - `_plane_mesh(points, expected_normal, wrap_u)` — triangulate a structured `(nv, nu, 3)` grid, optionally wrapping in the u direction.
   - `_disc_mesh(base, e1, e2, radius, ntheta, nradial, expected_normal, theta_range=None)` — filled disc (fan + rings), supports partial angular range.
   - `_annulus_mesh(...)` — flat annulus between two radii.
   - `_rect_with_hole_mesh(...)` — flat rectangle with a central circular hole.
   - `_grid_faces(nv, nu, wrap_u)` — raw triangle index generation; use only when `_plane_mesh` is too coarse.
   - `_combine_meshes(meshes)` — stitch multiple `(V, F)` pairs into one.
3. Decide your frame. Every non-axis-aligned mesh is built on top of `_orthonormal_frame(axis)` which returns `(axis_unit, e1, e2)`. The `e1`/`e2` pair is arbitrary but orthonormal — always route geometry through them instead of assuming `(x, y, z)`.

## Class skeleton

```python
class NewShapeMeshSurface(MeshSurface3D):
    """One-line description.

    Longer docstring explaining the geometry and what MPIEMSES primitive
    (if any) this corresponds to.
    """

    def __init__(
        self,
        center: Union[Tuple[float, float], Tuple[float, float, float], np.ndarray],
        axis: AxisSpec,
        # ... geometry params in grid units ...
        *,
        # resolution params, theta_range, parts, flip_normal, ...
    ):
        self.center = _center_to_3vec(center)
        self.axis, self.e1, self.e2 = _orthonormal_frame(axis)
        # validate + stash params
        self.ntheta = _normalize_count(ntheta, name="ntheta", minimum=3)
        # etc.

    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        # Build (nv, nu, 3) points or call a helper.
        return _plane_mesh(points, expected_normal=tuple(self.axis), wrap_u=False)
```

Conventions:

- Every length is in **grid units**. Unit conversion (`use_si`) is the boundary layer's concern, not the mesh class.
- Axial range (`length` vs `tmin`/`tmax`) goes through `_axial_range(length=..., tmin=..., tmax=...)`.
- Angular range goes through `_resolve_theta_range(theta_range)` + `_sample_theta(...)`.
- If your shape has multiple parts (e.g. side + top + bottom), accept a `parts=` kwarg and normalize with `_normalize_selection(value, allowed=self._allowed_parts, name="parts")`.
- If your shape has a natural outward normal, accept `flip_normal=False` and, at `mesh()` time, pass the flipped or unflipped `axis` to `_plane_mesh` via `expected_normal`.
- Validation errors are plain `ValueError` with a short, specific message.

## Wiring checklist

When the class is written:

1. Add it to the `__all__` list at the bottom of `mesh.py`.
2. Add it to the `from .mesh import (...)` block **and** the `__all__` list in `emout/plot/surface_cut/__init__.py`.
3. If this mesh is meant to be consumed by `data.boundaries[i]`, also add a matching `Boundary` subclass via the `add-boundary` skill and register it in `_BOUNDARY_CLASS_MAP`.
4. Run `run-tests` skill or `pytest tests/plot/test_surface_cut_mesh.py -q` at minimum.

## Test pattern

Add a test to `tests/plot/test_surface_cut_mesh.py`. Look at `test_sphere_mesh_surface_points_lie_on_sphere` or `test_rectangle_mesh_surface_builds_flat_panel` as templates. Assertions worth including:

- Shape of `V` and `F` (counts depend on `ntheta`/`nradial` — compute, do not hardcode).
- Geometric invariants: points lie on the expected surface, extrema match the declared bounds, no degenerate triangles when they shouldn't exist.
- For closed solids, the mesh is watertight-enough for visualization (every edge appears in ≥ 1 triangle — you rarely need to strictly check watertightness for viz meshes).
- If the class supports `theta_range`, include a half-section test.
- If the class supports `flip_normal` or face selection, test at least one non-default choice.

## Common mistakes

- **Forgetting `wrap_u=True` on cylindrical/annular patches.** Missing wrap leaves a visible seam.
- **Passing `expected_normal` in the wrong sign.** `_orient_faces_to_normal` flips winding to match, so if the result renders inside-out, flip the sign you pass.
- **Mutating shared numpy arrays.** `center`, `e1`, `e2` are stored on `self`; if you compute `points = center[None, None, :] + ...`, do not then write back into `center`.
- **Accepting a `length` AND `tmin`/`tmax`.** `_axial_range` rejects this combination. Let the helper enforce it.
- **Hard-coding `(0, 0, 1)` as the axis.** Always go through `axis, e1, e2 = _orthonormal_frame(axis)`.
