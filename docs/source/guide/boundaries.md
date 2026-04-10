# Boundary Meshes (`data.boundaries`)

The `finbound` / legacy boundaries defined under MPIEMSES's `&ptcond` are exposed
as Python objects via `data.boundaries`. Each boundary produces a `MeshSurface3D`
that you can overlay onto a 3D field plot or style individually.

## Access

```python
# The whole collection
data.boundaries                 # BoundaryCollection (iterable, indexable)
len(data.boundaries)
data.boundaries.types           # list of boundary_types strings
data.boundaries.skipped         # [(index, type_name, reason), ...]

# An individual boundary
data.boundaries[0]              # subclass (SphereBoundary, CylinderBoundary, ...)
data.boundaries[0].btype        # e.g. "sphere", "cylinderz"
data.boundaries[0].mesh()       # → MeshSurface3D (SI units by default)
data.boundaries[0].mesh(use_si=False)  # keep grid units
```

## Overlay on a 3D field plot

Pass `data.boundaries` straight to `Data3d.plot_surfaces` to draw the
geometries on top of isosurfaces or slices.

```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

data.phisp[-1].plot_surfaces(
    ax=ax,
    surfaces=data.boundaries,         # auto-wrapped into RenderItems
)
plt.show()
```

## Composite mesh

All boundaries can be fused into a single mesh for further processing:

```python
composite = data.boundaries.mesh()    # → CompositeMeshSurface
V, F = composite.mesh()               # raw (vertices, faces) arrays
```

## Composition and per-boundary styling

```python
# Boundary + Boundary → BoundaryCollection
combined = data.boundaries[0] + data.boundaries[1]

# Different style per boundary
data.phisp[-1].plot_surfaces(
    ax=ax,
    surfaces=data.boundaries.render(
        per={
            0: dict(style="solid", solid_color="0.7"),
            1: dict(alpha=0.5),
        },
    ),
)
```

## Overriding mesh parameters

You can pass keyword arguments through `mesh()` to adjust resolution or geometry
without modifying the simulation parameters:

```python
# Higher angular resolution on a specific boundary
data.boundaries[0].mesh(ntheta=64)

# Per-boundary overrides via the collection
data.boundaries.mesh(per={0: dict(ntheta=64)})
```

## Supported boundary types

Both `boundary_type = "complex"` with `boundary_types(i)` and the legacy
single-body modes are supported.

| Category | Type names |
| --- | --- |
| Closed solids | `sphere`, `cuboid` |
| Cylinders | `cylinderx/y/z`, `open-cylinderx/y/z` |
| Flat panels | `rectangle`, `circlex/y/z`, `diskx/y/z`, `plane-with-circlex/y/z` |
| Legacy single-body | `flat-surface`, `rectangle-hole`, `cylinder-hole` |

See [MPIEMSES3D Parameters.md](https://github.com/Nkzono99/MPIEMSES3D/blob/main/docs/Parameters.md)
for the full parameter specification.

Unregistered types are recorded in `data.boundaries.skipped` as
`(index, type_name, reason)` tuples rather than raising an error.
You can inspect this list to check whether any boundaries were silently ignored.

## Available mesh surface classes

Each boundary type maps to one of these `MeshSurface3D` subclasses
(importable from `emout.plot.surface_cut`):

| Class | Used by |
| --- | --- |
| `SphereMeshSurface` | `sphere` |
| `BoxMeshSurface` | `cuboid` |
| `RectangleMeshSurface` | `rectangle` |
| `CircleMeshSurface` | `circlex/y/z` |
| `CylinderMeshSurface` | `cylinderx/y/z`, `open-cylinderx/y/z` |
| `DiskMeshSurface` | `diskx/y/z` |
| `PlaneWithCircleMeshSurface` | `plane-with-circlex/y/z` |
| `HollowCylinderMeshSurface` | `cylinder-hole` |
| `CompositeMeshSurface` | collection-level `mesh()` output |

These classes can also be instantiated directly for standalone mesh construction
without going through `data.boundaries`.
