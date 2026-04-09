Lang: [English](inp.md) | [日本語](inp.ja.md)

# Parameter File (`data.inp`)

emout reads the EMSES parameter file (`plasma.inp` or `plasma.toml`) and exposes it as a dictionary-like object with attribute access.

## Accessing Parameters

```python
import emout

data = emout.Emout("output_dir")

# Dictionary-style access with group name
data.inp["tmgrid"]["nx"]    # → e.g., 256
data.inp["plasma"]["wp"]    # → e.g., [1.0, 0.05]

# Group name can be omitted if the parameter name is unambiguous
data.inp["nx"]              # → same as data.inp["tmgrid"]["nx"]

# Attribute-style access
data.inp.tmgrid.nx
data.inp.nx                 # Group name omitted
```

## Commonly Used Parameters

### Grid

| Parameter | Group | Description |
| --- | --- | --- |
| `nx`, `ny`, `nz` | `tmgrid` | Grid dimensions |
| `dx`, `dy`, `dz` | `tmgrid` | Grid spacing (in EMSES units; see `data.unit.length` for SI) |

```python
nx, ny, nz = data.inp.nx, data.inp.ny, data.inp.nz
```

### Time

| Parameter | Group | Description |
| --- | --- | --- |
| `dt` | `tmgrid` | Time step |
| `ifdiag` | `tmgrid` | Output interval (every `ifdiag` steps) |
| `nstep` | `tmgrid` | Total number of steps |

```python
dt = data.inp.dt
total_time_steps = data.inp.nstep
output_interval = data.inp.ifdiag
```

### Plasma

| Parameter | Group | Description |
| --- | --- | --- |
| `nspec` | `plasma` | Number of particle species |
| `wp` | `plasma` | Plasma frequency per species |
| `qm` | `plasma` | Charge-to-mass ratio per species |
| `path` | `plasma` | Thermal velocity per species |
| `peth` | `plasma` | Thermal velocity (alternative) |

```python
nspec = data.inp.nspec
wp = data.inp.wp        # list of plasma frequencies
qm = data.inp.qm        # list of charge-to-mass ratios
```

### Boundaries

| Parameter | Group | Description |
| --- | --- | --- |
| `mtd_vbnd` | `emissn` | Boundary type per axis (0=periodic, 1=Dirichlet, 2=Neumann) |

```python
btypes = data.inp.mtd_vbnd  # e.g., [0, 2, 0] for periodic-Neumann-periodic
```

## TOML Format (`plasma.toml`)

When `plasma.toml` exists, emout automatically runs `toml2inp` to generate `plasma.inp`, then loads it.
The `toml2inp` command is bundled with [MPIEMSES3D](https://github.com/Nkzono99/MPIEMSES3D).

```python
data = emout.Emout("output_dir")
data.inp.nx          # Same interface regardless of file format
```

### Accessing Raw TOML Structure

Use `data.toml` for direct access to the native TOML structure:

```python
data.toml                        # TomlData wrapper (None if plasma.inp only)
data.toml.tmgrid.nx              # attribute access
data.toml["tmgrid"]["nx"]        # dict-style access
data.toml.species[0].wp          # nested structures
```

The TOML format uses section headers corresponding to namelist groups:

```toml
[tmgrid]
nx = 256
ny = 256
nz = 512
dt = 0.5

[[species]]
wp = 1.0
qm = -1.0

[[species]]
wp = 0.05
qm = 0.001
```

### Separating Input and Output

```python
data = emout.Emout(input_path="/path/to/plasma.toml", output_directory="output_dir")
```

## Unit Conversion Key

The first line of `plasma.inp` may contain a unit conversion key:

```text
!!key dx=[0.5],to_c=[10000.0]
```

- `dx`: Grid spacing in meters [m]
- `to_c`: Speed of light in EMSES internal units

This key enables SI unit conversion via `data.unit`. In `plasma.toml`, this is expressed as:

```toml
[meta.unit_conversion]
dx = 0.5
to_c = 10000.0
```

If no conversion key is present, `data.unit` will be `None` and SI features (`val_si`, `use_si=True`) will fall back to raw EMSES units.
