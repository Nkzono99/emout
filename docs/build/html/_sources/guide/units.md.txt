# Unit Conversion (`data.unit`)

emout provides bidirectional conversion between EMSES internal (normalized) units and SI units.

## Prerequisites

Unit conversion requires a conversion key in the parameter file.

**`plasma.inp` format** — first line:

```text
!!key dx=[0.5],to_c=[10000.0]
```

**`plasma.toml` format:**

```toml
[meta.unit_conversion]
dx = 0.5
to_c = 10000.0
```

- `dx`: Grid spacing in meters [m]
- `to_c`: Speed of light in EMSES normalized units

If no key is present, `data.unit` is `None`.

## Using Unit Translators

Each physical quantity has a `UnitTranslator` accessible via `data.unit.<name>`:

```python
import emout

data = emout.Emout("output_dir")

# SI → EMSES
emses_velocity = data.unit.v.trans(1e5)     # 1e5 m/s → EMSES

# EMSES → SI
si_velocity = data.unit.v.reverse(4.107)    # 4.107 EMSES → m/s
```

### `trans(value)` and `reverse(value)`

| Method | Direction | Use case |
| --- | --- | --- |
| `trans(x)` | SI → EMSES | Setting initial conditions, comparing with theory |
| `reverse(x)` | EMSES → SI | Interpreting simulation results |

## Direct SI Values from Data

The `.val_si` property on any grid data array returns the values in SI units:

```python
# Potential in Volts
phisp_V = data.phisp[-1].val_si

# Current density in A/m^2
j1z_A_m2 = data.j1z[-1].val_si

# Number density in /m^3
nd1p_m3 = data.nd1p[-1].val_si
```

This also works for sliced data:

```python
# SI values for a 2D slice
phi_slice = data.phisp[-1, :, 32, :].val_si
```

## SI Units in Plots

By default, `plot()` uses SI units for axis labels and colorbar:

```python
data.phisp[-1, 100, :, :].plot()              # SI units (default)
data.phisp[-1, 100, :, :].plot(use_si=False)  # EMSES units
```

## Available Unit Translators

| Name | Quantity | SI Unit |
| --- | --- | --- |
| `phi` | Potential | V |
| `E` | Electric field | V/m |
| `B` | Magnetic flux density | T |
| `J` | Current density | A/m^2 |
| `n` | Number density | /m^3 |
| `rho` | Charge density | C/m^3 |
| `v` | Velocity | m/s |
| `t` | Time | s |
| `f` | Frequency | Hz |
| `length` | Length | m |
| `q` | Charge | C |
| `m` | Mass | kg |
| `W` | Energy | J |
| `w` | Energy density | J/m^3 |
| `P` | Power | W |
| `T` | Temperature | K |
| `F` | Force | N |
| `a` | Acceleration | m/s^2 |
| `i` | Current | A |
| `N` | Flux | /m^2s |
| `c` | Speed of light | m/s |
| `eps` | Permittivity | F/m |
| `mu` | Permeability | H/m |
| `C` | Capacitance | F |
| `L` | Inductance | H |
| `G` | Conductance | S |
| `q_m` | Charge-to-mass ratio | C/kg |
| `qe` | Elementary charge | C |
| `qe_me` | Electron charge-to-mass ratio | C/kg |
| `kB` | Boltzmann constant | J/K |
| `e0` | Vacuum permittivity | F/m |
| `m0` | Vacuum permeability | N/A^2 |

## Custom Time-Axis Units

By default, the time axis in plots uses seconds (SI). You can switch to plasma-frequency-normalized time ($\omega_{pe} t$):

```python
from emout.emout.units import wpet_unit

# Register globally for all subsequent plots
emout.Emout.name2unit["t"] = wpet_unit
```

## Combining Translators

`UnitTranslator` objects support multiplication to compose conversions:

```python
# Time axis in output steps → SI seconds
t_translator = data.unit.t * UnitTranslator(data.inp.ifdiag * data.inp.dt, 1)
```
