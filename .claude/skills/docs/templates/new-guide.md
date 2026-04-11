<!--
  New guide page skeleton (English mirror).
  Copy to `docs/source/guide/<name>.md` after the Japanese canonical
  version `<name>.ja.md` has been drafted from
  `.claude/skills/docs/templates/new-guide.ja.md`. Structure, section
  order, tables, and code blocks must match the Japanese file exactly.

  Browse the existing guides (quickstart / plotting / animation / inp /
  units / boundaries / backtrace / distributed) to match the section
  granularity and length. When in doubt, `backtrace.md` is the closest
  structural reference.
-->

# <Feature name> (`<canonical API call>`) <— Experimental if applicable>

<1-3 sentence introduction. What the feature does and who the target
reader is. No code snippets — prose only.>

> **Requirements (if any):** external dependencies (e.g.
> `vdist-solver-fortran`), optional extras (e.g.
> `pip install "emout[pyvista]"`), Python version constraints. Mention
> the failure mode when the dependency is missing (e.g. `ImportError`).

## When to use it

- <Use case 1>
- <Use case 2>
- <Use case 3>

<If the feature has a significant cost or side-effect, add a one-paragraph
note here. e.g. "Large max_step values make the call expensive, so prefer
running it on an HPC node.">

## Quick start

```python
import emout

data = emout.Emout("output_dir")

# <Shortest idiomatic usage — single line if possible>
<code>

# <Alternate entry point if one exists>
<code>
```

## <Primary API 1>: `<method_name>`

`<method_name>(arg1, arg2, ...)` <one-line description>. Returns a
:class:`<ReturnType>`, <one-line qualifier>.

```python
<code example>
```

| Attribute / arg | Shape / type | Meaning |
| --- | --- | --- |
| `<attr>` | `(N,)` | <description> |
| `<attr>` | `(N, 3)` | <description> |

### Shorthand access (if applicable)

<Describe shorthand attribute access patterns such as `.vxvz.plot()` or
`__getattr__`-based pair lookups. `backtrace.md` is the reference.>

```python
<code>
```

## <Primary API 2>: `<method_name>`

<Repeat the same structure. Aim for 2-4 primary API sections.>

## Remote execution integration (if applicable)

`data.remote()` supports the same API. <Note which route returns a
dedicated proxy vs a generic RemoteRef.>

```python
from emout.distributed import remote_figure, remote_scope

rdata = emout.Emout("output_dir").remote()

with remote_scope():
    <remote usage example>
```

See the [remote execution guide](distributed.md) for details.

## Common gotchas

- <Gotcha 1>
- <Gotcha 2>
- <Gotcha 3>

## Related classes

See the API reference (the `<emout.sub.package>` package) for full
signatures.

- `<ClassName1>` — <role>
- `<ClassName2>` — <role>
