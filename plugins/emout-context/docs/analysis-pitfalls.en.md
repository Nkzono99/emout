Lang: [English](analysis-pitfalls.en.md) | [日本語](analysis-pitfalls.md)

# emout Analysis Pitfalls

The same classes of mistakes often appear in emout user questions. During diagnosis, check the output directory, input file, data size, and execution environment, not just the traceback.

## Common Causes

| Symptom | Check | Fix |
| --- | --- | --- |
| The plotted plane is not the expected one | Whether slicing follows axis order `(t, z, y, x)` | For an xz-plane, use a form such as `data.phisp[-1, :, y_index, :]` |
| Out-of-memory errors or slow processing | Whether the full 4D array is loaded with `.val` / `.val_si` | Slice by time, plane, and range before reading values |
| `val_si` or displayed units look wrong | Whether `!!key` or `[meta.unit_conversion]` is present | Check unit conversion metadata; without it, treat values as EMSES units |
| A variable attribute is missing | Whether HDF5 filenames map to the expected EMSES variable names | Check `ls *.h5` and the variable resolution rules in the guides |
| Vector plots do not match expectations | Whether `j1xy` / `j1xyz` components match the slice plane | Choose the vector attribute that matches the view plane |
| 3D plotting raises an import error | Whether PyVista is installed | Suggest `pip install "emout[pyvista]"` |
| Boundaries do not render | Whether `data.boundaries` is empty or the finbound shape is supported | Check the input boundary settings and the boundaries guide |
| Remote plotting fails | Python 3.10+, Dask server, TLS settings, session name | Check `emout server status` and the distributed guide |
| Continuation output is not joined | Whether appended output was requested | Try `emout.Emout("output_dir", ad="auto")` |

## Information To Collect

- Output of `python -m pip show emout` or `python -c "import importlib.metadata as m; print(m.version('emout'))"`
- Installation method
- A summarized output file listing
- The unit conversion and grid-size portions of `plasma.inp` / `plasma.toml`
- A minimal Python script and the complete traceback
- For HPC / remote execution, a summary of `emout server status`

## Before Filing Issues

Do not paste personal paths, hostnames, job IDs, access tokens, or unpublished dataset names as-is. Keep only what is needed for reproduction, and create a minimal sample when output data itself is required.
