import os
import sys
import warnings

sys.path.insert(0, os.path.abspath("../../"))

try:
    from sphinx.deprecation import RemovedInSphinx10Warning
except ImportError:
    RemovedInSphinx10Warning = None

if RemovedInSphinx10Warning is not None:
    warnings.filterwarnings(
        "ignore",
        category=RemovedInSphinx10Warning,
        module=r"sphinx_autodoc_typehints\._parser",
    )

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "emout"
copyright = "2025, Jin Nakazono"
author = "Jin Nakazono"
release = "2.20.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = []

# -- Intersphinx (cross-reference external projects) -------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# ``sphinx-build -n`` is useful for real broken links, but autodoc also
# turns many NumPy-style type fragments (``optional``, ``array-like``,
# ``default True``) into cross-reference targets. Keep the ignore list exact
# instead of suppressing all ``ref.*`` warnings so new broken references still
# surface.
nitpick_ignore = [
    ("py:attr", "name"),
    ("py:class", '"Bounds3D"'),
    ("py:class", '"Emout"'),
    ("py:class", "'center'}"),
    ("py:class", "'cont'}"),
    ("py:class", "'contour'}"),
    ("py:class", "'dirichlet'"),
    ("py:class", "'neumann'}"),
    ("py:class", "'pyvista'}"),
    ("py:class", "'quiver'}"),
    ("py:class", "'slice'"),
    ("py:class", "'streamline'"),
    ("py:class", "'vec'"),
    ("py:class", "'volume'"),
    ("py:class", "ANIMATER_PLOT_MODE"),
    ("py:class", "Axes"),
    ("py:class", "Axes3D"),
    ("py:class", "AxesImage"),
    ("py:class", "BacktraceResult"),
    ("py:class", "BoundaryCollection"),
    ("py:class", "Btype"),
    ("py:class", "Data4d"),
    ("py:class", "Emout"),
    ("py:class", "Figure"),
    ("py:class", "Line2D"),
    ("py:class", "ParticlesSeries"),
    ("py:class", "Path"),
    ("py:class", "PathLike"),
    ("py:class", "TomlData"),
    ("py:class", "_type_"),
    ("py:class", "a_deg"),
    ("py:class", "array-like"),
    ("py:class", "b_deg"),
    ("py:class", "bin_edges"),
    ("py:class", "callable"),
    ("py:class", "default 'mpl'"),
    ("py:class", "default 0"),
    ("py:class", "default 0.5"),
    ("py:class", "default 0.8"),
    ("py:class", "default 1"),
    ("py:class", "default 1.0"),
    ("py:class", "default 50"),
    ("py:class", "default 200.0"),
    ("py:class", "default False"),
    ("py:class", "default True"),
    ("py:class", "default='both'"),
    ("py:class", "default='plasma'"),
    ("py:class", "default='viridis'"),
    ("py:class", "default=False"),
    ("py:class", "default=None"),
    ("py:class", "direction"),
    ("py:class", "emout.core.boundaries.Boundary"),
    ("py:class", "emout.core.boundaries.BoundaryCollection"),
    ("py:class", "emout.core.data.data.GridDataSeries"),
    ("py:class", "emout.utils.UnitTranslator"),
    ("py:class", "generator"),
    ("py:class", "h5py.Datasets"),
    ("py:class", "h5py.File"),
    ("py:class", "hist"),
    ("py:class", "matplotlib.Colormap"),
    ("py:class", "mpl_toolkits.mplot3d.Axes3D"),
    ("py:class", "np.array"),
    ("py:class", "np.ndarray"),
    ("py:class", "optional"),
    ("py:class", "pandas.core.frame.DataFrame"),
    ("py:class", "path-like"),
    ("py:class", "pd.DataFrame"),
    ("py:class", "plt.Axes"),
    ("py:class", "plt.Figure"),
    ("py:class", "pyvista.Plotter"),
    ("py:class", "scalar"),
    ("py:class", "sequence"),
    ("py:class", "{'auto'"),
    ("py:class", "{'auto'}"),
    ("py:class", "{'box'"),
    ("py:class", "{'minmax'"),
    ("py:class", "{'mpl'"),
    ("py:class", "{'periodic'"),
    ("py:class", "{'stream'"),
    ("py:data", "typing.Union"),
    ("py:func", "emout.plot.contour3d.contour3d"),
    ("py:func", "matplotlib.axes.Axes.pcolormesh"),
    ("py:func", "matplotlib.axes.Axes.plot"),
    ("py:func", "pcolormesh"),
    ("py:mod", "_plot_2d"),
    ("py:mod", "_plot_3d"),
    ("py:mod", "emout.emout.boundaries"),
    ("py:mod", "f90nml"),
]

# -- Copybutton settings ----------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2563eb",
        "color-brand-content": "#2563eb",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",
        "color-brand-content": "#60a5fa",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view", "edit"],
}

html_title = "emout"

html_js_files = ["lang-switcher.js"]
html_css_files = ["lang-switcher.css"]
