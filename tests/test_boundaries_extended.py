"""Extended tests for :mod:`emout.core.boundaries` — collection composition,
helpers, and uncovered edge-case paths.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from emout.core.boundaries import (
    BoundaryCollection,
    CuboidBoundary,
    SphereBoundary,
)
from emout.core.boundaries._helpers import (
    _accepted_kwargs,
    _domain_extent,
    _get_scalar,
    _get_vector,
    _safe_attr,
)
from emout.core.boundaries._collection import _offset_mesh
from emout.plot.surface_cut import (
    CompositeMeshSurface,
    MeshSurface3D,
    SphereMeshSurface,
)
from emout.utils import InpFile, Units


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_inp(tmp_path: Path, ptcond_body: str) -> InpFile:
    path = tmp_path / "plasma.inp"
    path.write_text("!!key dx=[0.1],to_c=[10000.0]\n" + ptcond_body)
    return InpFile(path)


@pytest.fixture
def unit() -> Units:
    return Units(dx=0.1, to_c=10000.0)


@pytest.fixture
def two_sphere_inp(tmp_path: Path) -> InpFile:
    return _make_inp(
        tmp_path,
        """\
&ptcond
    boundary_type = 'complex'
    boundary_types(1) = 'sphere'
    boundary_types(2) = 'sphere'
    sphere_origin(:, 1) = 1.0, 2.0, 3.0
    sphere_radius(1) = 1.0
    sphere_origin(:, 2) = 10.0, 20.0, 30.0
    sphere_radius(2) = 2.0
/
""",
    )


@pytest.fixture
def coll(two_sphere_inp: InpFile, unit: Units) -> BoundaryCollection:
    return BoundaryCollection(two_sphere_inp, unit)


# ===================================================================
# BoundaryCollection.__add__ / __radd__
# ===================================================================


class TestCollectionAdd:
    """Cover BoundaryCollection.__add__ and __radd__ (lines 245-261)."""

    def test_add_two_collections(self, coll: BoundaryCollection):
        combined = coll + coll
        assert isinstance(combined, BoundaryCollection)
        assert len(combined) == 4

    def test_add_collection_and_boundary(self, coll: BoundaryCollection):
        single = coll[0]
        combined = coll + single
        assert isinstance(combined, BoundaryCollection)
        assert len(combined) == 3
        assert combined[2] is single

    def test_add_collection_and_mesh(self, coll: BoundaryCollection):
        extra = SphereMeshSurface(center=(0, 0, 0), radius=0.5, ntheta=8, nphi=4)
        result = coll + extra
        assert isinstance(result, CompositeMeshSurface)

    def test_add_collection_returns_not_implemented(self, coll: BoundaryCollection):
        result = coll.__add__("not a boundary")
        assert result is NotImplemented

    def test_radd_zero_returns_self(self, coll: BoundaryCollection):
        result = coll.__radd__(0)
        assert result is coll

    def test_radd_mesh(self, coll: BoundaryCollection):
        extra = SphereMeshSurface(center=(0, 0, 0), radius=0.5, ntheta=8, nphi=4)
        result = extra + coll
        assert isinstance(result, CompositeMeshSurface)

    def test_radd_returns_not_implemented(self, coll: BoundaryCollection):
        result = coll.__radd__("something")
        assert result is NotImplemented

    def test_add_unit_propagation_when_self_none(self, coll: BoundaryCollection, unit: Units):
        # Create a collection with unit=None
        no_unit_coll = BoundaryCollection.from_boundaries(list(coll), unit=None)
        no_unit_coll.unit = None  # force None
        combined = no_unit_coll + coll
        assert combined.unit is coll.unit


# ===================================================================
# BoundaryCollection.from_boundaries
# ===================================================================


class TestFromBoundaries:
    """Cover from_boundaries classmethod (lines 158-175)."""

    def test_from_empty_list(self):
        coll = BoundaryCollection.from_boundaries([])
        assert len(coll) == 0
        assert coll.inp is None
        assert coll.unit is None

    def test_from_boundary_list(self, coll: BoundaryCollection, unit: Units):
        explicit = BoundaryCollection.from_boundaries(
            [coll[0], coll[1]], unit=unit,
        )
        assert len(explicit) == 2
        assert explicit.unit is unit
        assert explicit.inp is None

    def test_from_boundaries_flattens_collections(self, coll: BoundaryCollection):
        # Pass a BoundaryCollection as one of the items — it should flatten.
        explicit = BoundaryCollection.from_boundaries([coll])
        assert len(explicit) == len(coll)
        for i in range(len(coll)):
            assert explicit[i] is coll[i]

    def test_from_boundaries_infers_unit(self, coll: BoundaryCollection):
        # unit=None should pick up unit from first boundary that has one.
        explicit = BoundaryCollection.from_boundaries(list(coll), unit=None)
        assert explicit.unit is coll[0].unit


# ===================================================================
# BoundaryCollection._build with None inp / btype
# ===================================================================


class TestBuildEdgeCases:
    """Cover _build with None inp (line 181) and None btype (line 185)."""

    def test_build_with_none_inp(self):
        coll = BoundaryCollection.__new__(BoundaryCollection)
        coll.inp = None
        coll.unit = None
        coll._emout_open_kwargs = None
        coll.skipped = []
        coll._boundaries = coll._build()
        assert len(coll) == 0

    def test_build_with_none_btype(self, tmp_path: Path, unit: Units):
        # An inp file with no boundary_type key at all.
        path = tmp_path / "plasma.inp"
        path.write_text("&ptcond\n/\n")
        inp = InpFile(path)
        coll = BoundaryCollection(inp, unit)
        assert len(coll) == 0


# ===================================================================
# _offset_mesh (lines 101-113)
# ===================================================================


class TestOffsetMesh:
    def test_offset_shifts_vertices(self):
        mesh = SphereMeshSurface(center=(5.0, 5.0, 5.0), radius=1.0, ntheta=8, nphi=4)
        V_orig, _ = mesh.mesh()
        center_before = V_orig.mean(axis=0).copy()

        shifted = _offset_mesh(mesh, (10.0, -5.0, 0.0))
        V_shifted, F_shifted = shifted.mesh()

        assert V_shifted.shape == V_orig.shape
        # x shifted by +10
        np.testing.assert_allclose(
            V_shifted[:, 0], V_orig[:, 0] + 10.0, atol=1e-10,
        )
        # y shifted by -5
        np.testing.assert_allclose(
            V_shifted[:, 1], V_orig[:, 1] - 5.0, atol=1e-10,
        )
        # z unchanged (offset=0.0)
        np.testing.assert_allclose(
            V_shifted[:, 2], V_orig[:, 2], atol=1e-10,
        )

    def test_offset_with_none_elements(self):
        mesh = SphereMeshSurface(center=(0, 0, 0), radius=1.0, ntheta=8, nphi=4)
        V_orig, _ = mesh.mesh()

        shifted = _offset_mesh(mesh, (None, 3.0, None))
        V_shifted, _ = shifted.mesh()

        # x unchanged
        np.testing.assert_allclose(V_shifted[:, 0], V_orig[:, 0], atol=1e-10)
        # y shifted by +3
        np.testing.assert_allclose(V_shifted[:, 1], V_orig[:, 1] + 3.0, atol=1e-10)
        # z unchanged
        np.testing.assert_allclose(V_shifted[:, 2], V_orig[:, 2], atol=1e-10)


# ===================================================================
# _accepted_kwargs (lines 52-74)
# ===================================================================


class TestAcceptedKwargs:
    def test_none_input(self):
        assert _accepted_kwargs(None) is None

    def test_class_with_var_keyword(self):
        class Foo:
            def __init__(self, a, b=1, **kwargs):
                pass

        result = _accepted_kwargs(Foo)
        assert result is None

    def test_class_with_explicit_params(self):
        class Bar:
            def __init__(self, x, y, z=10):
                pass

        result = _accepted_kwargs(Bar)
        assert result == {"x", "y", "z"}

    def test_class_with_type_error_on_signature(self):
        # Built-in types can raise TypeError/ValueError on inspect.signature.
        # Use object() as a proxy — its __init__ is a built-in wrapper.
        result = _accepted_kwargs(int)
        # int.__init__ should still work but the result depends on the
        # implementation. Just ensure it doesn't crash.
        assert result is None or isinstance(result, set)


# ===================================================================
# _get_scalar (lines 82-100)
# ===================================================================


class TestGetScalar:
    def _make_nml(self, data: dict, start_index: dict | None = None):
        """Build a dict-like nml_group with an optional start_index attr."""

        class NmlGroup(dict):
            pass

        nml = NmlGroup(data)
        nml.start_index = start_index if start_index is not None else {}
        return nml

    def test_missing_name(self):
        nml = self._make_nml({})
        assert _get_scalar(nml, "missing_key", 1) is None

    def test_out_of_bounds_index(self):
        nml = self._make_nml({"arr": [10.0, 20.0]})
        # Asking for Fortran index 5 but array only has 2 elements.
        assert _get_scalar(nml, "arr", 5) is None

    def test_none_element(self):
        nml = self._make_nml({"arr": [None, 42.0]})
        assert _get_scalar(nml, "arr", 1) is None
        assert _get_scalar(nml, "arr", 2) == 42.0

    def test_valid_lookup(self):
        nml = self._make_nml({"vals": [100.0, 200.0, 300.0]})
        assert _get_scalar(nml, "vals", 1) == 100.0
        assert _get_scalar(nml, "vals", 3) == 300.0

    def test_negative_index(self):
        nml = self._make_nml(
            {"vals": [10.0]},
            start_index={"vals": [3]},
        )
        # Fortran index 1 < start 3 => idx < 0 => None
        assert _get_scalar(nml, "vals", 1) is None
        # Fortran index 3 = start => idx 0 => 10.0
        assert _get_scalar(nml, "vals", 3) == 10.0


# ===================================================================
# _get_vector (lines 103-137)
# ===================================================================


class TestGetVector:
    def _make_nml(self, data: dict, start_index: dict | None = None):
        class NmlGroup(dict):
            pass

        nml = NmlGroup(data)
        nml.start_index = start_index if start_index is not None else {}
        return nml

    def test_missing_name(self):
        nml = self._make_nml({})
        assert _get_vector(nml, "missing_key", 1) is None

    def test_out_of_bounds(self):
        nml = self._make_nml({"vec": [[1.0, 2.0]]})
        assert _get_vector(nml, "vec", 99) is None

    def test_none_entry(self):
        nml = self._make_nml({"vec": [None, [3.0, 4.0]]})
        assert _get_vector(nml, "vec", 1) is None
        assert _get_vector(nml, "vec", 2) == [3.0, 4.0]

    def test_all_none_values(self):
        nml = self._make_nml({"vec": [[None, None, None]]})
        assert _get_vector(nml, "vec", 1) is None

    def test_valid_vector(self):
        nml = self._make_nml({"origin": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]})
        assert _get_vector(nml, "origin", 1) == [1.0, 2.0, 3.0]
        assert _get_vector(nml, "origin", 2) == [4.0, 5.0, 6.0]

    def test_sparse_array_start_index_2d(self):
        # 2-D arrays: start_index = [None, start_for_dim2]
        nml = self._make_nml(
            {"origin": [[10.0, 20.0, 30.0]]},
            start_index={"origin": [None, 3]},
        )
        # Fortran index 3 maps to Python index 0
        assert _get_vector(nml, "origin", 3) == [10.0, 20.0, 30.0]
        # Fortran index 2 < start 3 => None
        assert _get_vector(nml, "origin", 2) is None

    def test_sparse_array_start_index_1d(self):
        # 1-D start_index: [start_for_dim1]
        nml = self._make_nml(
            {"vals": [[7.0, 8.0]]},
            start_index={"vals": [2]},
        )
        assert _get_vector(nml, "vals", 2) == [7.0, 8.0]
        assert _get_vector(nml, "vals", 1) is None

    def test_non_list_entry_returns_none(self):
        # If a single element is stored as a scalar instead of a list.
        nml = self._make_nml({"val": [42.0]})
        assert _get_vector(nml, "val", 1) is None


# ===================================================================
# _safe_attr (lines 140-150)
# ===================================================================


class TestSafeAttr:
    def test_existing_attribute(self):
        obj = SimpleNamespace(foo=42)
        assert _safe_attr(obj, "foo") == 42

    def test_missing_attribute(self):
        obj = SimpleNamespace()
        assert _safe_attr(obj, "missing") is None

    def test_key_error_is_caught(self):
        # InpFile raises KeyError for missing keys via __getattr__.
        class FakeInp:
            def __getattr__(self, key):
                raise KeyError(key)

        assert _safe_attr(FakeInp(), "anything") is None

    def test_attribute_error_is_caught(self):
        class Strict:
            def __getattr__(self, key):
                raise AttributeError(key)

        assert _safe_attr(Strict(), "missing") is None


# ===================================================================
# _domain_extent
# ===================================================================


class TestDomainExtent:
    def test_returns_floats(self):
        obj = SimpleNamespace(nx=64, ny=48, nz=32)
        result = _domain_extent(obj)
        assert result == (64.0, 48.0, 32.0)

    def test_fallback_on_missing_keys(self):
        class NoAttrs:
            def __getattr__(self, key):
                raise AttributeError(key)

        result = _domain_extent(NoAttrs())
        assert result == (1.0, 1.0, 1.0)
