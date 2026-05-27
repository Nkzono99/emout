"""Article data recording and replay support.

This module keeps figure scripts reproducible by recording the minimum grid
data consumed by :meth:`Data.plot` and :meth:`Data.to_numpy`, then replaying
those same slices without requiring the original EMSES output files.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from emout.core.data.selectors import (
    compose_selector as _compose_selector,
    normalize_item as _normalize_item_base,
    selector_length as _selector_length,
)
from emout.core.io.diagnostics import read_icur, read_pbody
from emout.utils import InpFile, UnitTranslator, Units

ENV_MODE = "EMOUT_ARTICLE_MODE"
ENV_RECORDS_PATH = "EMOUT_ARTICLE_RECORDS_PATH"
ENV_RECORDS_PATH_ALIAS = "EMOUT_RECORDS_PATH"
ENV_NAME = "EMOUT_ARTICLE_NAME"

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ArticleConfig:
    """Resolved article mode configuration."""

    mode: str
    records_path: Path | None
    article_name: str


def resolve_config(
    *,
    article_mode: str | None = None,
    article_records_path: str | Path | None = None,
    records_path: str | Path | None = None,
    article_name: str | None = None,
) -> ArticleConfig:
    """Resolve article settings from explicit arguments and environment."""
    mode = article_mode if article_mode is not None else os.getenv(ENV_MODE, "normal")
    mode = mode.lower()
    if mode not in {"normal", "record", "replay"}:
        raise ValueError(f"Unsupported article mode: {mode!r}. Expected 'normal', 'record', or 'replay'.")

    path_value = (
        article_records_path or records_path or os.getenv(ENV_RECORDS_PATH) or os.getenv(ENV_RECORDS_PATH_ALIAS)
    )
    resolved_records_path = Path(path_value).expanduser() if path_value is not None else None
    if mode != "normal" and resolved_records_path is None:
        raise ValueError(
            f"Article record/replay mode requires records_path, article_records_path, or {ENV_RECORDS_PATH}."
        )

    resolved_name = article_name or os.getenv(ENV_NAME) or "default"
    return ArticleConfig(mode=mode, records_path=resolved_records_path, article_name=resolved_name)


def source_id(source_path: str | Path) -> str:
    """Return a stable record directory name for a source path."""
    path = Path(source_path).expanduser()
    resolved = path.resolve(strict=False)
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:10]
    stem = resolved.name or "dataset"
    safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
    return f"{safe_stem}-{digest}"


def resolve_record_dir(source_path: str | Path, config: ArticleConfig) -> Path:
    """Return the directory for one source dataset and article name."""
    if config.records_path is None:
        raise ValueError("records_path is required outside normal article mode.")
    return config.records_path / "datasets" / source_id(source_path) / config.article_name


def resolve_existing_record_dir(source_path: str | Path, config: ArticleConfig) -> Path:
    """Resolve a replay directory, falling back to a single matching article."""
    record_dir = resolve_record_dir(source_path, config)
    if record_dir.exists():
        return record_dir

    if config.records_path is None:
        raise ValueError("records_path is required for article replay.")

    matches = sorted((config.records_path / "datasets").glob(f"*/{config.article_name}"))
    if len(matches) == 1:
        warnings.warn(
            f"Article record for {source_path!s} was not found at {record_dir}; "
            f"falling back to the single matching article record {matches[0]}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Article record not found: {record_dir}")
    raise FileNotFoundError(
        f"Article record not found for {source_path!s}; multiple '{config.article_name}' records exist."
    )


def _load_manifest(record_dir: Path) -> dict[str, Any]:
    manifest_path = record_dir / "manifest.json"
    data_path = record_dir / "data.h5"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Article manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = manifest.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported article manifest schema_version {schema_version!r}; expected {SCHEMA_VERSION}.")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Invalid article manifest records in {manifest_path}")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Invalid article manifest record at index {index} in {manifest_path}")
        for key in ("dataset", "field", "selector", "slices", "slice_axes"):
            if key not in record:
                raise ValueError(f"Invalid article manifest record {index}: missing {key!r}")
    if records and not data_path.exists():
        raise FileNotFoundError(f"Article data file not found: {data_path}")
    if records:
        with h5py.File(data_path, "r") as h5:
            for index, record in enumerate(records):
                dataset_path = record["dataset"]
                if dataset_path not in h5:
                    raise ValueError(f"Invalid article manifest record {index}: dataset {dataset_path!r} is missing")
                shape = record.get("shape")
                if shape is not None and list(h5[dataset_path].shape) != list(shape):
                    raise ValueError(f"Invalid article manifest record {index}: dataset shape does not match manifest")
    return manifest


def _verify_source_json(record_dir: Path) -> None:
    source_path = record_dir.parent / "source.json"
    if not source_path.exists():
        return

    source = json.loads(source_path.read_text(encoding="utf-8"))
    recorded_files = source.get("recorded_files") or {}
    for filename, expected_hash in recorded_files.items():
        path = record_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Recorded article file is missing: {path}")
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Recorded article file hash mismatch: {path}")


def attach_recorder(obj: Any, recorder: "ArticleRecorder | None", source_shape: tuple[int, ...] | None = None) -> Any:
    """Attach *recorder* recursively to grid data objects."""
    if recorder is None:
        return obj

    if hasattr(obj, "objs"):
        for child in obj.objs:
            attach_recorder(child, recorder, source_shape)
        return obj

    if hasattr(obj, "series"):
        setattr(obj, "_article_recorder", recorder)
        setattr(obj, "_article_source_shape", tuple(getattr(obj, "shape", source_shape or ())))
        for child in obj.series:
            attach_recorder(child, recorder, getattr(child, "shape", source_shape))
        return obj

    setattr(obj, "_article_recorder", recorder)
    if source_shape is None and hasattr(obj, "shape") and getattr(obj, "ndim", None) == 4:
        source_shape = tuple(obj.shape)
    if source_shape is not None:
        setattr(obj, "_article_source_shape", tuple(source_shape))
    return obj


def record_data_access(data: Any, kind: str, kwargs: dict[str, Any] | None = None) -> None:
    """Record a data object when an article recorder is attached."""
    recorder = getattr(data, "_article_recorder", None)
    if recorder is None:
        return
    recorder.record_data(data, kind=kind, kwargs=kwargs)


def _record_key(field: str | None, selector: list[Any]) -> str:
    payload = {"field": field, "selector": selector}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _raise_unsupported_replay_api(name: str) -> None:
    raise NotImplementedError(
        f"Article replay does not provide {name}. "
        "Only recorded grid slices, input metadata, diagnostics, and boundaries are available."
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class ArticleRecorder:
    """Persist article data records for one :class:`emout.Emout` source."""

    def __init__(self, source_path: str | Path, config: ArticleConfig):
        self.source_path = Path(source_path).expanduser().resolve(strict=False)
        self.config = config
        self.record_dir = resolve_record_dir(self.source_path, config)
        self.dataset_dir = self.record_dir.parent
        self.data_path = self.record_dir / "data.h5"
        self._staging_data_path = self.record_dir / ".data.h5.tmp"
        self.manifest_path = self.record_dir / "manifest.json"
        self.records: list[dict[str, Any]] = []
        self._dataset_by_key: dict[str, dict[str, Any]] = {}

        self.record_dir.mkdir(parents=True, exist_ok=True)
        self._staging_data_path.unlink(missing_ok=True)
        self._write_manifest()
        self._write_source_json()

    def write_input(self, inp: InpFile | None) -> None:
        """Save the input file required for unit reconstruction."""
        if inp is not None:
            inp.save(self.record_dir / "plasma.inp", convkey=inp.convkey)
            self._write_source_json()

    def copy_input_path(self, input_path: str | Path | None) -> None:
        """Copy the original input file when it exists."""
        if input_path is None:
            return
        src = Path(input_path)
        if src.exists():
            shutil.copyfile(src, self.record_dir / src.name)
            self._write_source_json()

    def copy_source_files(self, input_directory: str | Path, output_directory: str | Path) -> None:
        """Copy small source-side files useful for article replay."""
        input_directory = Path(input_directory)
        output_directory = Path(output_directory)
        for filename in ("plasma.inp", "plasma.toml"):
            src = input_directory / filename
            if src.exists():
                shutil.copyfile(src, self.record_dir / filename)

        for filename in ("icur", "pbody"):
            src = output_directory / filename
            if src.exists():
                shutil.copyfile(src, self.record_dir / filename)
        self._write_source_json()

    def record_data(self, data: Any, *, kind: str, kwargs: dict[str, Any] | None = None) -> None:
        """Append a consumed data slice to the article bundle."""
        if not hasattr(data, "slices"):
            return

        selector = _encode_selector_tuple(_to_recipe_index(data))
        key = _record_key(getattr(data, "name", None), selector)
        if key in self._dataset_by_key:
            return

        record_id = f"record_{len(self.records):06d}"
        dataset_path = f"records/{record_id}"
        array = np.asarray(data)

        record = {
            "id": record_id,
            "kind": kind,
            "field": getattr(data, "name", None),
            "dataset": dataset_path,
            "shape": list(array.shape),
            "source_shape": list(getattr(data, "_article_source_shape", ())),
            "slices": [_encode_slice(slc) for slc in data.slices],
            "selector": selector,
            "slice_axes": list(data.slice_axes),
            "axisunits": [_encode_unit(unit) for unit in getattr(data, "axisunits", [])],
            "valunit": _encode_unit(getattr(data, "valunit", None)),
            "kwargs": _jsonable(kwargs or {}),
        }
        self._write_dataset(dataset_path, array)
        self.records.append(record)
        self._dataset_by_key[key] = record
        self._write_manifest()

    def _write_dataset(self, dataset_path: str, array: np.ndarray) -> None:
        """Atomically add one dataset to ``data.h5``."""
        self._staging_data_path.unlink(missing_ok=True)
        if self.data_path.exists():
            shutil.copyfile(self.data_path, self._staging_data_path)

        try:
            with h5py.File(self._staging_data_path, "a") as h5:
                h5.create_dataset(dataset_path, data=array)
            self._staging_data_path.replace(self.data_path)
        finally:
            self._staging_data_path.unlink(missing_ok=True)

    def _write_source_json(self) -> None:
        source = {
            "schema_version": SCHEMA_VERSION,
            "source_path": str(self.source_path),
            "source_path_hash": hashlib.sha1(str(self.source_path).encode("utf-8")).hexdigest(),
            "article_name": self.config.article_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "recorded_files": self._recorded_file_hashes(),
        }
        source_path = self.dataset_dir / "source.json"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(source_path, source)

    def _recorded_file_hashes(self) -> dict[str, str]:
        files = {}
        for filename in ("plasma.inp", "plasma.toml", "icur", "pbody"):
            path = self.record_dir / filename
            if path.exists():
                files[filename] = _sha256_file(path)
        return files

    def _write_manifest(self) -> None:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "article_name": self.config.article_name,
            "source_path": str(self.source_path),
            "records": self.records,
        }
        _write_json_atomic(self.manifest_path, manifest)


class ArticleReplayEmout:
    """Replay proxy returned by ``Emout(..., article_mode='replay')``."""

    def __init__(self, source_path: str | Path, config: ArticleConfig):
        self._source_path = Path(source_path).expanduser().resolve(strict=False)
        self._config = config
        self._record_dir = resolve_existing_record_dir(self._source_path, config)
        self._data_path = self._record_dir / "data.h5"
        self._manifest = _load_manifest(self._record_dir)
        _verify_source_json(self._record_dir)
        self._records = self._manifest.get("records", [])
        self._records_by_field: dict[str, list[dict[str, Any]]] = {}
        for record in self._records:
            field = record.get("field")
            if field is not None:
                self._records_by_field.setdefault(field, []).append(record)

        inp_path = self._record_dir / "plasma.inp"
        self._inp = InpFile(inp_path) if inp_path.exists() else None
        toml_path = self._record_dir / "plasma.toml"
        if toml_path.exists():
            from emout.utils.toml_converter import load_toml

            self._toml = load_toml(toml_path, resolve_groups=True, purge_groups=True)
        else:
            self._toml = None

        if self._inp is not None and self._inp.convkey is not None:
            self._unit = Units(self._inp.convkey.dx, self._inp.convkey.to_c)
        else:
            self._unit = None

    @property
    def directory(self) -> Path:
        """Return the article replay directory."""
        return self._record_dir

    @property
    def inp(self) -> InpFile | None:
        """Return the recorded input file."""
        return self._inp

    @property
    def toml(self):
        """Return the recorded TOML configuration when available."""
        return self._toml

    @property
    def unit(self) -> Units | None:
        """Return units reconstructed from the recorded input file."""
        return self._unit

    def is_valid(self) -> bool:
        """Return whether the replay bundle has the files required by its manifest."""
        return not self._records or self._data_path.exists()

    def available_fields(self) -> list[str]:
        """Return recorded scalar field names."""
        return sorted(self._records_by_field)

    @property
    def icur(self) -> pd.DataFrame:
        """Return the recorded ``icur`` diagnostic file as a DataFrame."""
        if self._inp is None:
            raise RuntimeError("icur replay requires a recorded plasma.inp")
        return read_icur(self._record_dir / "icur", self._inp)

    @property
    def pbody(self) -> pd.DataFrame:
        """Return the recorded ``pbody`` diagnostic file as a DataFrame."""
        if self._inp is None:
            raise RuntimeError("pbody replay requires a recorded plasma.inp")
        return read_pbody(self._record_dir / "pbody", self._inp)

    @property
    def boundaries(self):
        """Return boundary meshes reconstructed from the recorded input file."""
        from emout.core.boundaries import BoundaryCollection

        return BoundaryCollection(self._inp, self._unit)

    @property
    def backtrace(self):
        """Backtrace is not part of article replay bundles."""
        _raise_unsupported_replay_api("backtrace")

    def remote(self, *args, **kwargs):
        """Remote execution is not available for article replay bundles."""
        _raise_unsupported_replay_api("remote")

    def particle(self, *args, **kwargs):
        """Particle files are not part of article replay bundles."""
        _raise_unsupported_replay_api("particle")

    def __getattr__(self, name: str):
        """Resolve recorded scalar and vector field names."""
        if name in self._records_by_field:
            return ArticleSeries(self, name)

        from emout.core.data.vector_resolver import resolve_vector_name

        try:
            vector = resolve_vector_name(name, lambda component: getattr(self, component), fallback_3d=False)
        except AttributeError:
            vector = None
        if vector is not None:
            return vector

        raise AttributeError(f"Article replay field is not recorded: {name}")

    def _load_record_data(self, record: dict[str, Any]):
        with h5py.File(self._data_path, "r") as h5:
            array = np.array(h5[record["dataset"]])

        from emout.core.data.factory import data_from_array

        return data_from_array(
            array,
            filename=self._data_path,
            name=record.get("field"),
            tslice=_decode_slice(record["slices"][0]),
            zslice=_decode_slice(record["slices"][1]),
            yslice=_decode_slice(record["slices"][2]),
            xslice=_decode_slice(record["slices"][3]),
            slice_axes=list(record["slice_axes"]),
            axisunits=[_decode_unit(unit) for unit in record.get("axisunits", [])],
            valunit=_decode_unit(record.get("valunit")),
        )

    def _find_record(self, field: str, selectors: tuple[Any, ...]) -> dict[str, Any]:
        encoded_selector = _encode_selector_tuple(selectors)
        for record in self._records_by_field.get(field, []):
            if record.get("selector") == encoded_selector:
                return record
        raise KeyError(f"Article replay data is not recorded: {field}{selectors!r}")


class ArticleSeries:
    """Lazy selector for one recorded field."""

    def __init__(self, replay: ArticleReplayEmout, field: str):
        self._replay = replay
        self.name = field
        first = replay._records_by_field[field][0]
        source_shape = first.get("source_shape") or _source_shape_from_record(first)
        self.shape = tuple(source_shape)
        self.ndim = 4

    def __getitem__(self, item):
        selectors = _normalize_item(item, self.shape)
        ndim = sum(not isinstance(selector, int) for selector in selectors)
        selection = ArticleSelection(self._replay, self.name, selectors, self.shape)
        if ndim <= 3:
            return selection.materialize()
        return selection


class ArticleSelection:
    """Replay selection for a recorded field."""

    def __init__(
        self,
        replay: ArticleReplayEmout,
        field: str,
        selectors: tuple[Any, ...],
        source_shape: tuple[int, ...],
    ):
        self._replay = replay
        self.name = field
        self._selectors = selectors
        self._source_shape = source_shape
        self.slice_axes = [axis for axis, selector in enumerate(selectors) if not isinstance(selector, int)]
        self.shape = tuple(
            _selector_length(selector, size)
            for selector, size in zip(selectors, source_shape)
            if not isinstance(selector, int)
        )
        self.ndim = len(self.shape)

    def __getitem__(self, item):
        selectors = _compose_item(self._selectors, self.slice_axes, item, self._source_shape)
        result = type(self)(self._replay, self.name, selectors, self._source_shape)
        if result.ndim <= 3:
            return result.materialize()
        return result

    def materialize(self):
        """Load the recorded data object for this selection."""
        record = self._replay._find_record(self.name, self._selectors)
        return self._replay._load_record_data(record)

    def to_numpy(self) -> np.ndarray:
        """Return the recorded NumPy array."""
        data = self.materialize()
        if hasattr(data, "to_numpy"):
            return data.to_numpy()
        return np.asarray(data)

    def plot(self, **kwargs):
        """Plot the recorded data using the normal Data plotting path."""
        return self.materialize().plot(**kwargs)


def _to_recipe_index(data: Any) -> tuple[Any, ...]:
    if hasattr(data, "_to_recipe_index"):
        return tuple(data._to_recipe_index())

    result = []
    for slc in data.slices:
        if slc.stop - slc.start == slc.step:
            result.append(slc.start)
        else:
            result.append(slc)
    return tuple(result)


def _normalize_item(item: Any, shape: tuple[int, ...]) -> tuple[Any, ...]:
    return _normalize_item_base(item, shape, target="article data")


def _compose_item(
    selectors: tuple[Any, ...],
    slice_axes: list[int],
    item: Any,
    shape: tuple[int, ...],
) -> tuple[Any, ...]:
    if not isinstance(item, tuple):
        item = (item,)
    item = item + (slice(None),) * (len(slice_axes) - len(item))
    if len(item) > len(slice_axes):
        raise IndexError(f"too many indices for {len(slice_axes)}-dimensional article selection")

    result = list(selectors)
    for dim, sub_selector in enumerate(item):
        axis = slice_axes[dim]
        result[axis] = _compose_selector(result[axis], sub_selector, shape[axis])
    return tuple(result)


def _source_shape_from_record(record: dict[str, Any]) -> list[int]:
    slices = [_decode_slice(encoded) for encoded in record["slices"]]
    shape = []
    for axis, slc in enumerate(slices):
        if axis in record["slice_axes"]:
            shape.append(max(slc.stop, 1))
        else:
            shape.append(max(slc.start + 1, 1))
    return shape


def _encode_slice(slc: slice) -> dict[str, Any]:
    return {"type": "slice", "start": slc.start, "stop": slc.stop, "step": slc.step}


def _decode_slice(value: dict[str, Any]) -> slice:
    return slice(value["start"], value["stop"], value["step"])


def _encode_selector_tuple(selectors: tuple[Any, ...]) -> list[Any]:
    return [_encode_selector(selector) for selector in selectors]


def _encode_selector(selector: Any) -> Any:
    if isinstance(selector, slice):
        return _encode_slice(selector)
    if isinstance(selector, tuple):
        return {"type": "tuple", "items": list(selector)}
    if isinstance(selector, list):
        return {"type": "tuple", "items": selector}
    if isinstance(selector, np.integer):
        return int(selector)
    return selector


def _encode_unit(unit: UnitTranslator | None) -> dict[str, Any] | None:
    if unit is None:
        return None
    return {
        "from_unit": unit.from_unit,
        "to_unit": unit.to_unit,
        "name": unit.name,
        "unit": unit.unit,
    }


def _decode_unit(value: dict[str, Any] | None) -> UnitTranslator | None:
    if value is None:
        return None
    return UnitTranslator(value["from_unit"], value["to_unit"], name=value.get("name"), unit=value.get("unit"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    if isinstance(value, slice):
        return _encode_slice(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)
