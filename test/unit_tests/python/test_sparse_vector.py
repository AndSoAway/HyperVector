from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Module loader helpers (same pattern as test_hypervec_server_engine.py)
# ---------------------------------------------------------------------------

def _load_module(relative_path: str, name: str):
    path = Path(__file__).parents[3] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_schema_module():
    return _load_module("pyhypervec/pyhypervec/schema.py", "schema_under_test")


def load_scalar_store_module():
    return _load_module("src/python/hypervec_scalar_store.py", "scalar_store_under_test")


def load_engine_module():
    return _load_module("src/python/hypervec_server_engine.py", "engine_under_test")


# ---------------------------------------------------------------------------
# FakeHypervec (no C++ dependency — copied from test_hypervec_server_engine)
# ---------------------------------------------------------------------------

class FakeIndexFlatL2:
    def __init__(self, d: int, *, trained: bool = True) -> None:
        self.d = d
        self.n_total = 0
        self.is_trained = trained
        self.vectors = np.empty((0, d), dtype=np.float32)

    def train(self, x) -> None:
        self.is_trained = True

    def add(self, x) -> None:
        self.vectors = np.vstack([self.vectors, np.asarray(x, dtype=np.float32)])
        self.n_total = len(self.vectors)

    def search(self, x, k: int):
        x = np.asarray(x, dtype=np.float32)
        distances = ((x[:, None, :] - self.vectors[None, :, :]) ** 2).sum(axis=2)
        labels = np.argsort(distances, axis=1)[:, :k].astype(np.int64)
        dists = np.take_along_axis(distances, labels, axis=1).astype(np.float32)
        return dists, labels


class FakeHypervec:
    kMetricL2 = 1
    kMetricInnerProduct = 0

    def __init__(self) -> None:
        self.saved_index = None

    def IndexFlatL2(self, d: int):
        return FakeIndexFlatL2(d)

    def IndexFlatIP(self, d: int):
        return FakeIndexFlatL2(d)

    def IndexIVFFlat(self, d, nlist, metric):
        return FakeIndexFlatL2(d, trained=False)

    def IndexIVFLVQ(self, d, nlist, nlocal, nbits, metric):
        return FakeIndexFlatL2(d, trained=False)

    def IndexIVFPQ(self, d, nlist, m_pq, nbits, metric):
        return FakeIndexFlatL2(d, trained=False)

    def IndexHNSWFlat(self, d, m_hnsw, metric):
        return FakeIndexFlatL2(d)

    def IndexHNSWLVQ(self, d, nlocal, nbits, m_hnsw, metric):
        return FakeIndexFlatL2(d, trained=False)

    def IndexHNSWPQ(self, d, m_pq, nbits, m_hnsw, metric):
        return FakeIndexFlatL2(d, trained=False)

    def write_index(self, index, path: str) -> None:
        self.saved_index = index
        Path(path).write_text("fake", encoding="utf-8")

    def read_index(self, path: str):
        return self.saved_index


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SPARSE_SCHEMA = {
    "auto_id": False,
    "enable_dynamic_field": True,
    "fields": [
        {"name": "id", "datatype": "VARCHAR", "is_primary": True},
        {"name": "sparse_vec", "datatype": "SPARSE_FLOAT_VECTOR"},
        {"name": "contents", "datatype": "VARCHAR"},
    ],
}

DENSE_SCHEMA = {
    "auto_id": False,
    "enable_dynamic_field": True,
    "fields": [
        {"name": "id", "datatype": "VARCHAR", "is_primary": True},
        {"name": "vector", "datatype": "FLOAT_VECTOR", "dim": 4},
        {"name": "contents", "datatype": "VARCHAR"},
    ],
}


# ---------------------------------------------------------------------------
# 1. DataType constant
# ---------------------------------------------------------------------------

def test_datatype_constant_exists():
    # Load only the DataType portion by reading and evaling just that class
    path = Path(__file__).parents[3] / "pyhypervec/pyhypervec/schema.py"
    source = path.read_text(encoding="utf-8")
    # Extract and exec just the DataType class definition (lines before @dataclass)
    lines = source.splitlines()
    class_lines = []
    in_class = False
    for line in lines:
        if line.startswith("class DataType:"):
            in_class = True
        if in_class:
            if line.strip() == "" and class_lines:
                break
            if line.startswith("@") and class_lines:
                break
            if line.startswith("class ") and class_lines and not line.startswith("class DataType:"):
                break
            class_lines.append(line)
    ns = {}
    exec("\n".join(class_lines), ns)
    DataType = ns["DataType"]
    assert DataType.SPARSE_FLOAT_VECTOR == "SPARSE_FLOAT_VECTOR"
    assert DataType.FLOAT_VECTOR == "FLOAT_VECTOR"


# ---------------------------------------------------------------------------
# 2. encode / decode sparse round-trip
# ---------------------------------------------------------------------------

def test_encode_decode_sparse_roundtrip():
    mod = load_scalar_store_module()
    store = mod.ScalarStore

    sparse = {0: 0.5, 100: 1.2, 50000: 0.01}
    encoded = store._encode_sparse(sparse)
    decoded = store._decode_sparse(encoded)
    assert set(decoded.keys()) == set(sparse.keys())
    for k in sparse:
        assert abs(decoded[k] - sparse[k]) < 1e-6, f"mismatch at key {k}"


def test_encode_decode_sparse_empty():
    mod = load_scalar_store_module()
    store = mod.ScalarStore
    assert store._decode_sparse(store._encode_sparse({})) == {}


def test_encode_sparse_binary_layout():
    mod = load_scalar_store_module()
    store = mod.ScalarStore
    sparse = {3: 0.25}
    data = store._encode_sparse(sparse)
    assert len(data) == 4 + 8  # nnz=1 header + 1 pair
    nnz = struct.unpack_from("<I", data, 0)[0]
    assert nnz == 1
    idx, val = struct.unpack_from("<If", data, 4)
    assert idx == 3
    assert abs(val - 0.25) < 1e-7


# ---------------------------------------------------------------------------
# 3. _encode_vector dispatches to sparse / dense
# ---------------------------------------------------------------------------

def test_encode_vector_dense():
    mod = load_scalar_store_module()
    store = mod.ScalarStore
    arr = [1.0, 2.0, 3.0]
    encoded = store._encode_vector(arr)
    result = np.frombuffer(encoded, dtype=np.float32)
    np.testing.assert_allclose(result, arr)


def test_encode_vector_sparse():
    mod = load_scalar_store_module()
    store = mod.ScalarStore
    sparse = {10: 0.9}
    encoded = store._encode_vector(sparse)
    decoded = store._decode_sparse(encoded)
    assert decoded == {10: pytest.approx(0.9, abs=1e-6)}


# ---------------------------------------------------------------------------
# 4. _export_vector detects sparse vs dense
# ---------------------------------------------------------------------------

def test_export_vector_detects_sparse():
    mod = load_scalar_store_module()
    store = mod.ScalarStore
    sparse = {7: 0.3, 42: 1.5}
    blob = store._encode_sparse(sparse)
    result = store._export_vector(blob)
    assert isinstance(result, dict)
    assert "indices" in result and "values" in result
    assert set(result["indices"]) == {7, 42}


def test_export_vector_detects_dense():
    mod = load_scalar_store_module()
    store = mod.ScalarStore
    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    blob = arr.tobytes()
    result = store._export_vector(blob)
    assert isinstance(result, list)
    assert len(result) == 4


# ---------------------------------------------------------------------------
# 5. ScalarStore: insert and retrieve sparse rows
# ---------------------------------------------------------------------------

def test_scalar_store_insert_sparse(tmp_path):
    mod = load_scalar_store_module()
    store = mod.ScalarStore(tmp_path / "scalar.db")
    store.ensure_table("col")
    sparse_vec = {0: 0.1, 100: 0.9}
    store.insert_batch("col", [(0, "doc0", sparse_vec, "hello", {})])
    rows = store.export_rows("col")
    assert len(rows) == 1
    v = rows[0]["vector"]
    assert isinstance(v, dict)
    assert set(v["indices"]) == {0, 100}
    values_by_idx = dict(zip(v["indices"], v["values"]))
    assert abs(values_by_idx[0] - 0.1) < 1e-6
    assert abs(values_by_idx[100] - 0.9) < 1e-6


def test_scalar_store_get_sparse_vectors(tmp_path):
    mod = load_scalar_store_module()
    store = mod.ScalarStore(tmp_path / "scalar.db")
    store.ensure_table("col")
    s0 = {1: 0.5, 200: 0.3}
    s1 = {50: 1.0}
    store.insert_batch("col", [(0, "a", s0, "", {}), (1, "b", s1, "", {})])
    result = store.get_sparse_vectors("col")
    assert len(result) == 2
    assert result[0][1] == pytest.approx(0.5, abs=1e-6)
    assert result[1][50] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 6. import_rows round-trip for sparse
# ---------------------------------------------------------------------------

def test_import_rows_sparse_roundtrip(tmp_path):
    mod = load_scalar_store_module()
    store = mod.ScalarStore(tmp_path / "scalar.db")
    store.ensure_table("col")
    sparse_vec = {5: 0.7, 999: 0.2}
    store.insert_batch("col", [(0, "x", sparse_vec, "text", {"tag": "a"})])
    exported = store.export_rows("col")

    store2 = mod.ScalarStore(tmp_path / "scalar2.db")
    store2.import_rows("col", exported)
    re_exported = store2.export_rows("col")

    assert len(re_exported) == 1
    v = re_exported[0]["vector"]
    by_idx = dict(zip(v["indices"], v["values"]))
    assert abs(by_idx[5] - 0.7) < 1e-6
    assert abs(by_idx[999] - 0.2) < 1e-6


# ---------------------------------------------------------------------------
# 7. Engine: create_collection with SPARSE_FLOAT_VECTOR
# ---------------------------------------------------------------------------

def test_engine_create_sparse_collection(tmp_path):
    mod = load_engine_module()
    engine = mod.HypervecServerEngine(str(tmp_path), hypervec_module=FakeHypervec())
    result = engine.create_collection("sp_col", schema=SPARSE_SCHEMA)
    assert result["collection_name"] == "sp_col"
    meta = engine.meta_store.get("sp_col")
    assert meta is not None
    assert meta.vector_field == "sparse_vec"
    assert meta.dim is None


# ---------------------------------------------------------------------------
# 8. Engine: insert sparse rows (dict format and indices/values format)
# ---------------------------------------------------------------------------

def test_engine_insert_sparse_dict_format(tmp_path):
    mod = load_engine_module()
    engine = mod.HypervecServerEngine(str(tmp_path), hypervec_module=FakeHypervec())
    engine.create_collection("sp_col", schema=SPARSE_SCHEMA)

    rows = [
        {"id": "doc0", "sparse_vec": {0: 0.5, 10: 1.0}, "contents": "hello"},
        {"id": "doc1", "sparse_vec": {5: 0.3}, "contents": "world"},
    ]
    result = engine.insert("sp_col", rows)
    assert result["insert_count"] == 2
    assert result["total"] == 2


def test_engine_insert_sparse_indices_values_format(tmp_path):
    mod = load_engine_module()
    engine = mod.HypervecServerEngine(str(tmp_path), hypervec_module=FakeHypervec())
    engine.create_collection("sp_col", schema=SPARSE_SCHEMA)

    rows = [
        {
            "id": "doc0",
            "sparse_vec": {"indices": [3, 77], "values": [0.4, 0.6]},
            "contents": "test",
        }
    ]
    result = engine.insert("sp_col", rows)
    assert result["insert_count"] == 1
    exported = engine.scalar_store.export_rows("sp_col")
    v = exported[0]["vector"]
    by_idx = dict(zip(v["indices"], v["values"]))
    assert abs(by_idx[3] - 0.4) < 1e-6
    assert abs(by_idx[77] - 0.6) < 1e-6


# ---------------------------------------------------------------------------
# 9. Engine: flush raises NotImplementedError for sparse collections
# ---------------------------------------------------------------------------

def test_engine_flush_sparse_raises(tmp_path):
    mod = load_engine_module()
    engine = mod.HypervecServerEngine(str(tmp_path), hypervec_module=FakeHypervec())
    engine.create_collection("sp_col", schema=SPARSE_SCHEMA)
    engine.insert("sp_col", [{"id": "d0", "sparse_vec": {1: 0.9}, "contents": ""}])
    with pytest.raises(NotImplementedError, match="sparse index building"):
        engine.flush("sp_col")


# ---------------------------------------------------------------------------
# 10. Engine: search raises NotImplementedError for sparse collections
# ---------------------------------------------------------------------------

def test_engine_search_sparse_raises(tmp_path):
    mod = load_engine_module()
    engine = mod.HypervecServerEngine(str(tmp_path), hypervec_module=FakeHypervec())
    engine.create_collection("sp_col", schema=SPARSE_SCHEMA)
    engine.insert("sp_col", [{"id": "d0", "sparse_vec": {1: 0.9}, "contents": ""}])
    with pytest.raises(NotImplementedError, match="sparse vector search"):
        engine.search("sp_col", data=[[0.1, 0.2]], limit=1)


# ---------------------------------------------------------------------------
# 11. Engine: export_rows returns correct sparse JSON format
# ---------------------------------------------------------------------------

def test_engine_export_rows_sparse(tmp_path):
    mod = load_engine_module()
    engine = mod.HypervecServerEngine(str(tmp_path), hypervec_module=FakeHypervec())
    engine.create_collection("sp_col", schema=SPARSE_SCHEMA)
    engine.insert("sp_col", [
        {"id": "d0", "sparse_vec": {0: 0.1, 5: 0.8}, "contents": "foo"},
    ])
    rows = engine.scalar_store.export_rows("sp_col")
    assert len(rows) == 1
    v = rows[0]["vector"]
    assert isinstance(v, dict)
    assert "indices" in v and "values" in v
    by_idx = dict(zip(v["indices"], v["values"]))
    assert abs(by_idx[0] - 0.1) < 1e-6
    assert abs(by_idx[5] - 0.8) < 1e-6


# ---------------------------------------------------------------------------
# 12. Regression: dense collection is unaffected
# ---------------------------------------------------------------------------

def test_dense_collection_unaffected(tmp_path):
    mod = load_engine_module()
    engine = mod.HypervecServerEngine(str(tmp_path), hypervec_module=FakeHypervec())
    index_params = {"indexes": [{"field_name": "vector", "metric_type": "L2",
                                  "index_type": "Flat", "params": {}}]}
    engine.create_collection("dense_col", schema=DENSE_SCHEMA, index_params=index_params)
    engine.insert("dense_col", [
        {"id": "d0", "vector": [1.0, 0.0, 0.0, 0.0], "contents": "a"},
        {"id": "d1", "vector": [0.0, 1.0, 0.0, 0.0], "contents": "b"},
    ])
    flush_result = engine.flush("dense_col")
    assert flush_result["flushed"]
    assert flush_result["total"] == 2
    assert flush_result["dim"] == 4
    search_result = engine.search(
        "dense_col", data=[[1.0, 0.0, 0.0, 0.0]], limit=1
    )
    assert len(search_result) == 1
    assert search_result[0][0]["id"] == "d0"


# ---------------------------------------------------------------------------
# 13. dim stays None for sparse, is set for dense
# ---------------------------------------------------------------------------

def test_meta_dim_none_for_sparse(tmp_path):
    mod = load_engine_module()
    engine = mod.HypervecServerEngine(str(tmp_path), hypervec_module=FakeHypervec())
    engine.create_collection("sp", schema=SPARSE_SCHEMA)
    engine.insert("sp", [{"id": "x", "sparse_vec": {0: 1.0, 9999: 0.5}, "contents": ""}])
    meta = engine.meta_store.get("sp")
    assert meta.dim is None


def test_meta_dim_set_for_dense(tmp_path):
    mod = load_engine_module()
    engine = mod.HypervecServerEngine(str(tmp_path), hypervec_module=FakeHypervec())
    index_params = {"indexes": [{"field_name": "vector", "metric_type": "L2",
                                  "index_type": "Flat", "params": {}}]}
    engine.create_collection("dense", schema=DENSE_SCHEMA, index_params=index_params)
    engine.insert("dense", [{"id": "y", "vector": [0.1, 0.2, 0.3, 0.4], "contents": ""}])
    meta = engine.meta_store.get("dense")
    assert meta.dim == 4
