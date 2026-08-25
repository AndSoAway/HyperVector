/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hypervec {

struct IOReader;
struct IOWriter;

/** A single sparse float vector, aligned with Milvus/knowhere SparseRow.
 *
 * Stores `nnz` non-zero entries as parallel (index, value) arrays kept sorted
 * by index ascending.  Indices are unsigned 32-bit; values are 32-bit float.
 *
 * On-disk / serialized layout (little-endian, tightly packed, no padding):
 *
 *   [uint32 nnz][ (uint32 index, float32 value) * nnz ]   // sorted by index
 *
 * Total size == 4 + 8 * nnz bytes.  This byte layout is a cross-language
 * contract shared with the Python ScalarStore (_encode_sparse / _decode_sparse
 * in src/python/hypervec_scalar_store.py): a blob produced here decodes there
 * and vice versa.
 */
struct SparseRow {
  SparseRow() = default;

  /// Construct from parallel arrays; entries are re-sorted by index ascending.
  /// Throws if `indices` and `values` differ in length.
  SparseRow(std::vector<uint32_t> indices, std::vector<float> values);

  /// Number of non-zero entries.
  size_t nnz() const { return indices_.size(); }

  /// Index of the i-th stored entry (0 <= i < nnz()).  No bounds check.
  uint32_t index_at(size_t i) const { return indices_[i]; }

  /// Value of the i-th stored entry (0 <= i < nnz()).  No bounds check.
  float value_at(size_t i) const { return values_[i]; }

  /// Set (or overwrite) the value at `index`, keeping entries sorted by index.
  void set(uint32_t index, float value);

  bool operator==(const SparseRow& other) const {
    return indices_ == other.indices_ && values_ == other.values_;
  }
  bool operator!=(const SparseRow& other) const { return !(*this == other); }

  std::vector<uint32_t> indices_;
  std::vector<float> values_;
};

/** Write a SparseRow to `f` in the byte layout documented on SparseRow.
 *
 * Writes a uint32 nnz followed by nnz interleaved (uint32 index, float32 value)
 * pairs.  Deliberately does NOT use WRITEVECTOR (which prefixes an 8-byte
 * size_t and stores arrays contiguously rather than interleaved) so the output
 * byte-matches the Python contract.
 */
void write_sparse_row(const SparseRow& row, IOWriter* f);

/** Read a SparseRow written by write_sparse_row.  Bounds-checks nnz against the
 *  deserialization byte limit to guard against corrupt / hostile input. */
SparseRow read_sparse_row(IOReader* f);

}  // namespace hypervec
