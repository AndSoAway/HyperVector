/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <utils/structures/sparse_row.h>

#include <persistence/io.h>
#include <persistence/io_macros.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <utility>

namespace hypervec {

SparseRow::SparseRow(std::vector<uint32_t> indices, std::vector<float> values) {
  HYPERVEC_THROW_IF_NOT_FMT(
    indices.size() == values.size(),
    "SparseRow: indices size %zd != values size %zd", indices.size(),
    values.size());
  indices_ = std::move(indices);
  values_ = std::move(values);
  // Sort (index, value) pairs by index ascending to satisfy the layout
  // contract.  Build an order permutation to avoid two parallel sorts.
  std::vector<size_t> order(indices_.size());
  for (size_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  std::sort(order.begin(), order.end(),
            [&](size_t a, size_t b) { return indices_[a] < indices_[b]; });
  std::vector<uint32_t> sorted_idx(indices_.size());
  std::vector<float> sorted_val(values_.size());
  for (size_t i = 0; i < order.size(); ++i) {
    sorted_idx[i] = indices_[order[i]];
    sorted_val[i] = values_[order[i]];
  }
  indices_ = std::move(sorted_idx);
  values_ = std::move(sorted_val);
}

void SparseRow::set(uint32_t index, float value) {
  auto it = std::lower_bound(indices_.begin(), indices_.end(), index);
  size_t pos = static_cast<size_t>(it - indices_.begin());
  if (it != indices_.end() && *it == index) {
    values_[pos] = value;  // overwrite existing
    return;
  }
  indices_.insert(it, index);
  values_.insert(values_.begin() + pos, value);
}

void write_sparse_row(const SparseRow& row, IOWriter* f) {
  uint32_t nnz = static_cast<uint32_t>(row.nnz());
  WRITE1(nnz);
  // Interleave (index, value) exactly as Python struct.pack("<If", ...) does;
  // do NOT use WRITEVECTOR (8-byte size_t prefix + non-interleaved arrays).
  for (size_t i = 0; i < row.nnz(); ++i) {
    uint32_t idx = row.index_at(i);
    float val = row.value_at(i);
    WRITE1(idx);
    WRITE1(val);
  }
}

SparseRow read_sparse_row(IOReader* f) {
  uint32_t nnz = 0;
  READ1(nnz);
  // Guard against corrupt / hostile nnz: each entry is 8 bytes on the wire.
  HYPERVEC_THROW_IF_NOT(static_cast<size_t>(nnz) <
                        (get_deserialization_vector_byte_limit() / 8));
  SparseRow row;
  row.indices_.resize(nnz);
  row.values_.resize(nnz);
  for (uint32_t i = 0; i < nnz; ++i) {
    uint32_t idx = 0;
    float val = 0.0f;
    READ1(idx);
    READ1(val);
    row.indices_[i] = idx;
    row.values_[i] = val;
  }
  return row;
}

}  // namespace hypervec
