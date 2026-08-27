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
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

namespace hypervec {

namespace {

// Build an owning packed byte buffer from a sorted list of SparseElements.
MaybeOwnedVector<uint8_t> pack_elements(const std::vector<SparseElement>& elems) {
  MaybeOwnedVector<uint8_t> buf(elems.size() * sizeof(SparseElement));
  if (!elems.empty()) {
    std::memcpy(buf.data(), elems.data(), buf.byte_size());
  }
  return buf;
}

}  // namespace

SparseRow::SparseRow(std::vector<uint32_t> indices, std::vector<float> values) {
  HYPERVEC_THROW_IF_NOT_FMT(
    indices.size() == values.size(),
    "SparseRow: indices size %zd != values size %zd", indices.size(),
    values.size());
  // Pack into SparseElement[] then sort by index ascending to satisfy the
  // layout contract.
  std::vector<SparseElement> elems(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    elems[i].index = indices[i];
    elems[i].value = values[i];
  }
  std::sort(elems.begin(), elems.end(),
            [](const SparseElement& a, const SparseElement& b) {
              return a.index < b.index;
            });
  buf_ = pack_elements(elems);
}

SparseRow SparseRow::create_view(
  void* address, size_t nnz,
  const std::shared_ptr<MaybeOwnedVectorOwner>& owner) {
  SparseRow row;
  row.buf_ = MaybeOwnedVector<uint8_t>::create_view(
    address, nnz * sizeof(SparseElement), owner);
  return row;
}

uint32_t SparseRow::dim() const {
  const size_t n = nnz();
  if (n == 0) {
    return 0;
  }
  // Entries are sorted by index ascending, so the last one holds the max.
  return index_at(n - 1) + 1;
}

void SparseRow::set(uint32_t index, float value) {
  const size_t n = nnz();
  const SparseElement* elems = data();
  // Binary search for the insertion point by index.
  size_t lo = 0;
  size_t hi = n;
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (elems[mid].index < index) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo < n && elems[lo].index == index) {
    // Overwrite existing value in place.  Requires an owning buffer; a viewed
    // row asserts inside MaybeOwnedVector when we take a mutable pointer.
    HYPERVEC_ASSERT_MSG(buf_.is_owned,
                        "SparseRow::set cannot be performed on a viewed row");
    SparseElement* mutable_elems =
      reinterpret_cast<SparseElement*>(buf_.data());
    mutable_elems[lo].value = value;
    return;
  }
  // Insert a new element at position `lo`, preserving order.
  std::vector<SparseElement> elems_copy(n + 1);
  for (size_t i = 0; i < lo; ++i) {
    elems_copy[i] = elems[i];
  }
  elems_copy[lo].index = index;
  elems_copy[lo].value = value;
  for (size_t i = lo; i < n; ++i) {
    elems_copy[i + 1] = elems[i];
  }
  buf_ = pack_elements(elems_copy);
}

float SparseRow::dot(const SparseRow& other, DocValueComputer computer,
                     float other_extra) const {
  const size_t n_a = nnz();
  const size_t n_b = other.nnz();
  const SparseElement* a = data();
  const SparseElement* b = other.data();
  float acc = 0.0f;
  size_t i = 0;
  size_t j = 0;
  // Two-pointer merge over the shared indices (both rows are sorted).
  while (i < n_a && j < n_b) {
    if (a[i].index < b[j].index) {
      ++i;
    } else if (a[i].index > b[j].index) {
      ++j;
    } else {
      const float other_val =
        computer ? computer(b[j].value, other_extra) : b[j].value;
      acc += a[i].value * other_val;
      ++i;
      ++j;
    }
  }
  return acc;
}

float SparseRow::dot_bm25(const SparseRow& query_idf, const BM25Params& params,
                          float doc_len) const {
  const size_t n_doc = nnz();
  const size_t n_q = query_idf.nnz();
  const SparseElement* doc = data();
  const SparseElement* q = query_idf.data();
  const float avgdl = std::max(params.avgdl, 1.0f);
  // BM25 denominator norm term is constant across matched terms for a document.
  const float norm = params.k1 * (1.0f - params.b + params.b * doc_len / avgdl);
  float acc = 0.0f;
  size_t i = 0;
  size_t j = 0;
  while (i < n_doc && j < n_q) {
    if (doc[i].index < q[j].index) {
      ++i;
    } else if (doc[i].index > q[j].index) {
      ++j;
    } else {
      const float tf = doc[i].value;
      const float weighted = tf * (params.k1 + 1.0f) / (tf + norm);
      acc += q[j].value * weighted;  // query value = IDF
      ++i;
      ++j;
    }
  }
  return acc;
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
  std::vector<SparseElement> elems(nnz);
  for (uint32_t i = 0; i < nnz; ++i) {
    uint32_t idx = 0;
    float val = 0.0f;
    READ1(idx);
    READ1(val);
    elems[i].index = idx;
    elems[i].value = val;
  }
  SparseRow row;
  row.buf_ = pack_elements(elems);
  return row;
}

}  // namespace hypervec
