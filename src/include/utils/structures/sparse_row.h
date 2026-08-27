/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */
#pragma once

#include <utils/common/platform_macros.h>
#include <utils/structures/maybe_owned_vector.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace hypervec {

struct IOReader;
struct IOWriter;

/** A single packed (index, value) entry of a SparseRow.
 *
 * Tightly packed to exactly 8 bytes so an array of these mirrors the on-disk /
 * cross-language byte layout with no padding.  Aligned with Milvus/knowhere's
 * SparseRow element (`{table_t index; fp32 value;}`).
 */
HYPERVEC_PACK_STRUCTS_BEGIN
struct HYPERVEC_PACKED SparseElement {
  uint32_t index;
  float value;
};
HYPERVEC_PACK_STRUCTS_END

static_assert(sizeof(SparseElement) == 8,
              "SparseElement must be tightly packed to 8 bytes");

/** Signature of an injectable doc-value transform used by SparseRow::dot.
 *
 * Given a raw stored document value and an auxiliary per-document quantity
 * (e.g. document length for BM25), returns the effective value to accumulate.
 * A function pointer (not std::function) is used deliberately to match the
 * repository convention of avoiding runtime std::function in headers; the
 * knowhere analogue is `DocValueComputer = std::function<float(float,float)>`.
 */
using DocValueComputer = float (*)(float doc_value, float doc_extra);

/** BM25 scoring parameters (see GetDocValueBM25Computer in knowhere). */
struct BM25Params {
  float k1;
  float b;
  float avgdl;  // average document length; clamped to >= 1.0 when applied
};

/** A single sparse float vector, aligned with Milvus/knowhere SparseRow.
 *
 * Stores `nnz` non-zero entries as a contiguous, tightly packed array of
 * `SparseElement` ({uint32 index, float32 value}), kept sorted by index
 * ascending.  The backing store is a MaybeOwnedVector<uint8_t>, so a row can
 * either own its bytes or be a zero-copy view over an external (e.g. mmap'd)
 * buffer.  data() exposes the packed array as a raw pointer for SIMD / merge.
 *
 * On-disk / serialized layout (little-endian, tightly packed, no padding):
 *
 *   [uint32 nnz][ (uint32 index, float32 value) * nnz ]   // sorted by index
 *
 * Total size == 4 + 8 * nnz bytes.  This byte layout is a cross-language
 * contract shared with the Python ScalarStore (_encode_sparse / _decode_sparse
 * in src/python/hypervec_scalar_store.py): a blob produced here decodes there
 * and vice versa.  The `index` field holds a term id, not a term string; the
 * string term dictionary is maintained separately (see term_dictionary.h).
 */
struct SparseRow {
  SparseRow() = default;

  /// Construct from parallel arrays; entries are re-sorted by index ascending.
  /// Throws if `indices` and `values` differ in length.
  SparseRow(std::vector<uint32_t> indices, std::vector<float> values);

  /// Construct a zero-copy view over `nnz` packed SparseElements at `address`.
  /// `owner` keeps the backing buffer alive; may be nullptr.  The resulting
  /// row is read-only (set() will assert).
  static SparseRow create_view(
    void* address, size_t nnz,
    const std::shared_ptr<MaybeOwnedVectorOwner>& owner);

  /// Number of non-zero entries.
  size_t nnz() const { return buf_.byte_size() / sizeof(SparseElement); }

  /// Highest index + 1 (0 for an empty row).  Mirrors knowhere SparseRow::dim.
  uint32_t dim() const;

  /// Sum of all stored values (BM25 document length = sum of term freqs).
  /// O(nnz).  Header-only inline, consistent with nnz()/index_at/value_at.
  float row_sum() const {
    float s = 0.0f;
    for (size_t i = 0; i < nnz(); ++i) s += value_at(i);
    return s;
  }

  /// Raw pointer to the packed element array (nnz() elements).
  const SparseElement* data() const {
    return reinterpret_cast<const SparseElement*>(buf_.data());
  }

  /// Index of the i-th stored entry (0 <= i < nnz()).  No bounds check.
  uint32_t index_at(size_t i) const { return data()[i].index; }

  /// Value of the i-th stored entry (0 <= i < nnz()).  No bounds check.
  float value_at(size_t i) const { return data()[i].value; }

  /// Set (or overwrite) the value at `index`, keeping entries sorted by index.
  /// Only valid on an owning row (a view will assert via MaybeOwnedVector).
  void set(uint32_t index, float value);

  /// Sparse inner product with `other` via a two-pointer merge over the shared
  /// indices.  When `computer` is non-null, each contributing `other` value is
  /// transformed by `computer(other.value, other_extra)` before multiplication;
  /// otherwise a plain inner product is computed.
  float dot(const SparseRow& other, DocValueComputer computer = nullptr,
            float other_extra = 0.0f) const;

  /// BM25 score with `*this` as the document (values = term frequencies) and
  /// `query_idf` as the query (values = IDF weights).  `doc_len` is this
  /// document's length (typically the sum of its term frequencies).
  float dot_bm25(const SparseRow& query_idf, const BM25Params& params,
                 float doc_len) const;

  bool operator==(const SparseRow& other) const { return buf_ == other.buf_; }
  bool operator!=(const SparseRow& other) const { return !(*this == other); }

  MaybeOwnedVector<uint8_t> buf_;  // packed SparseElement[nnz], sorted by index
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
