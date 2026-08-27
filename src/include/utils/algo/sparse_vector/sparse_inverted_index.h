/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */
#pragma once

#include <index/index.h>                   // SearchParameters, idx_t
#include <utils/structures/sparse_row.h>   // SparseRow

namespace hypervec {

/** BM25 search parameters for a sparse inverted index.
 *
 * Extends the shared SearchParameters base (defined in index.h).  SHAPE ONLY —
 * these are the parameters the future BM25 inverted-index task will consume;
 * no scoring is implemented here.  See knowhere GetDocValueBM25Computer.
 */
struct SparseSearchParameters : SearchParameters {
  float k1 = 1.2f;
  float b = 0.75f;
  float avgdl = 1.0f;  // average document length; clamped to >= 1.0 when applied
};

/** Abstract interface mirroring Milvus/knowhere's InvertedIndex SHAPE.
 *
 * Pure-virtual, header-only.  This declares the interface FORM that the future
 * BM25 / sparse text-retrieval task will implement (concrete inverted index,
 * posting encoding, WAND/MaxScore pruning, BM25 scoring); this header
 * deliberately contains NO algorithm.  It exists so the data-model layer
 * (SparseRow + TermDictionary) has a stable downstream contract to build on.
 *
 * Conventions mirror hypervec::Index (index.h): PascalCase virtuals, idx_t
 * counts/labels, raw out-params for distances/labels, optional params pointer.
 */
struct SparseInvertedIndex {
  virtual ~SparseInvertedIndex() = default;

  /// Add `n` sparse rows (term-id space) of dimensionality `dim`.
  virtual void Add(idx_t n, const SparseRow* rows, idx_t dim) = 0;

  /// k-NN sparse search.  `distances` and `labels` are caller-allocated
  /// out-params of length `k` (padded with -1 / +inf when fewer than k hits).
  virtual void Search(const SparseRow& query, idx_t k, float* distances,
                      idx_t* labels,
                      const SparseSearchParameters* params = nullptr) const = 0;

  /// BM25 document length of stored `row` (sum of its term frequencies).
  virtual float row_sum(idx_t row) const = 0;

  /// Highest term id + 1 across the whole index.
  virtual idx_t dim() const = 0;
};

}  // namespace hypervec
