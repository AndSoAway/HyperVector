/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <utils/algo/sparse_vector/sparse_inverted_index.h>
#include <utils/structures/sparse_row.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace hypervec;

namespace {

// A minimal in-test implementation of the interface.  Its sole purpose is to
// prove, at compile time, that the SparseInvertedIndex SHAPE is implementable
// and callable through the abstract base — it implements no real algorithm.
struct MockSparseInvertedIndex : SparseInvertedIndex {
  idx_t added = 0;
  idx_t dim_ = 0;

  void Add(idx_t n, const SparseRow* rows, idx_t dim) override {
    (void)rows;
    added += n;
    dim_ = dim;
  }

  void Search(const SparseRow& query, idx_t k, float* distances, idx_t* labels,
              const SparseSearchParameters* params = nullptr) const override {
    (void)query;
    (void)params;
    for (idx_t i = 0; i < k; ++i) {
      distances[i] = 0.0f;
      labels[i] = -1;
    }
  }

  float row_sum(idx_t row) const override {
    (void)row;
    return 0.0f;
  }

  idx_t dim() const override { return dim_; }
};

}  // namespace

// Compile-check: the interface can be implemented and driven through the base
// pointer, and SparseSearchParameters up-casts to SearchParameters*.
TEST(SparseInvertedIndex, InterfaceIsImplementable) {
  MockSparseInvertedIndex mock;
  SparseInvertedIndex* p = &mock;

  SparseRow rows[2] = {SparseRow({0, 2}, {1.0f, 2.0f}),
                       SparseRow({1}, {3.0f})};
  p->Add(2, rows, 3);
  EXPECT_EQ(p->dim(), 3);

  SparseSearchParameters params;
  params.k1 = 1.5f;
  params.b = 0.6f;
  params.avgdl = 10.0f;
  const SearchParameters* base = &params;  // up-cast must compile
  EXPECT_NE(base, nullptr);
}

// Compile-check: the Search signature is usable with caller-allocated buffers.
TEST(SparseInvertedIndex, SearchSignatureUsable) {
  MockSparseInvertedIndex mock;
  SparseRow query({0, 5}, {0.5f, 0.5f});
  const idx_t k = 3;
  std::vector<float> distances(k);
  std::vector<idx_t> labels(k);
  SparseSearchParameters params;
  mock.Search(query, k, distances.data(), labels.data(), &params);
  EXPECT_EQ(labels[0], -1);  // mock pads with -1
}

// row_sum() on SparseRow = sum of term frequencies (BM25 document length).
TEST(SparseRow, RowSum) {
  SparseRow row({0, 5, 9}, {1.0f, 2.0f, 0.5f});
  EXPECT_FLOAT_EQ(row.row_sum(), 3.5f);

  SparseRow empty;
  EXPECT_FLOAT_EQ(empty.row_sum(), 0.0f);
}
