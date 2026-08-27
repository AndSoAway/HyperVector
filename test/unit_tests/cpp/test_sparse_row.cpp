/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <utils/structures/sparse_row.h>

#include <gtest/gtest.h>
#include <persistence/io.h>

#include <cstdint>
#include <cstring>
#include <vector>

using namespace hypervec;

namespace {

// Serialize a SparseRow to a byte buffer via the VectorIOWriter.
std::vector<uint8_t> Serialize(const SparseRow& row) {
  VectorIOWriter wr;
  write_sparse_row(row, &wr);
  return wr.data;
}

// Round-trip: serialize then read back.
SparseRow RoundTrip(const SparseRow& row) {
  VectorIOWriter wr;
  write_sparse_row(row, &wr);
  VectorIOReader rd;
  rd.data = wr.data;
  return read_sparse_row(&rd);
}

}  // namespace

TEST(SparseRow, ConstructAccessors) {
  SparseRow row({100, 0, 50000}, {1.2f, 0.5f, 0.01f});
  ASSERT_EQ(row.nnz(), 3u);
  // Entries are sorted by index ascending.
  EXPECT_EQ(row.index_at(0), 0u);
  EXPECT_EQ(row.index_at(1), 100u);
  EXPECT_EQ(row.index_at(2), 50000u);
  EXPECT_FLOAT_EQ(row.value_at(0), 0.5f);
  EXPECT_FLOAT_EQ(row.value_at(1), 1.2f);
  EXPECT_FLOAT_EQ(row.value_at(2), 0.01f);
}

TEST(SparseRow, ConstructMismatchedLengthsThrows) {
  EXPECT_ANY_THROW(SparseRow({1, 2}, {0.5f}));
}

TEST(SparseRow, SetMaintainsSortedOrder) {
  SparseRow row;
  row.set(50, 0.5f);
  row.set(10, 0.1f);
  row.set(30, 0.3f);
  ASSERT_EQ(row.nnz(), 3u);
  EXPECT_EQ(row.index_at(0), 10u);
  EXPECT_EQ(row.index_at(1), 30u);
  EXPECT_EQ(row.index_at(2), 50u);
  // Overwrite existing index does not grow nnz.
  row.set(30, 0.99f);
  EXPECT_EQ(row.nnz(), 3u);
  EXPECT_FLOAT_EQ(row.value_at(1), 0.99f);
}

TEST(SparseRow, EmptyRoundTrip) {
  SparseRow empty;
  std::vector<uint8_t> bytes = Serialize(empty);
  ASSERT_EQ(bytes.size(), 4u);  // just the uint32 nnz header
  uint32_t nnz = 0;
  std::memcpy(&nnz, bytes.data(), 4);
  EXPECT_EQ(nnz, 0u);
  EXPECT_EQ(RoundTrip(empty), empty);
}

TEST(SparseRow, RoundTrip) {
  SparseRow row({0, 100, 50000}, {0.5f, 1.2f, 0.01f});
  SparseRow decoded = RoundTrip(row);
  EXPECT_EQ(decoded, row);
}

// Cross-language byte-layout contract: must match the Python
// test_encode_sparse_binary_layout ({3: 0.25} -> 12 bytes) in
// test/unit_tests/python/test_sparse_vector.py.
TEST(SparseRow, ByteLayoutMatchesPythonContract) {
  SparseRow row({3}, {0.25f});
  std::vector<uint8_t> bytes = Serialize(row);
  ASSERT_EQ(bytes.size(), 4u + 8u);  // nnz header + 1 pair

  uint32_t nnz = 0;
  std::memcpy(&nnz, bytes.data() + 0, 4);
  EXPECT_EQ(nnz, 1u);

  uint32_t idx = 0;
  std::memcpy(&idx, bytes.data() + 4, 4);
  EXPECT_EQ(idx, 3u);

  float val = 0.0f;
  std::memcpy(&val, bytes.data() + 8, 4);
  EXPECT_FLOAT_EQ(val, 0.25f);
}

// The packed element must be exactly 8 bytes and data() must expose the
// contiguous array so that pointer stepping matches the accessors.
TEST(SparseRow, DataPointerPacked) {
  static_assert(sizeof(SparseElement) == 8, "SparseElement must be 8 bytes");
  SparseRow row({0, 100, 50000}, {0.5f, 1.2f, 0.01f});
  const SparseElement* p = row.data();
  ASSERT_EQ(row.nnz(), 3u);
  EXPECT_EQ(p[0].index, 0u);
  EXPECT_EQ(p[1].index, 100u);
  EXPECT_EQ(p[2].index, 50000u);
  EXPECT_FLOAT_EQ(p[1].value, 1.2f);
  // Pointer stepping agrees with index_at / value_at.
  for (size_t i = 0; i < row.nnz(); ++i) {
    EXPECT_EQ(p[i].index, row.index_at(i));
    EXPECT_FLOAT_EQ(p[i].value, row.value_at(i));
  }
}

TEST(SparseRow, DimReturnsMaxPlusOne) {
  EXPECT_EQ(SparseRow().dim(), 0u);
  SparseRow row({3, 0, 99}, {0.1f, 0.2f, 0.3f});
  EXPECT_EQ(row.dim(), 100u);  // max index 99 + 1
}

// A zero-copy view shares the backing bytes and is read-only.
TEST(SparseRow, ViewSharesBytesAndIsReadOnly) {
  SparseRow owning({1, 2, 3}, {0.1f, 0.2f, 0.3f});
  SparseRow view = SparseRow::create_view(
    const_cast<SparseElement*>(owning.data()), owning.nnz(), nullptr);
  ASSERT_EQ(view.nnz(), 3u);
  EXPECT_EQ(view.index_at(2), 3u);
  EXPECT_FLOAT_EQ(view.value_at(1), 0.2f);
  EXPECT_EQ(view, owning);
  // set() on a view asserts (HYPERVEC_ASSERT aborts); use death test.
  EXPECT_DEATH_IF_SUPPORTED(view.set(4, 0.4f), "");
}

// Plain sparse inner product over shared indices.
TEST(SparseRow, DotInnerProduct) {
  SparseRow a({0, 2, 5}, {1.0f, 2.0f, 3.0f});
  SparseRow b({2, 5, 9}, {4.0f, 5.0f, 6.0f});
  // Shared indices: 2 (2*4=8) and 5 (3*5=15) => 23.
  EXPECT_FLOAT_EQ(a.dot(b), 23.0f);
  // Symmetric.
  EXPECT_FLOAT_EQ(b.dot(a), 23.0f);
  // No shared indices => 0.
  SparseRow c({1, 3}, {7.0f, 8.0f});
  EXPECT_FLOAT_EQ(a.dot(c), 0.0f);
}

TEST(SparseRow, EmptyDot) {
  SparseRow empty;
  SparseRow a({0, 1}, {1.0f, 2.0f});
  EXPECT_FLOAT_EQ(empty.dot(a), 0.0f);
  EXPECT_FLOAT_EQ(a.dot(empty), 0.0f);
  EXPECT_FLOAT_EQ(empty.dot(empty), 0.0f);
}

// BM25 score: doc holds term frequencies, query holds IDF weights.
TEST(SparseRow, DotBM25) {
  // doc term frequencies at ids 0,1,2; query IDF at ids 1,2,3.
  SparseRow doc({0, 1, 2}, {3.0f, 1.0f, 2.0f});
  SparseRow query_idf({1, 2, 3}, {0.5f, 1.5f, 2.0f});
  BM25Params params{1.2f, 0.75f, 4.0f};
  const float doc_len = 6.0f;  // sum of tf

  const float avgdl = 4.0f;
  const float norm = params.k1 * (1.0f - params.b + params.b * doc_len / avgdl);
  auto bm25 = [&](float tf) {
    return tf * (params.k1 + 1.0f) / (tf + norm);
  };
  // Shared ids 1 (tf=1, idf=0.5) and 2 (tf=2, idf=1.5).
  const float expected = 0.5f * bm25(1.0f) + 1.5f * bm25(2.0f);
  EXPECT_FLOAT_EQ(doc.dot_bm25(query_idf, params, doc_len), expected);
}
