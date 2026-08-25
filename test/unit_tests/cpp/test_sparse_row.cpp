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
