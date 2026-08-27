/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <utils/structures/term_dictionary.h>

#include <gtest/gtest.h>
#include <persistence/io.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace hypervec;

namespace {

std::vector<uint8_t> Serialize(const TermDictionary& dict) {
  VectorIOWriter wr;
  write_term_dictionary(dict, &wr);
  return wr.data;
}

TermDictionary RoundTrip(const TermDictionary& dict) {
  VectorIOWriter wr;
  write_term_dictionary(dict, &wr);
  VectorIOReader rd;
  rd.data = wr.data;
  return read_term_dictionary(&rd);
}

}  // namespace

TEST(TermDictionary, GetOrAddFirstSeenOrder) {
  TermDictionary dict;
  EXPECT_EQ(dict.get_or_add("banana"), 0u);
  EXPECT_EQ(dict.get_or_add("apple"), 1u);
  EXPECT_EQ(dict.get_or_add("cherry"), 2u);
  EXPECT_EQ(dict.size(), 3u);
}

TEST(TermDictionary, GetOrAddRepeatedReturnsSameId) {
  TermDictionary dict;
  uint32_t first = dict.get_or_add("foo");
  EXPECT_EQ(dict.get_or_add("bar"), 1u);
  EXPECT_EQ(dict.get_or_add("foo"), first);  // no new id
  EXPECT_EQ(dict.size(), 2u);
}

TEST(TermDictionary, LookupRoundTrip) {
  TermDictionary dict;
  dict.get_or_add("alpha");
  dict.get_or_add("beta");
  EXPECT_TRUE(dict.contains("alpha"));
  EXPECT_FALSE(dict.contains("gamma"));
  EXPECT_EQ(dict.id_of("beta"), 1u);
  EXPECT_EQ(dict.term_of(0u), "alpha");
  EXPECT_ANY_THROW(dict.id_of("gamma"));
  EXPECT_ANY_THROW(dict.term_of(99u));
}

TEST(TermDictionary, SerializeRoundTrip) {
  TermDictionary dict;
  dict.get_or_add("the");
  dict.get_or_add("quick");
  dict.get_or_add("brown");
  TermDictionary decoded = RoundTrip(dict);
  ASSERT_EQ(decoded.size(), 3u);
  EXPECT_EQ(decoded.term_of(0u), "the");
  EXPECT_EQ(decoded.term_of(1u), "quick");
  EXPECT_EQ(decoded.term_of(2u), "brown");
  EXPECT_EQ(decoded.id_of("brown"), 2u);
}

TEST(TermDictionary, EmptyRoundTrip) {
  TermDictionary empty;
  std::vector<uint8_t> bytes = Serialize(empty);
  ASSERT_EQ(bytes.size(), 4u);  // just the uint32 n_terms header
  uint32_t n = 0;
  std::memcpy(&n, bytes.data(), 4);
  EXPECT_EQ(n, 0u);
  TermDictionary decoded = RoundTrip(empty);
  EXPECT_EQ(decoded.size(), 0u);
}

// Cross-language byte-layout contract: must match the Python
// TermDictionary.serialize() in src/python/hypervec_term_dictionary.py.
// {"foo"} -> [uint32 n=1][uint32 len=3]["foo"] == 4 + 4 + 3 == 11 bytes.
TEST(TermDictionary, ByteLayoutMatchesPythonContract) {
  TermDictionary dict;
  dict.get_or_add("foo");
  std::vector<uint8_t> bytes = Serialize(dict);
  ASSERT_EQ(bytes.size(), 4u + 4u + 3u);

  uint32_t n = 0;
  std::memcpy(&n, bytes.data() + 0, 4);
  EXPECT_EQ(n, 1u);

  uint32_t term_len = 0;
  std::memcpy(&term_len, bytes.data() + 4, 4);
  EXPECT_EQ(term_len, 3u);

  std::string term(reinterpret_cast<const char*>(bytes.data() + 8), 3);
  EXPECT_EQ(term, "foo");
}
