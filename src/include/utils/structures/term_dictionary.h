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
#include <string>
#include <unordered_map>
#include <vector>

namespace hypervec {

struct IOReader;
struct IOWriter;

/** A term-string <-> term-id dictionary, aligned with knowhere's `dim_map_`.
 *
 * Maps opaque string terms (words / tokens) to dense unsigned 32-bit ids.  Ids
 * are assigned in *first-seen order* (0, 1, 2, ...); this ordering is a
 * cross-language contract shared with the Python mirror
 * (src/python/hypervec_term_dictionary.py), so both sides must allocate ids
 * identically for the same stream of terms.
 *
 * Serialized layout (little-endian, tightly packed, no padding):
 *
 *   [uint32 n_terms][ (uint32 term_len, bytes term_utf8) * n_terms ]
 *
 * Entries are written in ascending term-id order (i.e. first-seen order), so
 * the i-th written term has id i and no explicit id is stored.
 */
struct TermDictionary {
  /// Return the id of `term`, allocating the next id if unseen (first-seen
  /// order).  This is the only mutating lookup.
  uint32_t get_or_add(const std::string& term);

  /// Whether `term` already has an id.
  bool contains(const std::string& term) const;

  /// Id of an existing `term`.  Throws if `term` is unknown.
  uint32_t id_of(const std::string& term) const;

  /// Term for an existing `id`.  Throws if `id` is out of range.
  const std::string& term_of(uint32_t id) const;

  /// Number of terms in the dictionary.
  size_t size() const { return id_to_term_.size(); }

  std::unordered_map<std::string, uint32_t> term_to_id_;
  std::vector<std::string> id_to_term_;  // index == term id
};

/** Write a TermDictionary to `f` in the layout documented above. */
void write_term_dictionary(const TermDictionary& dict, IOWriter* f);

/** Read a TermDictionary written by write_term_dictionary.  Bounds-checks the
 *  term count and each term length against the deserialization byte limit. */
TermDictionary read_term_dictionary(IOReader* f);

}  // namespace hypervec
