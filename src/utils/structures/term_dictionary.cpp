/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <utils/structures/term_dictionary.h>

#include <persistence/io.h>
#include <persistence/io_macros.h>
#include <utils/log/assert.h>

#include <cerrno>
#include <cstring>

namespace hypervec {

uint32_t TermDictionary::get_or_add(const std::string& term) {
  auto it = term_to_id_.find(term);
  if (it != term_to_id_.end()) {
    return it->second;
  }
  const uint32_t id = static_cast<uint32_t>(id_to_term_.size());
  id_to_term_.push_back(term);
  term_to_id_.emplace(term, id);
  return id;
}

bool TermDictionary::contains(const std::string& term) const {
  return term_to_id_.find(term) != term_to_id_.end();
}

uint32_t TermDictionary::id_of(const std::string& term) const {
  auto it = term_to_id_.find(term);
  HYPERVEC_THROW_IF_NOT_FMT(it != term_to_id_.end(),
                            "TermDictionary: unknown term '%s'", term.c_str());
  return it->second;
}

const std::string& TermDictionary::term_of(uint32_t id) const {
  HYPERVEC_THROW_IF_NOT_FMT(static_cast<size_t>(id) < id_to_term_.size(),
                            "TermDictionary: term id %u out of range (size %zd)",
                            id, id_to_term_.size());
  return id_to_term_[id];
}

void write_term_dictionary(const TermDictionary& dict, IOWriter* f) {
  uint32_t n_terms = static_cast<uint32_t>(dict.id_to_term_.size());
  WRITE1(n_terms);
  // Write in ascending id order so the reader can reconstruct ids implicitly.
  for (uint32_t id = 0; id < n_terms; ++id) {
    const std::string& term = dict.id_to_term_[id];
    uint32_t term_len = static_cast<uint32_t>(term.size());
    WRITE1(term_len);
    if (term_len > 0) {
      const char* bytes = term.data();
      WRITEANDCHECK(bytes, term_len);
    }
  }
}

TermDictionary read_term_dictionary(IOReader* f) {
  uint32_t n_terms = 0;
  READ1(n_terms);
  // Guard against corrupt / hostile counts.
  HYPERVEC_THROW_IF_NOT(static_cast<size_t>(n_terms) <
                        get_deserialization_vector_byte_limit());
  TermDictionary dict;
  dict.id_to_term_.reserve(n_terms);
  for (uint32_t id = 0; id < n_terms; ++id) {
    uint32_t term_len = 0;
    READ1(term_len);
    HYPERVEC_THROW_IF_NOT(static_cast<size_t>(term_len) <
                          get_deserialization_vector_byte_limit());
    std::string term(term_len, '\0');
    if (term_len > 0) {
      char* bytes = &term[0];
      READANDCHECK(bytes, term_len);
    }
    // First-seen order == ascending id, so this rebuilds the same mapping.
    dict.id_to_term_.push_back(term);
    dict.term_to_id_.emplace(std::move(term), id);
  }
  return dict;
}

}  // namespace hypervec
