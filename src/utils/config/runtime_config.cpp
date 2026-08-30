/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <utils/config/runtime_config.h>

#include <sstream>

namespace hypervec {

const std::vector<ConfigOption>& GetConfigOptions() {
  static const std::vector<ConfigOption> options = {
    {"server", "data_root", ConfigValueType::kString, "./data",
     "Root directory for server collection data.", false},
    {"server", "host", ConfigValueType::kString, "127.0.0.1",
     "Server bind host.", false},
    {"server", "port", ConfigValueType::kInt, "8080",
     "Server bind port.", false},
    {"server", "server_mode", ConfigValueType::kString, "http",
     "Startup mode: http, grpc, or dual.", false},
    {"server", "enable_http2", ConfigValueType::kBool, "true",
     "Enable HTTP/2 when the selected ASGI server supports it.", false},
    {"logging", "enable_logging", ConfigValueType::kBool, "true",
     "Global runtime logging switch.", false},
    {"logging", "log_level", ConfigValueType::kString, "info",
     "Global minimum log level.", false},
    {"logging", "log_to_stderr", ConfigValueType::kBool, "true",
     "Write runtime logs to stderr.", false},
    {"logging", "log_to_file", ConfigValueType::kBool, "false",
     "Write runtime logs to a file.", false},
    {"logging", "log_file_path", ConfigValueType::kString,
     "logs/hypervec.log", "Runtime log file path.", true},
  };
  return options;
}

HypervecConfig DefaultRuntimeConfig() {
  return HypervecConfig{};
}

std::string RenderSampleConfig() {
  std::ostringstream out;
  const char* current_section = "";
  for (const auto& option : GetConfigOptions()) {
    if (std::string(current_section) != option.section) {
      if (current_section[0] != '\0') {
        out << "\n";
      }
      current_section = option.section;
      out << "[" << current_section << "]\n";
    }
    out << "# " << option.description << "\n";
    out << option.key << " = " << option.default_value << "\n";
  }
  return out.str();
}

}  // namespace hypervec
