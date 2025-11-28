#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace dip {
std::string encode_length_prefixed(const std::string& payload);
bool decode_length_prefixed(const std::vector<uint8_t>& buf, size_t& offset, std::string& out);
}

