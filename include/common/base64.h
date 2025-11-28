#pragma once
#include <string>
#include <vector>

namespace dip {
std::string base64_encode(const std::vector<unsigned char>& data);
}