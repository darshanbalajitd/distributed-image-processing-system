#include <string>
#include <vector>
#include <cstdint>

namespace dip {
std::string encode_length_prefixed(const std::string& payload) {
    uint32_t len = static_cast<uint32_t>(payload.size());
    std::string out;
    out.resize(4 + payload.size());
    out[0] = static_cast<char>((len >> 24) & 0xFF);
    out[1] = static_cast<char>((len >> 16) & 0xFF);
    out[2] = static_cast<char>((len >> 8) & 0xFF);
    out[3] = static_cast<char>(len & 0xFF);
    std::memcpy(&out[4], payload.data(), payload.size());
    return out;
}

bool decode_length_prefixed(const std::vector<uint8_t>& buf, size_t& offset, std::string& out) {
    if (offset + 4 > buf.size()) return false;
    uint32_t len = (uint32_t(buf[offset]) << 24) | (uint32_t(buf[offset+1]) << 16) | (uint32_t(buf[offset+2]) << 8) | uint32_t(buf[offset+3]);
    if (offset + 4 + len > buf.size()) return false;
    out.assign(reinterpret_cast<const char*>(&buf[offset+4]), len);
    offset += 4 + len;
    return true;
}
}

