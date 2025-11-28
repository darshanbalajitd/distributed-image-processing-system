#include <string>
#include <vector>
#include <cstdint>

namespace dip {
static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64_encode(const std::vector<unsigned char>& data) {
    std::string out;
    size_t i = 0;
    while (i + 2 < data.size()) {
        uint32_t triple = (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8) | uint32_t(data[i + 2]);
        out.push_back(table[(triple >> 18) & 0x3F]);
        out.push_back(table[(triple >> 12) & 0x3F]);
        out.push_back(table[(triple >> 6) & 0x3F]);
        out.push_back(table[triple & 0x3F]);
        i += 3;
    }
    if (i < data.size()) {
        uint32_t triple = uint32_t(data[i]) << 16;
        bool two = false;
        if (i + 1 < data.size()) {
            triple |= uint32_t(data[i + 1]) << 8;
            two = true;
        }
        out.push_back(table[(triple >> 18) & 0x3F]);
        out.push_back(table[(triple >> 12) & 0x3F]);
        out.push_back(two ? table[(triple >> 6) & 0x3F] : '=');
        out.push_back('=');
    }
    return out;
}
}