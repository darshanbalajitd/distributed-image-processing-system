#include <string>
#include <vector>
#include <winsock2.h>
#include <ws2tcpip.h>
#include "networking/tcp_client.h"
#include "networking/protocol.h"

namespace dip {
TcpClient::TcpClient() { WSADATA wsa; WSAStartup(MAKEWORD(2,2), &wsa); }
bool TcpClient::connect(const std::string& host, uint16_t port) {
    sock_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons(port); inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
    return ::connect(sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
}
bool TcpClient::send(const std::string& payload) {
    auto framed = encode_length_prefixed(payload);
    int n = ::send(sock_, framed.data(), static_cast<int>(framed.size()), 0);
    return n == static_cast<int>(framed.size());
}
bool TcpClient::read(std::string& out) {
    std::vector<uint8_t> buf(65536);
    int n = ::recv(sock_, reinterpret_cast<char*>(buf.data()), static_cast<int>(buf.size()), 0);
    if (n <= 0) return false;
    buf.resize(n);
    size_t off = 0; return decode_length_prefixed(buf, off, out);
}
}
