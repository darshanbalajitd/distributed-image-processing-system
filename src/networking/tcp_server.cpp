#include <winsock2.h>
#include <ws2tcpip.h>
#include <vector>
#include <string>
#include <thread>
#include "networking/tcp_server.h"
#include "networking/protocol.h"

namespace dip {
TcpServer::TcpServer(const std::string& bind_addr, uint16_t port, OnMessage on_message) : on_message_(on_message) {
    WSADATA wsa; WSAStartup(MAKEWORD(2,2), &wsa);
    listen_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons(port); inet_pton(AF_INET, bind_addr.c_str(), &addr.sin_addr);
    ::bind(listen_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    listen(listen_, SOMAXCONN);
}
void TcpServer::start() { do_accept(); }
void TcpServer::do_accept() {
    while (true) {
        SOCKET sock = accept(listen_, nullptr, nullptr);
        if (sock == INVALID_SOCKET) break;
        std::thread([this, sock]{ connection_loop(sock); }).detach();
    }
}
bool TcpServer::read_exact(SOCKET sock, char* buf, size_t len) {
    size_t read_total = 0;
    while (read_total < len) {
        int n = ::recv(sock, buf + read_total, static_cast<int>(len - read_total), 0);
        if (n <= 0) return false;
        read_total += static_cast<size_t>(n);
    }
    return true;
}
void TcpServer::connection_loop(SOCKET sock) {
    for (;;) {
        char hdr[4];
        if (!read_exact(sock, hdr, 4)) break;
        uint32_t len = (uint8_t)hdr[0]; len = (len << 8) | (uint8_t)hdr[1]; len = (len << 8) | (uint8_t)hdr[2]; len = (len << 8) | (uint8_t)hdr[3];
        if (len == 0) continue;
        std::string payload; payload.resize(len);
        if (!read_exact(sock, payload.data(), len)) break;
        on_message_(payload, sock);
    }
}
}
