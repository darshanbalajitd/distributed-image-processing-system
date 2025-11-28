#pragma once
#include <string>
#include <functional>
#include <memory>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

namespace dip {
class TcpServer {
public:
    using OnMessage = std::function<void(const std::string&, SOCKET)>;
    TcpServer(const std::string& bind_addr, uint16_t port, OnMessage on_message);
    void start();
private:
    SOCKET listen_ = INVALID_SOCKET;
    OnMessage on_message_;
    void do_accept();
    static bool read_exact(SOCKET sock, char* buf, size_t len);
    void connection_loop(SOCKET sock);
};
}
