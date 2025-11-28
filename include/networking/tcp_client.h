#pragma once
#include <string>
#include <winsock2.h>
#include <ws2tcpip.h>

namespace dip {
class TcpClient {
public:
    TcpClient();
    bool connect(const std::string& host, uint16_t port);
    bool send(const std::string& payload);
    bool read(std::string& out);
private:
    SOCKET sock_ = INVALID_SOCKET;
};
}
