#pragma once
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <winsock2.h>
#include "networking/tcp_server.h"

namespace dip {
struct NetJob { std::string label; std::string id; std::string base64; std::string path; };
class NetMaster {
public:
    using OnResult = std::function<void(const std::string&, const std::string&, const std::vector<float>&)>;
    NetMaster(const std::string& bind_addr, uint16_t port, OnResult on_result);
    void enqueue(const NetJob& job);
    void run();
private:
    OnResult on_result_;
    TcpServer* server_ = nullptr;
    std::unordered_map<SOCKET, int> workers_;
    std::queue<NetJob> jobs_;
    std::mutex mtx_;
    void on_message(const std::string& msg, SOCKET sock);
    void send_next(SOCKET sock);
};
}
