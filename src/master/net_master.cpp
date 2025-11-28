#include <vector>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <string>
#include <memory>
#include "networking/tcp_server.h"
#include "networking/protocol.h"
#include "master/net_master.h"

namespace dip {
static std::string make_task_payload(const NetJob& job) {
    std::string p;
    p += "type=task\n";
    p += "label=" + job.label + "\n";
    p += "id=" + job.id + "\n";
    p += "path=" + job.path + "\n";
    return p;
}
static bool parse_result(const std::string& msg, std::string& label, std::string& path, std::vector<float>& emb) {
    if (msg.find("type=result") == std::string::npos) return false;
    auto getv = [&](const std::string& key){ auto k = key + "="; auto pos = msg.find(k); if (pos==std::string::npos) return std::string(); auto end = msg.find('\n', pos); return msg.substr(pos + k.size(), end == std::string::npos ? std::string::npos : end - (pos + k.size())); };
    label = getv("label"); path = getv("path"); std::string ecsv = getv("embedding");
    emb.clear();
    size_t start = 0;
    while (start < ecsv.size()) { auto comma = ecsv.find(',', start); std::string tok = ecsv.substr(start, comma == std::string::npos ? std::string::npos : comma - start); if (!tok.empty()) emb.push_back(static_cast<float>(std::stod(tok))); if (comma == std::string::npos) break; start = comma + 1; }
    return true;
}
NetMaster::NetMaster(const std::string& bind_addr, uint16_t port, OnResult on_result) : on_result_(on_result) {
    server_ = new TcpServer(bind_addr, port, [this](const std::string& m, SOCKET s){ on_message(m,s); });
}
void NetMaster::enqueue(const NetJob& job) { std::lock_guard<std::mutex> lk(mtx_); jobs_.push(job); }
void NetMaster::run() { server_->start(); }
void NetMaster::on_message(const std::string& msg, SOCKET sock) {
    if (msg.find("type=hello") != std::string::npos) {
        workers_[sock] = 0;
        send_next(sock);
    } else if (msg.find("type=result") != std::string::npos) {
        std::string label, path; std::vector<float> emb;
        if (parse_result(msg, label, path, emb)) { on_result_(label, path, emb); }
        workers_[sock] = 0;
        OutputDebugStringA("master: result received\n");
        send_next(sock);
    }
}
void NetMaster::send_next(SOCKET sock) {
    NetJob job;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (jobs_.empty()) return;
        job = jobs_.front(); jobs_.pop();
    }
    auto framed = encode_length_prefixed(make_task_payload(job));
    ::send(sock, framed.data(), static_cast<int>(framed.size()), 0);
    // optional: lightweight log for tracing
    OutputDebugStringA("master: task sent\n");
}
}
