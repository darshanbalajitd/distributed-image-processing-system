#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <thread>
#include <chrono>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <fstream>
#include "networking/tcp_client.h"
#include "inference/onnx_backend.h"

int main(int argc, char** argv) {
    std::string master_host = "127.0.0.1";
    uint16_t master_port = 5555;
    int gpu_workers = 0;
    int cpu_workers = 1;
    for (int i=1;i<argc;++i){
        std::string a = argv[i];
        if (a == "--master" && i+1<argc){
            std::string hp = argv[++i];
            auto pos = hp.find(":");
            if (pos!=std::string::npos){ master_host = hp.substr(0,pos); master_port = static_cast<uint16_t>(std::stoi(hp.substr(pos+1))); }
        }
        else if (a == "--gpu-workers" && i+1<argc){ gpu_workers = std::stoi(argv[++i]); }
        else if (a == "--cpu-workers" && i+1<argc){ cpu_workers = std::stoi(argv[++i]); }
    }
    auto start_worker = [&](bool prefer_cuda, int idx){
        std::thread([&, prefer_cuda, idx]{
            dip::TcpClient client;
            if (!client.connect(master_host, master_port)) return;
            std::string wid = std::string(prefer_cuda?"gpu-":"cpu-") + std::to_string(idx);
            std::string hello = std::string("type=hello\nworker_id=") + wid + std::string("\nprovider=") + (prefer_cuda?"cuda":"cpu") + std::string("\n");
            client.send(hello);
            std::unique_ptr<dip::IInferenceBackend> backend(new dip::OnnxRuntimeBackend(prefer_cuda ? dip::ProviderPref::CUDA : dip::ProviderPref::CPU));
            std::string model = (std::string("models/") + "vggface2_resnet50.onnx");
            bool ok = backend->init(model);
            std::cout << "worker " << wid << " initialized provider=" << (prefer_cuda && ok?"cuda":"cpu") << std::endl;
            if (!ok && prefer_cuda) { backend.reset(new dip::OnnxRuntimeBackend(dip::ProviderPref::CPU)); backend->init(model); std::cout << "worker " << wid << " fallback provider=cpu" << std::endl; }
            while (true) {
                std::string msg;
                if (!client.read(msg)) break;
                if (msg.find("type=task") != std::string::npos) {
                    auto getv = [&](const std::string& key){ auto k = key + "="; auto pos = msg.find(k); if (pos==std::string::npos) return std::string(); auto end = msg.find('\n', pos); return msg.substr(pos + k.size(), end == std::string::npos ? std::string::npos : end - (pos + k.size())); };
                    auto label = getv("label");
                    auto path = getv("path");
                    auto id = getv("id");
                    std::vector<unsigned char> bytes;
                    std::ifstream f(path, std::ios::binary);
                    if (f.good()) {
                        f.seekg(0, std::ios::end);
                        std::streampos szpos = f.tellg();
                        size_t sz = szpos > 0 ? static_cast<size_t>(szpos) : 0;
                        f.seekg(0, std::ios::beg);
                        bytes.resize(sz);
                        if (sz > 0) f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(sz));
                    }
                    auto res = backend->infer(bytes);
                    if (res) {
                        std::ostringstream oss;
                        oss << "type=result\nlabel=" << label << "\npath=" << path << "\nid=" << id << "\nembedding=";
                        for (size_t i=0;i<res->embedding.size();++i){ if (i) oss << ","; oss << res->embedding[i]; }
                        client.send(oss.str());
                    }
                }
            }
        }).detach();
    };
    for (int i=0;i<gpu_workers;++i) start_worker(true, i);
    for (int i=0;i<cpu_workers;++i) start_worker(false, i);
    std::this_thread::sleep_for(std::chrono::hours(24));
    return 0;
}
