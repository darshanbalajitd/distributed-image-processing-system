#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cctype>
#include <sstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <iomanip>
#include "common/csv_writer.h"
#include "inference/factory.h"
#if defined(DIP_HAS_ONNX)
#include "inference/onnx_backend.h"
#endif
#if defined(DIP_HAS_NETWORKING)
#include "master/net_master.h"
#include "networking/tcp_client.h"
#endif

namespace fs = std::filesystem;
using namespace dip;

static std::vector<unsigned char> read_file_bytes(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<unsigned char> bytes;
    f.seekg(0, std::ios::end);
    std::streampos size = f.tellg();
    f.seekg(0, std::ios::beg);
    bytes.resize(static_cast<size_t>(size));
    if (size > 0) f.read(reinterpret_cast<char*>(bytes.data()), size);
    return bytes;
}

int main(int argc, char** argv) {
    fs::path image_root = fs::path("data") / "images";
    fs::path output_dir = fs::path("output");
    fs::create_directories(output_dir);
    CsvWriter csv((output_dir / "embeddings.csv").string());
    bool net_mode = false;
    std::string bind = "0.0.0.0";
    uint16_t port = 5555;

    struct Job { std::string label; fs::path path; std::vector<unsigned char> bytes; };
    struct Result { std::string label; fs::path path; std::vector<float> embedding; };
    std::queue<Job> jobs;
    std::mutex mtx;
    std::condition_variable cvq;
    std::atomic<bool> done{false};
    std::atomic<size_t> processed{0};
    size_t total = 0;

    int gpu_workers = 0;
    int cpu_workers = 1;
    int local_gpu_workers = 0;
    int local_cpu_workers = 0;
    for (int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--gpu-workers" && i+1 < argc) { gpu_workers = std::stoi(argv[++i]); }
        else if (arg == "--cpu-workers" && i+1 < argc) { cpu_workers = std::stoi(argv[++i]); }
        else if (arg == "--mode" && i+1 < argc) { std::string m = argv[++i]; net_mode = (m == "master"); }
        else if (arg == "--bind" && i+1 < argc) { bind = argv[++i]; }
        else if (arg == "--port" && i+1 < argc) { port = static_cast<uint16_t>(std::stoi(argv[++i])); }
        else if (arg == "--local-gpu-workers" && i+1 < argc) { local_gpu_workers = std::stoi(argv[++i]); }
        else if (arg == "--local-cpu-workers" && i+1 < argc) { local_cpu_workers = std::stoi(argv[++i]); }
    }

    std::queue<Result> results_q;
    std::mutex results_mtx;
    std::condition_variable cv_results;
    bool writer_done = false;

    fs::path model_path = fs::path("models") / "vggface2_resnet50.onnx";

    auto producer = std::thread([&](){
        for (auto& dir : fs::directory_iterator(image_root)) {
            if (!dir.is_directory()) continue;
            std::string dirname = dir.path().filename().string();
            std::string label = dirname;
            if (label.rfind("pin_", 0) == 0) label = label.substr(4);
            for (auto& entry : fs::recursive_directory_iterator(dir.path())) {
                if (!entry.is_regular_file()) continue;
                auto ext = entry.path().extension().string();
                for (auto& c : ext) c = std::tolower(c);
                if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;
                auto bytes = read_file_bytes(entry.path());
                if (!net_mode) {
                    {
                        std::lock_guard<std::mutex> lk(mtx);
                        jobs.push(Job{label, entry.path(), std::move(bytes)});
                        total++;
                    }
                    cvq.notify_one();
                }
            }
        }
        done = true;
        cvq.notify_all();
    });

    auto worker_fn = [&](bool use_cuda){
        std::unique_ptr<IInferenceBackend> backend;
#if defined(DIP_HAS_ONNX)
        backend.reset(new OnnxRuntimeBackend(use_cuda ? ProviderPref::CUDA : ProviderPref::CPU));
        if (!backend->init(model_path.string())) {
            std::cerr << "ERROR: ONNX backend init failed for provider=" << (use_cuda?"cuda":"cpu") << std::endl;
            return;
        }
#else
        std::cerr << "ERROR: Built without ONNX Runtime. Reconfigure with USE_ONNXRUNTIME=ON." << std::endl;
        return;
#endif
        while (true) {
            Job job;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cvq.wait(lk, [&]{ return !jobs.empty() || done.load(); });
                if (jobs.empty()) {
                    if (done.load()) break;
                    else continue;
                }
                job = std::move(jobs.front());
                jobs.pop();
            }
            auto res = backend->infer(job.bytes);
            if (res && !res->embedding.empty()) {
                {
                    std::lock_guard<std::mutex> g(results_mtx);
                    results_q.push(Result{job.label, job.path, res->embedding});
                }
                cv_results.notify_one();
            }
            processed++;
        }
    };

    std::vector<std::thread> threads;
    if (!net_mode) {
        for (int i=0; i<gpu_workers; ++i) threads.emplace_back(worker_fn, true);
        for (int i=0; i<cpu_workers; ++i) threads.emplace_back(worker_fn, false);
    }

    auto progress_thr = std::thread([&]{
        while (!done.load() || processed.load() < total) {
            size_t p = processed.load();
            size_t t = total;
            double pct = t ? (100.0 * double(p) / double(t)) : 0.0;
            std::cout << "progress " << p << "/" << t << " (" << std::fixed << std::setprecision(1) << pct << "%)" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });

    const size_t target_dim = 512;
    bool header_written = false;
    auto writer_thr = std::thread([&]{
        while (true) {
            std::unique_lock<std::mutex> lk(results_mtx);
            cv_results.wait(lk, [&]{ return !results_q.empty() || (done.load() && writer_done); });
            if (results_q.empty()) {
                if (done.load() && writer_done) break;
                continue;
            }
            auto r = std::move(results_q.front());
            results_q.pop();
            lk.unlock();
            if (!header_written) {
                std::vector<std::string> header;
                header.push_back("label");
                header.push_back("path");
                for (size_t i = 0; i < target_dim; ++i) header.push_back(std::string("e") + std::to_string(i));
                csv.write_header(header);
                header_written = true;
            }
            std::vector<float> vec = r.embedding;
            if (vec.size() > target_dim) vec.resize(target_dim);
            if (vec.size() < target_dim) vec.insert(vec.end(), target_dim - vec.size(), 0.0f);
            std::vector<std::string> row;
            row.push_back(r.label);
            row.push_back(r.path.string());
            row.reserve(2 + target_dim);
            for (auto v : vec) { std::ostringstream oss; oss.precision(6); oss << std::fixed << v; row.push_back(oss.str()); }
            csv.write_row(row);
        }
    });

    if (net_mode) {
        dip::NetMaster nm(bind, port, [&](const std::string& label, const std::string& path, const std::vector<float>& emb){
            {
                std::lock_guard<std::mutex> g(results_mtx);
                results_q.push(Result{label, fs::path(path), emb});
            }
            cv_results.notify_one();
            processed++;
        });
        std::thread server_thr([&]{ nm.run(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        auto start_local_worker = [&](bool prefer_cuda, int idx){
            std::thread([&, prefer_cuda, idx]{
                dip::TcpClient c;
                if (!c.connect("127.0.0.1", port)) return;
                std::string wid = std::string(prefer_cuda?"local-gpu-":"local-cpu-") + std::to_string(idx);
                std::string hello = std::string("type=hello\nworker_id=") + wid + std::string("\nprovider=") + (prefer_cuda?"cuda":"cpu") + std::string("\n");
                c.send(hello);
                std::unique_ptr<dip::IInferenceBackend> be(new dip::OnnxRuntimeBackend(prefer_cuda ? dip::ProviderPref::CUDA : dip::ProviderPref::CPU));
                std::string model = (std::string("models/") + "vggface2_resnet50.onnx");
                bool ok = be->init(model);
                std::cout << "local " << wid << " initialized provider=" << (prefer_cuda && ok?"cuda":"cpu") << std::endl;
                if (!ok && prefer_cuda) { be.reset(new dip::OnnxRuntimeBackend(dip::ProviderPref::CPU)); be->init(model); std::cout << "local " << wid << " fallback provider=cpu" << std::endl; }
                while (true) {
                    std::string msg;
                    if (!c.read(msg)) break;
                    if (msg.find("type=task") != std::string::npos) {
                        auto getv = [&](const std::string& key){ auto k = key + "="; auto pos = msg.find(k); if (pos==std::string::npos) return std::string(); auto end = msg.find('\n', pos); return msg.substr(pos + k.size(), end == std::string::npos ? std::string::npos : end - (pos + k.size())); };
                        auto base64 = getv("base64");
                        auto label = getv("label");
                        auto path = getv("path");
                        auto id = getv("id");
                        std::vector<unsigned char> bytes(base64.begin(), base64.end());
                        auto res = be->infer(bytes);
                        if (res) {
                            std::ostringstream oss;
                            oss << "type=result\nlabel=" << label << "\npath=" << path << "\nid=" << id << "\nembedding=";
                            for (size_t i=0;i<res->embedding.size();++i){ if (i) oss << ","; oss << res->embedding[i]; }
                            c.send(oss.str());
                        }
                    }
                }
            }).detach();
        };
        int lg = local_gpu_workers ? local_gpu_workers : gpu_workers;
        int lc = local_cpu_workers ? local_cpu_workers : cpu_workers;
        for (int i=0;i<lg;++i) start_local_worker(true, i);
        for (int i=0;i<lc;++i) start_local_worker(false, i);
        for (auto& dir : fs::directory_iterator(image_root)) {
            if (!dir.is_directory()) continue;
            std::string dirname = dir.path().filename().string();
            std::string label = dirname;
            if (label.rfind("pin_", 0) == 0) label = label.substr(4);
            for (auto& entry : fs::recursive_directory_iterator(dir.path())) {
                if (!entry.is_regular_file()) continue;
                auto ext = entry.path().extension().string();
                for (auto& c : ext) c = std::tolower(c);
                if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;
                auto bytes = read_file_bytes(entry.path());
                std::string b64(bytes.begin(), bytes.end());
                dip::NetJob nj{label, entry.path().string(), b64, entry.path().string()};
                nm.enqueue(nj);
                total++;
            }
        }
        server_thr.join();
        done = true;
    }
    producer.join();
    for (auto& th : threads) th.join();
    progress_thr.join();
    writer_done = true;
    cv_results.notify_all();
    writer_thr.join();
    std::cout << "Processed " << processed.load() << " images. CSV: " << (output_dir / "embeddings.csv").string() << std::endl;
    return 0;
}
