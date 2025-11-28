#include <string>
#include <vector>
#include <optional>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <array>
#include <filesystem>
#if defined(DIP_HAS_ONNX)
#include "onnxruntime_cxx_api.h"
#endif
#if defined(DIP_HAS_OPENCV)
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif
#include "inference/onnx_backend.h"

namespace dip {
#if defined(DIP_HAS_ONNX)
struct OnnxRuntimeBackend::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "dip"};
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> session;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    bool use_cuda = false;
};

OnnxRuntimeBackend::OnnxRuntimeBackend(ProviderPref pref) : impl(new Impl{}) { impl->use_cuda = (pref == ProviderPref::CUDA); }
OnnxRuntimeBackend::~OnnxRuntimeBackend() {}

static void to_nhwc_float_rgb(
#if defined(DIP_HAS_OPENCV)
    cv::Mat& img,
#else
    std::vector<unsigned char>& /*img*/,
#endif
    std::vector<float>& out) {
#if defined(DIP_HAS_OPENCV)
    cv::Mat resized; cv::resize(img, resized, cv::Size(224,224));
    cv::Mat rgb; cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    out.resize(224*224*3);
    size_t idx=0;
    for (int y=0; y<224; ++y) {
        for (int x=0; x<224; ++x) {
            cv::Vec3b v = rgb.at<cv::Vec3b>(y,x);
            out[idx++] = float(v[0]);
            out[idx++] = float(v[1]);
            out[idx++] = float(v[2]);
        }
    }
#else
    out.assign(224*224*3, 0.0f);
#endif
}

bool OnnxRuntimeBackend::init(const std::string& model_path) {
    try {
        if (impl->use_cuda) {
            try { OrtSessionOptionsAppendExecutionProvider_CUDA(impl->opts, 0); } catch (...) { impl->use_cuda = false; }
        }
        std::wstring wpath = std::filesystem::path(model_path).wstring();
        impl->session.reset(new Ort::Session(impl->env, wpath.c_str(), impl->opts));
        size_t n_in = impl->session->GetInputCount();
        size_t n_out = impl->session->GetOutputCount();
        impl->input_names.resize(n_in);
        impl->output_names.resize(n_out);
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i=0;i<n_in;++i) impl->input_names[i] = impl->session->GetInputNameAllocated(i, allocator).release();
        for (size_t i=0;i<n_out;++i) impl->output_names[i] = impl->session->GetOutputNameAllocated(i, allocator).release();
        return true;
    } catch (...) { return false; }
}

std::optional<InferenceResult> OnnxRuntimeBackend::infer(const std::vector<unsigned char>& image_bytes) {
#if defined(DIP_HAS_OPENCV)
    cv::Mat buf(1, static_cast<int>(image_bytes.size()), CV_8UC1, const_cast<unsigned char*>(image_bytes.data()));
    cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
    if (img.empty()) return {};
#endif
    std::vector<float> tensor;
#if defined(DIP_HAS_OPENCV)
    to_nhwc_float_rgb(img, tensor);
#else
    std::vector<unsigned char> dummy;
    to_nhwc_float_rgb(dummy, tensor);
#endif

    std::array<int64_t,4> shape{1,224,224,3};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(mem, tensor.data(), tensor.size(), shape.data(), shape.size());
    auto outputs = impl->session->Run(Ort::RunOptions{nullptr}, impl->input_names.data(), &input, 1, impl->output_names.data(), impl->output_names.size());
    InferenceResult r;
    r.faces.push_back({0,0,
#if defined(DIP_HAS_OPENCV)
        img.cols, img.rows,
#else
        224, 224,
#endif
        1.0f});
    if (!outputs.empty() && outputs[0].IsTensor()) {
        float* p = outputs[0].GetTensorMutableData<float>();
        auto type = outputs[0].GetTensorTypeAndShapeInfo();
        auto sz = type.GetElementCount();
        r.embedding.assign(p, p + sz);
    }
    r.meta = impl->use_cuda ? std::string("provider=cuda") : std::string("provider=cpu");
    return r;
}
#endif
}
