#pragma once
#include <string>
#include <vector>
#include <optional>
#include <memory>
#include "inference/backend.h"

namespace dip {
#if defined(DIP_HAS_ONNX)
enum class ProviderPref { CPU, CUDA };
class OnnxRuntimeBackend : public IInferenceBackend {
public:
    explicit OnnxRuntimeBackend(ProviderPref pref);
    ~OnnxRuntimeBackend() override;
    bool init(const std::string& model_path) override;
    std::optional<InferenceResult> infer(const std::vector<unsigned char>& image_bytes) override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
#endif
}
