#pragma once
#include <string>
#include <vector>
#include <optional>

namespace dip {
struct FaceBox { int x; int y; int w; int h; float confidence; };
struct InferenceResult {
    std::vector<FaceBox> faces;
    std::vector<float> embedding;
    std::string meta;
};

class IInferenceBackend {
public:
    virtual ~IInferenceBackend() = default;
    virtual bool init(const std::string& model_path) = 0;
    virtual std::optional<InferenceResult> infer(const std::vector<unsigned char>& image_bytes) = 0;
};
}