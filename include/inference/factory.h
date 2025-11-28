#pragma once
#include <memory>
#include "inference/backend.h"

namespace dip {
#if defined(DIP_HAS_ONNX)
std::unique_ptr<IInferenceBackend> make_onnx_backend();
#endif
}
