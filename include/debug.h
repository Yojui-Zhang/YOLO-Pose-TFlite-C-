// TensorFlow Lite 核心
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/delegates/external/external_delegate.h"
#include "tensorflow/lite/c/common.h"

static bool SaveOutputTensorToTxt(tflite::Interpreter* interp,
                                  int output_index,
                                  const std::string& path) {
    if (!interp) return false;
    const auto& outs = interp->outputs();
    if (output_index < 0 || output_index >= static_cast<int>(outs.size())) return false;

    int tensor_idx = outs[output_index];
    TfLiteTensor* t = interp->tensor(tensor_idx);
    if (!t || !t->dims) return false;

    std::ofstream ofs(path);
    if (!ofs.is_open()) return false;

    // 寫入 dtype 與 shape
    ofs << "# dtype: ";
    switch (t->type) {
        case kTfLiteFloat32: ofs << "float32\n"; break;
        case kTfLiteInt8:    ofs << "int8 (dequantized to float32)\n"; break;
        default:             ofs << "unsupported\n"; break;
    }

    ofs << "# shape: [";
    for (int i = 0; i < t->dims->size; ++i) {
        ofs << t->dims->data[i];
        if (i + 1 < t->dims->size) ofs << ", ";
    }
    ofs << "]\n";

    // 元素總數
    int64_t count = 1;
    for (int i = 0; i < t->dims->size; ++i) count *= t->dims->data[i];

    ofs << std::fixed << std::setprecision(6);

    if (t->type == kTfLiteFloat32 && t->data.f) {
        const float* p = t->data.f;
        for (int64_t i = 0; i < count; ++i) ofs << p[i] << '\n';
        return true;
    } else if (t->type == kTfLiteInt8 && t->data.int8) {
        const int8_t* q = t->data.int8;
        const float scale = t->params.scale;
        const int32_t zp   = t->params.zero_point;
        for (int64_t i = 0; i < count; ++i) {
            float f = (static_cast<int32_t>(q[i]) - zp) * scale; // 反量化
            ofs << f << '\n';
        }
        return true;
    }

    // 其他 dtype 未支援
    return false;
}
