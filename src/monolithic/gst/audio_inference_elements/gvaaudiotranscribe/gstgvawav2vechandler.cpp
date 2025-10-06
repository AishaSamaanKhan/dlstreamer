#include "gstgvawav2vechandler.h"
#include <gst/gst.h>
#include <fstream>
#include <stdexcept>

bool WavVecHandler::initialize(const std::string &model_path, const std::string &device,
                               const std::string &language, const std::string &task,
                               bool return_timestamps) {
    (void)language; (void)task; (void)return_timestamps; // currently unused

    std::ifstream xml(model_path);
    if (!xml.good()) {
        throw std::runtime_error("Model XML not found: " + model_path);
    }
    std::string bin_path = model_path;
    auto pos = bin_path.rfind('.');
    if (pos != std::string::npos) {
        bin_path.replace(pos, std::string::npos, ".bin");
        std::ifstream bin(bin_path);
        if (!bin.good()) {
            GST_WARNING("Associated BIN file not found (continuing): %s", bin_path.c_str());
        }
    }

    core = std::make_shared<ov::Core>();
    auto model = core->read_model(model_path);
    try {
        auto inputs = model->inputs();
        if (!inputs.empty()) {
            ov::PartialShape dyn_shape = {1, ov::Dimension::dynamic()};
            model->reshape({{inputs[0].get_any_name(), dyn_shape}});
        }
    } catch (const std::exception &e) {
        GST_WARNING("WavVecHandler: dynamic reshape skipped: %s", e.what());
    }

    compiled_model = core->compile_model(model, device);
    infer_request = compiled_model.create_infer_request();
    alphabet = {"<pad>", "<s>", "</s>", "<unk>", "|", "e","t","a","o","n","i","h","s","r",
                "d","l","u","m","w","c","f","g","y","p","b","v","k","'","x","j","q","z"};
    GST_INFO("WavVecHandler initialized: %s on %s", model_path.c_str(), device.c_str());
    return true;
}

std::string WavVecHandler::transcribe(const std::vector<float> &audio_data, GstBuffer *buf) {
    (void)buf;
    if (!infer_request) return {};
    ov::Shape input_shape = {1, audio_data.size()};
    ov::Tensor input_tensor(compiled_model.input().get_element_type(), input_shape, const_cast<float*>(audio_data.data()));
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    ov::Tensor output_tensor = infer_request.get_output_tensor();
    auto shape = output_tensor.get_shape();
    if (shape.size() < 2) return {};
    size_t time_steps = (shape.size() == 3) ? shape[1] : shape[0];
    size_t vocab_size = (shape.size() == 3) ? shape[2] : shape[1];
    float *logits = output_tensor.data<float>();
    std::vector<int64_t> token_ids(time_steps);
    for (size_t t = 0; t < time_steps; ++t) {
        float max_val = -1e9f; int64_t max_idx = 0;
        for (size_t v = 0; v < vocab_size; ++v) {
            float val = logits[t * vocab_size + v];
            if (val > max_val) { max_val = val; max_idx = v; }
        }
        token_ids[t] = max_idx;
    }
    std::string transcription; int64_t prev = -1;
    for (auto id : token_ids) {
        if (id == 0 || id == prev) { prev = id; continue; }
        std::string token = (id < (int64_t)alphabet.size()) ? alphabet[id] : "";
        if (token == "|") token = " ";
        transcription += token; prev = id;
    }
    GST_INFO("WavVecHandler decoded: %s", transcription.c_str());
    return transcription;
}

void WavVecHandler::cleanup() {
    infer_request = {};
    compiled_model = {};
    core.reset();
}
