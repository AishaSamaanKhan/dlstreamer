#include "gstgvawhisperasrhandler.h"
#include <gst/gst.h>

bool WhisperHandler::initialize(const std::string &model_path, const std::string &device,
                                const std::string &language, const std::string &task,
                                bool return_timestamps) {
    try {
        pipeline = new ov::genai::WhisperPipeline(model_path, device);
        config = new ov::genai::WhisperGenerationConfig();
        *config = pipeline->get_generation_config();
        config->language = language;
        config->task = task;
        config->return_timestamps = return_timestamps;
        GST_INFO("WhisperHandler initialized (language=%s task=%s)", language.c_str(), task.c_str());
        return true;
    } catch (const std::exception &e) {
        GST_ERROR("WhisperHandler init failed: %s", e.what());
        return false;
    }
}

std::string WhisperHandler::transcribe(const std::vector<float> &audio_data, GstBuffer * /*buf*/) {
    if (!pipeline || !config) return {};
    ov::genai::RawSpeechInput input = audio_data;
    auto result = pipeline->generate(input, *config);
    if (result.texts.empty()) return {};
    return result.texts[0];
}

void WhisperHandler::cleanup() {
    if (pipeline) { delete pipeline; pipeline = nullptr; }
    if (config) { delete config; config = nullptr; }
}
