
#pragma once
#include "gstgvaaudiotranscribehandler.h"
#include <openvino/genai/whisper_pipeline.hpp>

class WhisperHandler : public GvaAudioTranscribeHandler {
public:
    bool initialize(const std::string &model_path, const std::string &device,
                    const std::string &language, const std::string &task,
                    bool return_timestamps) override;

    std::string transcribe(const std::vector<float> &audio_data, GstBuffer *buf) override;

    void cleanup() override;

private:
    ov::genai::WhisperPipeline *pipeline = nullptr;
    ov::genai::WhisperGenerationConfig *config = nullptr;
};
