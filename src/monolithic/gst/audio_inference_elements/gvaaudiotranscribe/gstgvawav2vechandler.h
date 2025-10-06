#pragma once
#include "gstgvaaudiotranscribehandler.h"
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

class WavVecHandler : public GvaAudioTranscribeHandler {
public:
    bool initialize(const std::string &model_path, const std::string &device,
                    const std::string &language, const std::string &task,
                    bool return_timestamps) override;

    std::string transcribe(const std::vector<float> &audio_data, GstBuffer *buf) override;

    void cleanup() override;

private:
    std::shared_ptr<ov::Core> core;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    std::vector<std::string> alphabet; // for greedy CTC decoding
};
