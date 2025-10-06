#pragma once
#include <gst/gst.h>
#include <vector>
#include <string>

class GvaAudioTranscribeHandler {
public:
    virtual ~GvaAudioTranscribeHandler() = default;

    virtual bool initialize(const std::string &model_path, const std::string &device,
                            const std::string &language, const std::string &task,
                            bool return_timestamps) = 0;

    virtual std::string transcribe(const std::vector<float> &audio_data, GstBuffer *buf) = 0;

    virtual void cleanup() = 0;
};
