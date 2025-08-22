/*******************************************************************************
 * Copyright (C) 2018-2024 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "gstgvaaudiotranscribe.h"
#include <gst/audio/audio.h>
#include <gst/gst.h>
#include <openvino/genai/whisper_pipeline.hpp>
#include <dlstreamer/gst/metadata/gva_audio_event_meta.h>
#include <string>
#include <vector>
#include <mutex>

#define ELEMENT_LONG_NAME "Audio transcription based on Whisper model"
#define ELEMENT_DESCRIPTION "Performs speech recognition using OpenVINO Whisper model."
#define SAMPLE_RATE 16000

GST_DEBUG_CATEGORY_STATIC(gva_audio_transcribe_debug_category);
#define GST_CAT_DEFAULT gva_audio_transcribe_debug_category
#define GST_AUDIO_TRANSCRIBE_THRESHOLD_SEC 3

enum {
    PROP_0,
    PROP_MODEL_PATH,
    PROP_DEVICE,
    PROP_LANGUAGE,
    PROP_TASK,
    PROP_RETURN_TIMESTAMPS
};

static GstStaticPadTemplate sink_factory =
    GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
                            GST_STATIC_CAPS("audio/x-raw, "
                                           "format=(string)S16LE, "
                                           "rate=(int)16000, "
                                           "channels=(int)1"));

static GstStaticPadTemplate src_factory =
    GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS,
                            GST_STATIC_CAPS("audio/x-raw, "
                                           "format=(string)S16LE, "
                                           "rate=(int)16000, "
                                           "channels=(int)1"));

G_DEFINE_TYPE_WITH_CODE(GvaAudioTranscribe, gst_gva_audio_transcribe, GST_TYPE_BASE_TRANSFORM,
                        GST_DEBUG_CATEGORY_INIT(gva_audio_transcribe_debug_category, "gvaaudiotranscribe", 0,
                                                "debug category for gvaaudiotranscribe element"));

// Forward declarations of methods
static void gst_gva_audio_transcribe_set_property(GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec);
static void gst_gva_audio_transcribe_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec);
static void gst_gva_audio_transcribe_finalize(GObject *object);
static gboolean gst_gva_audio_transcribe_start(GstBaseTransform *base);
static gboolean gst_gva_audio_transcribe_stop(GstBaseTransform *base);
static GstFlowReturn gst_gva_audio_transcribe_transform_ip(GstBaseTransform *base, GstBuffer *buf);

void gst_gva_audio_transcribe_class_init(GvaAudioTranscribeClass *gvaaudiotranscribe_class) {
    GstElementClass *element_class = GST_ELEMENT_CLASS(gvaaudiotranscribe_class);
    GObjectClass *gobject_class = G_OBJECT_CLASS(gvaaudiotranscribe_class);
    GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS(gvaaudiotranscribe_class);

    // Set virtual methods
    gobject_class->set_property = gst_gva_audio_transcribe_set_property;
    gobject_class->get_property = gst_gva_audio_transcribe_get_property;
    gobject_class->finalize = gst_gva_audio_transcribe_finalize;

    base_transform_class->start = GST_DEBUG_FUNCPTR(gst_gva_audio_transcribe_start);
    base_transform_class->stop = GST_DEBUG_FUNCPTR(gst_gva_audio_transcribe_stop);
    base_transform_class->transform_ip = GST_DEBUG_FUNCPTR(gst_gva_audio_transcribe_transform_ip);

    // Install properties
    g_object_class_install_property(
        gobject_class, PROP_MODEL_PATH,
        g_param_spec_string("model", "Model", "Path to the Whisper model", NULL, G_PARAM_READWRITE));

    g_object_class_install_property(
        gobject_class, PROP_DEVICE,
        g_param_spec_string("device", "Device", "Device to use for inference (CPU, GPU)", "CPU", G_PARAM_READWRITE));

    g_object_class_install_property(
        gobject_class, PROP_LANGUAGE,
        g_param_spec_string("language", "Language", "Language code (e.g., <|en|>)", "<|en|>", G_PARAM_READWRITE));

    g_object_class_install_property(
        gobject_class, PROP_TASK,
        g_param_spec_string("task", "Task", "Task: 'transcribe' or 'translate'", "transcribe", G_PARAM_READWRITE));

    g_object_class_install_property(
        gobject_class, PROP_RETURN_TIMESTAMPS,
        g_param_spec_boolean("return-timestamps", "Return Timestamps",
                            "Whether to return timestamps with transcription", FALSE, G_PARAM_READWRITE));

    // Setup pad templates
    gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
    gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

    gst_element_class_set_static_metadata(element_class, ELEMENT_LONG_NAME, "Audio Transcription",
                                          ELEMENT_DESCRIPTION, "Intel Corporation");
}

void gst_gva_audio_transcribe_init(GvaAudioTranscribe *gvaaudiotranscribe) {
    GST_DEBUG_OBJECT(gvaaudiotranscribe, "gst_gva_audio_transcribe_init");

    // Initialize properties with default values
    gvaaudiotranscribe->model_path = NULL;
    gvaaudiotranscribe->device = g_strdup("CPU");
    gvaaudiotranscribe->language = g_strdup("<|en|>");
    gvaaudiotranscribe->task = g_strdup("transcribe");
    gvaaudiotranscribe->return_timestamps = FALSE;

    // Initialize internal state
    gvaaudiotranscribe->pipeline = NULL;
    gvaaudiotranscribe->config = NULL;
    gvaaudiotranscribe->audio_data = new std::vector<float>();
    gvaaudiotranscribe->mutex = new std::mutex();

    GST_DEBUG_OBJECT(gvaaudiotranscribe, "Element initialized");

    GST_DEBUG_OBJECT(gvaaudiotranscribe, "Initialized gvaaudiotranscribe");
}

static void gst_gva_audio_transcribe_set_property(GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec) {
    GvaAudioTranscribe *gvaaudiotranscribe = GVA_AUDIO_TRANSCRIBE(object);

    switch (prop_id) {
    case PROP_MODEL_PATH:
        g_free(gvaaudiotranscribe->model_path);
        gvaaudiotranscribe->model_path = g_value_dup_string(value);
        break;
    case PROP_DEVICE:
        g_free(gvaaudiotranscribe->device);
        gvaaudiotranscribe->device = g_value_dup_string(value);
        break;
    case PROP_LANGUAGE:
        g_free(gvaaudiotranscribe->language);
        gvaaudiotranscribe->language = g_value_dup_string(value);
        break;
    case PROP_TASK:
        g_free(gvaaudiotranscribe->task);
        gvaaudiotranscribe->task = g_value_dup_string(value);
        break;
    case PROP_RETURN_TIMESTAMPS:
        gvaaudiotranscribe->return_timestamps = g_value_get_boolean(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void gst_gva_audio_transcribe_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec) {
    GvaAudioTranscribe *gvaaudiotranscribe = GVA_AUDIO_TRANSCRIBE(object);

    switch (prop_id) {
    case PROP_MODEL_PATH:
        g_value_set_string(value, gvaaudiotranscribe->model_path);
        break;
    case PROP_DEVICE:
        g_value_set_string(value, gvaaudiotranscribe->device);
        break;
    case PROP_LANGUAGE:
        g_value_set_string(value, gvaaudiotranscribe->language);
        break;
    case PROP_TASK:
        g_value_set_string(value, gvaaudiotranscribe->task);
        break;
    case PROP_RETURN_TIMESTAMPS:
        g_value_set_boolean(value, gvaaudiotranscribe->return_timestamps);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void gst_gva_audio_transcribe_finalize(GObject *object) {
    GvaAudioTranscribe *gvaaudiotranscribe = GVA_AUDIO_TRANSCRIBE(object);

    GST_DEBUG_OBJECT(gvaaudiotranscribe, "Finalizing");

    // Free string properties
    g_free(gvaaudiotranscribe->model_path);
    g_free(gvaaudiotranscribe->device);
    g_free(gvaaudiotranscribe->language);
    g_free(gvaaudiotranscribe->task);

    // Delete C++ objects
    delete static_cast<std::vector<float>*>(gvaaudiotranscribe->audio_data);
    delete static_cast<std::mutex*>(gvaaudiotranscribe->mutex);

    // Call parent finalize
    G_OBJECT_CLASS(gst_gva_audio_transcribe_parent_class)->finalize(object);
}

static gboolean gst_gva_audio_transcribe_start(GstBaseTransform *base) {
    GvaAudioTranscribe *gvaaudiotranscribe = GVA_AUDIO_TRANSCRIBE(base);

    if (!gvaaudiotranscribe->model_path) {
        GST_ERROR_OBJECT(gvaaudiotranscribe, "Model path not specified");
        return FALSE;
    }

    // Create and initialize Whisper pipeline
    try {
        std::string model_path(gvaaudiotranscribe->model_path);
        std::string device(gvaaudiotranscribe->device);

        GST_INFO_OBJECT(gvaaudiotranscribe, "Creating Whisper pipeline with model: %s on device: %s",
                        model_path.c_str(), device.c_str());

        auto *pipeline = new ov::genai::WhisperPipeline(model_path, device);
        gvaaudiotranscribe->pipeline = pipeline;

        GST_DEBUG_OBJECT(gvaaudiotranscribe, "Pipeline created, setting up configuration");

        auto *config = new ov::genai::WhisperGenerationConfig();
        *config = pipeline->get_generation_config();
        config->language = gvaaudiotranscribe->language;
        config->task = gvaaudiotranscribe->task;
        config->return_timestamps = gvaaudiotranscribe->return_timestamps;
        gvaaudiotranscribe->config = config;

        GST_INFO_OBJECT(gvaaudiotranscribe, "Whisper pipeline initialized successfully (language: %s, task: %s)",
                       gvaaudiotranscribe->language, gvaaudiotranscribe->task);

        return TRUE;
    } catch (const std::exception &e) {
        GST_ERROR_OBJECT(gvaaudiotranscribe, "Failed to initialize Whisper pipeline: %s", e.what());
        return FALSE;
    }
}

static gboolean gst_gva_audio_transcribe_stop(GstBaseTransform *base) {
    GvaAudioTranscribe *gvaaudiotranscribe = GVA_AUDIO_TRANSCRIBE(base);

    GST_DEBUG_OBJECT(gvaaudiotranscribe, "Stopping element");

    if (gvaaudiotranscribe->pipeline) {
        delete static_cast<ov::genai::WhisperPipeline *>(gvaaudiotranscribe->pipeline);
        gvaaudiotranscribe->pipeline = NULL;
    }

    if (gvaaudiotranscribe->config) {
        delete static_cast<ov::genai::WhisperGenerationConfig *>(gvaaudiotranscribe->config);
        gvaaudiotranscribe->config = NULL;
    }

    auto *audio_data = static_cast<std::vector<float> *>(gvaaudiotranscribe->audio_data);
    audio_data->clear();

    GST_DEBUG_OBJECT(gvaaudiotranscribe, "Element stopped successfully");
    return TRUE;
}

static GstFlowReturn gst_gva_audio_transcribe_transform_ip(GstBaseTransform *base, GstBuffer *buf) {
    GvaAudioTranscribe *gvaaudiotranscribe = GVA_AUDIO_TRANSCRIBE(base);
    auto *pipeline = static_cast<ov::genai::WhisperPipeline *>(gvaaudiotranscribe->pipeline);
    auto *config = static_cast<ov::genai::WhisperGenerationConfig *>(gvaaudiotranscribe->config);
    auto *audio_data = static_cast<std::vector<float> *>(gvaaudiotranscribe->audio_data);
    auto *mutex = static_cast<std::mutex *>(gvaaudiotranscribe->mutex);

    if (!pipeline || !config) {
        GST_ERROR_OBJECT(gvaaudiotranscribe, "Pipeline or config not initialized");
        return GST_FLOW_ERROR;
    }

    GstMapInfo map;
    if (!gst_buffer_map(buf, &map, GST_MAP_READ)) {
        GST_ERROR_OBJECT(gvaaudiotranscribe, "Failed to map buffer");
        return GST_FLOW_ERROR;
    }

    // Convert PCM data to float
    const int16_t *pcm_data = reinterpret_cast<const int16_t *>(map.data);
    size_t num_samples = map.size / sizeof(int16_t);

    // Add samples to buffer with proper locking
    {
        std::lock_guard<std::mutex> lock(*mutex);
        const size_t previous_size = audio_data->size();
        audio_data->reserve(previous_size + num_samples);

        for (size_t i = 0; i < num_samples; ++i) {
            audio_data->push_back(static_cast<float>(pcm_data[i]) / 32768.0f);  // Normalize to [-1.0, 1.0]
        }

        GST_LOG_OBJECT(gvaaudiotranscribe, "Added %zu samples, buffer now contains %zu samples",
                       num_samples, audio_data->size());
    }

    // Process audio when we have enough samples (e.g., 3 seconds)
    const size_t threshold_samples = SAMPLE_RATE * GST_AUDIO_TRANSCRIBE_THRESHOLD_SEC;
    if (audio_data->size() > threshold_samples) {
        GST_DEBUG_OBJECT(gvaaudiotranscribe, "Reached threshold of %zu samples, starting transcription", threshold_samples);
        try {
            std::lock_guard<std::mutex> lock(*mutex);

            ov::genai::RawSpeechInput input = *audio_data;
            GST_INFO_OBJECT(gvaaudiotranscribe, "Running transcription on %zu samples", input.size());

            auto result = pipeline->generate(input, *config);

            if (result.texts.empty()) {
                GST_WARNING_OBJECT(gvaaudiotranscribe, "Transcription result is empty");
            } else {
                const std::string &transcript = result.texts[0];
                GST_INFO_OBJECT(gvaaudiotranscribe, "Transcription result: %s", transcript.c_str());

                // Get timestamps from buffer (in nanoseconds)
                GstClockTime start_time = GST_BUFFER_PTS(buf);
                GstClockTime duration = GST_BUFFER_DURATION(buf);

                // If clock not valid, use default duration
                if (!GST_CLOCK_TIME_IS_VALID(start_time))
                    start_time = 0;
                if (!GST_CLOCK_TIME_IS_VALID(duration))
                    duration = GST_SECOND * GST_AUDIO_TRANSCRIBE_THRESHOLD_SEC;

                GstClockTime end_time = start_time + duration;

                if (gst_buffer_is_writable(buf)) {
                    // Create audio event metadata with the transcription text as the event type
                    GstGVAAudioEventMeta *meta = gst_gva_buffer_add_audio_event_meta(
                        buf, transcript.c_str(), start_time, end_time);

                    if (meta) {
                        GstStructure *detection = gst_structure_new(
                            "detection",
                            "label", G_TYPE_STRING, transcript.c_str(),
                            "text", G_TYPE_STRING, transcript.c_str(),
                            "start_timestamp", G_TYPE_UINT64, start_time,
                            "end_timestamp", G_TYPE_UINT64, end_time,
                            NULL);
                        gst_gva_audio_event_meta_add_param(meta, detection);

                        GST_INFO_OBJECT(gvaaudiotranscribe, "Added transcription metadata to buffer");
                    } else {
                        GST_ERROR_OBJECT(gvaaudiotranscribe, "Failed to add audio event metadata to buffer");
                    }
                }

                // Print additional texts if available
                for (size_t i = 1; i < result.texts.size(); i++) {
                    GST_INFO_OBJECT(gvaaudiotranscribe, "Additional text %zu: %s", i, result.texts[i].c_str());
                }
            }

            // Clear buffer after processing
            GST_LOG_OBJECT(gvaaudiotranscribe, "Clearing audio buffer after transcription");
            audio_data->clear();
        } catch (const std::exception &e) {
            GST_ERROR_OBJECT(gvaaudiotranscribe, "Error during transcription: %s", e.what());
        }
    }

    gst_buffer_unmap(buf, &map);
    return GST_FLOW_OK;
}
