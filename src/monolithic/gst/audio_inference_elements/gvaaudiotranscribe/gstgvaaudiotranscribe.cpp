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
#include <ctc_decode.h>
#include <nlohmann/json.hpp>
#include <fstream>


#define ELEMENT_LONG_NAME "Audio transcription based on wav 2 vec model model"
#define ELEMENT_DESCRIPTION "Performs speech recognition using OpenVINO wav2vec model."
#define SAMPLE_RATE 16000

GST_DEBUG_CATEGORY_STATIC(gva_audio_transcribe_debug_category);
#define GST_CAT_DEFAULT gva_audio_transcribe_debug_category
#define GST_AUDIO_TRANSCRIBE_THRESHOLD_SEC 3

enum {
    PROP_0,
    PROP_MODEL_PATH,
    PROP_DEVICE,
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
        g_param_spec_string("model", "Model", "Path to the wav2vec model", NULL, G_PARAM_READWRITE));

    g_object_class_install_property(
        gobject_class, PROP_DEVICE,
        g_param_spec_string("device", "Device", "Device to use for inference (CPU, GPU)", "CPU", G_PARAM_READWRITE));

    
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
    // Initialize internal state
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

    // Create and initialize model loading
    try {
        std::string model_path(gvaaudiotranscribe->model_path);
        std::string device(gvaaudiotranscribe->device);

        GST_INFO_OBJECT(gvaaudiotranscribe, "Here is the model: %s on device: %s",
                        model_path.c_str(), device.c_str());

        gvaaudiotranscribe->core = std::make_shared<ov::Core>();
        auto model = gvaaudiotranscribe->core->read_model(gvaaudiotranscribe->model_path);
        GST_INFO_OBJECT(gvaaudiotranscribe, "Model successfully read");
        
        try {
            auto inputs = model->inputs();
            if (!inputs.empty()) {
                ov::PartialShape dyn_shape = {1, ov::Dimension::dynamic()};
                model->reshape({{inputs[0].get_any_name(), dyn_shape}});
                GST_INFO_OBJECT(gvaaudiotranscribe, "Reshaped model input to {1, -1}");
            }
        } catch (const std::exception &e) {
            GST_WARNING_OBJECT(gvaaudiotranscribe, "Could not reshape model dynamically: %s", e.what());
        }

        gvaaudiotranscribe->compiled_model = gvaaudiotranscribe->core->compile_model(model, device.c_str());
        gvaaudiotranscribe->infer_request = gvaaudiotranscribe->compiled_model.create_infer_request();
        GST_INFO_OBJECT(gvaaudiotranscribe, "Model loaded and compiled successfully: %s on %s",
                        model_path.c_str(), device.c_str());

        


        //auto *pipeline = new ov::genai::WhisperPipeline(model_path, device);
        //gvaaudiotranscribe->pipeline = pipeline;

        //GST_DEBUG_OBJECT(gvaaudiotranscribe, "Pipeline created, setting up configuration");

        /* auto *config = new ov::genai::WhisperGenerationConfig();
        *config = pipeline->get_generation_config();
        config->language = gvaaudiotranscribe->language;
        config->task = gvaaudiotranscribe->task;
        config->return_timestamps = gvaaudiotranscribe->return_timestamps;
        gvaaudiotranscribe->config = config; */

        //GST_INFO_OBJECT(gvaaudiotranscribe, "Whisper pipeline initialized successfully (language: %s, task: %s)",
        //               gvaaudiotranscribe->language, gvaaudiotranscribe->task);

        return TRUE;
    } catch (const std::exception &e) {
        GST_ERROR_OBJECT(gvaaudiotranscribe, "Failed to initialize wav2vec: %s", e.what());
        return FALSE;
    }
}

static gboolean gst_gva_audio_transcribe_stop(GstBaseTransform *base) {
    GvaAudioTranscribe *gvaaudiotranscribe = GVA_AUDIO_TRANSCRIBE(base);

    GST_DEBUG_OBJECT(gvaaudiotranscribe, "Stopping element");

    if (gvaaudiotranscribe->infer_request) {
        gvaaudiotranscribe->infer_request = {};
    }

    if (gvaaudiotranscribe->compiled_model) {
       gvaaudiotranscribe->compiled_model={}; 
    }

    if (gvaaudiotranscribe->core)
    {
        gvaaudiotranscribe->core.reset();
    }

    auto *audio_data = static_cast<std::vector<float> *>(gvaaudiotranscribe->audio_data);
    audio_data->clear();

    GST_DEBUG_OBJECT(gvaaudiotranscribe, "Element stopped successfully");
    return TRUE;
}

static GstFlowReturn gst_gva_audio_transcribe_transform_ip(GstBaseTransform *base, GstBuffer *buf) {
    GvaAudioTranscribe *gvaaudiotranscribe = GVA_AUDIO_TRANSCRIBE(base);
    auto *audio_data = static_cast<std::vector<float> *>(gvaaudiotranscribe->audio_data);
    auto *mutex = static_cast<std::mutex *>(gvaaudiotranscribe->mutex);

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
    gst_buffer_unmap(buf, &map);

    // Process audio when we have enough samples (e.g., 3 seconds)
    const size_t threshold_samples = SAMPLE_RATE * GST_AUDIO_TRANSCRIBE_THRESHOLD_SEC;
    if (audio_data->size() > threshold_samples) {
        GST_DEBUG_OBJECT(gvaaudiotranscribe, "Reached threshold of %zu samples, starting transcription", threshold_samples);
        try {
            std::lock_guard<std::mutex> lock(*mutex);
            size_t total_samples = audio_data->size();


            //ov::genai::RawSpeechInput input = *audio_data;
            ov::Shape input_shape = {1, total_samples};
            ov::Tensor input_tensor(gvaaudiotranscribe->compiled_model.input().get_element_type(),
                                input_shape,
                                audio_data->data());
            gvaaudiotranscribe->infer_request.set_input_tensor(input_tensor);
            //std::memcpy(input_tensor.data<float>(), audio_data->data(),
            //    num_samples * sizeof(float));
            gvaaudiotranscribe->infer_request.infer();
            ov::Tensor output_tensor = gvaaudiotranscribe->infer_request.get_output_tensor();
            auto output_shape = output_tensor.get_shape();
 

            GST_INFO_OBJECT(gvaaudiotranscribe, "Output tensor element type: %s",
                output_tensor.get_element_type().c_type_string().c_str());

           
            if (output_shape.size() < 2) {
                GST_ERROR_OBJECT(gvaaudiotranscribe, "Unexpected output shape from model");
                return GST_FLOW_ERROR;
            }
            size_t time_steps =  (output_shape.size() == 3) ? output_shape[1] : output_shape[0];
            size_t vocab_size = (output_shape.size() == 3) ? output_shape[2] : output_shape[1];
            

            float *logits = output_tensor.data<float>();
            if (!logits) {
                GST_ERROR_OBJECT(gvaaudiotranscribe, "Output tensor data is null");
                return GST_FLOW_ERROR;
            }

    
            std::vector<int64_t> token_ids(time_steps);
            for (size_t t = 0; t < time_steps; t++) {
                float max_val = -1e9f;
                int64_t max_idx = 0;
                for (size_t v = 0; v < vocab_size; v++) {
                    float val = logits[t * vocab_size + v];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = v;
                    }
                }
                token_ids[t] = max_idx;
            }




            gvaaudiotranscribe->alphabet = {
            "<pad>", "<s>", "</s>", "<unk>", "|", "e","t","a","o","n","i","h","s","r",
            "d","l","u","m","w","c","f","g","y","p","b","v","k","'","x","j","q","z"
            };

            std::vector<std::string> &alphabet = gvaaudiotranscribe->alphabet; // already initialized
            std::string transcription;
            int64_t prev = -1;
            for (auto id : token_ids) {
                if (id == 0 || id == prev) { // skip <pad> and repeated
                    prev = id;
                    continue;
                }
                 std::string token = (id < static_cast<int64_t>(alphabet.size())) ? alphabet[id] : "";
                if (token == "|") token = " ";
                transcription += token;
                prev = id;
            }


            GST_INFO_OBJECT(gvaaudiotranscribe,"Decoded text: %s", transcription.c_str());





            GST_INFO_OBJECT(gvaaudiotranscribe, "Running transcription on  samples");

            GST_LOG_OBJECT(gvaaudiotranscribe, "Clearing audio buffer after transcription");
            audio_data->clear();
            }
        catch (const std::exception &e) {
            GST_ERROR_OBJECT(gvaaudiotranscribe, "Error during transcription: %s", e.what());
        }
    }

            
    //gst_buffer_unmap(buf, &map);
    return GST_FLOW_OK;
}
