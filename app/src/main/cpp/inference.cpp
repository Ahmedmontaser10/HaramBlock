#include <jni.h>
#include <string>
#include <android/bitmap.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <ncnn/net.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <map>

// Model configurations
const int FD_MODEL_WIDTH = 160;
const int FD_MODEL_HEIGHT = 120;
const int GENDER_MODEL_WIDTH = 32;
const int GENDER_MODEL_HEIGHT = 32;
const float FACE_THRESHOLD = 0.50f;
const float GENDER_CONF_THRESH = 0.9f;
const float IOU_THRESHOLD = 0.3f;

// Structure for detection results
struct Detection {
    float x1, y1, x2, y2;
    float score;
};

// Singleton class for model management
class ModelManager {
private:
    ncnn::Net face_net;
    ncnn::Net gender_net;
    bool models_loaded = false;
    std::mutex model_mutex;

    ModelManager() {} // Prevent direct instantiation

    // Configure network options for better performance
    void configureNetOptions(ncnn::Net& net) {
        net.opt.use_vulkan_compute = false;
        net.opt.num_threads = 4; // Increased threads for better performance
        net.opt.use_winograd_convolution = true;
        net.opt.use_sgemm_convolution = true;
        net.opt.use_fp16_packed = true;
        net.opt.use_fp16_storage = true;
        net.opt.use_fp16_arithmetic = true;
    }

public:
    // Delete copy constructor and assignment operator
    ModelManager(const ModelManager&) = delete;
    void operator=(const ModelManager&) = delete;

    // Get singleton instance
    static ModelManager& getInstance() {
        static ModelManager instance;
        return instance;
    }

    // Initialize models with thread safety
    std::string initialize_models(AAssetManager* mgr) {
        std::lock_guard<std::mutex> lock(model_mutex);
        __android_log_print(ANDROID_LOG_INFO, "ModelManager", "Starting model initialization");

        if (models_loaded) {
            __android_log_print(ANDROID_LOG_INFO, "ModelManager", "Models already loaded");
            return "SUCCESS";
        }

        if (!mgr) {
            __android_log_print(ANDROID_LOG_ERROR, "ModelManager", "Asset manager is null");
            return "ERROR: Asset manager is null";
        }

        // Load face detection model
        __android_log_print(ANDROID_LOG_INFO, "ModelManager", "Loading face detection model");
        configureNetOptions(face_net);

        int ret = face_net.load_param(mgr, "quant.param");
        if (ret) {
            __android_log_print(ANDROID_LOG_ERROR, "ModelManager", "Failed to load face detection param, error code: %d", ret);
            return "Failed to load face detection param";
        }
        __android_log_print(ANDROID_LOG_INFO, "ModelManager", "Face detection param loaded successfully");

        ret = face_net.load_model(mgr, "quant.bin");
        if (ret) {
            __android_log_print(ANDROID_LOG_ERROR, "ModelManager", "Failed to load face detection bin, error code: %d", ret);
            return "Failed to load face detection bin";
        }
        __android_log_print(ANDROID_LOG_INFO, "ModelManager", "Face detection model loaded successfully");

        // Load gender classification model
        __android_log_print(ANDROID_LOG_INFO, "ModelManager", "Loading gender classification model");
        configureNetOptions(gender_net);

        ret = gender_net.load_param(mgr, "gender.param");
        if (ret) {
            __android_log_print(ANDROID_LOG_ERROR, "ModelManager", "Failed to load gender param, error code: %d", ret);
            return "Failed to load gender param";
        }
        __android_log_print(ANDROID_LOG_INFO, "ModelManager", "Gender param loaded successfully");

        ret = gender_net.load_model(mgr, "gender.bin");
        if (ret) {
            __android_log_print(ANDROID_LOG_ERROR, "ModelManager", "Failed to load gender bin, error code: %d", ret);
            return "Failed to load gender bin";
        }
        __android_log_print(ANDROID_LOG_INFO, "ModelManager", "Gender model loaded successfully");

        models_loaded = true;
        __android_log_print(ANDROID_LOG_INFO, "ModelManager", "All models initialized successfully");
        return "SUCCESS";
    }

    // Check if models are loaded
    bool areModelsLoaded() {
        std::lock_guard<std::mutex> lock(model_mutex);
        return models_loaded;
    }

    // Get face net with thread safety
    ncnn::Net& getFaceNet() {
        std::lock_guard<std::mutex> lock(model_mutex);
        return face_net;
    }

    // Get gender net with thread safety
    ncnn::Net& getGenderNet() {
        std::lock_guard<std::mutex> lock(model_mutex);
        return gender_net;
    }
};

// Cache for classification results
class ClassificationCache {
private:
    std::map<std::string, bool> cache;
    std::mutex cache_mutex;

public:
    bool getResult(const std::string& image_hash) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = cache.find(image_hash);
        return it != cache.end() ? it->second : false;
    }

    void storeResult(const std::string& image_hash, bool result) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache[image_hash] = result;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache.clear();
    }
};

// Global cache instance
static ClassificationCache resultCache;

// Detailed processing result enum
enum class ProcessingResult {
    SUCCESS,
    NO_FACES,
    FEMALE_DETECTED,
    TIMEOUT,
    MODEL_LOAD_FAILED,
    IMAGE_PROCESSING_ERROR
};

// Fast NMS implementation with timeout protection
std::vector<Detection> hard_nms(std::vector<Detection>& detections, float iou_threshold) {
    if (detections.empty()) return {};

    auto start_time = std::chrono::high_resolution_clock::now();
    const int max_nms_time = 1000; // 1 second max for NMS

    // Sort by score
    std::sort(detections.begin(), detections.end(),
        [](const Detection& a, const Detection& b) { return a.score < b.score; });

    std::vector<Detection> picked;
    std::vector<bool> suppressed(detections.size(), false);

    for (int i = detections.size() - 1; i >= 0; --i) {
        // Check timeout periodically
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
        if (elapsed.count() > max_nms_time) {
            __android_log_print(ANDROID_LOG_WARN, "hard_nms", "NMS timeout, returning partial results");
            break;
        }

        if (suppressed[i]) continue;

        const Detection& current = detections[i];
        picked.push_back(current);

        float current_area = (current.x2 - current.x1) * (current.y2 - current.y1);

        for (int j = i - 1; j >= 0; --j) {
            if (suppressed[j]) continue;

            const Detection& test = detections[j];

            float xx1 = std::max(current.x1, test.x1);
            float yy1 = std::max(current.y1, test.y1);
            float xx2 = std::min(current.x2, test.x2);
            float yy2 = std::min(current.y2, test.y2);

            float w = std::max(0.0f, xx2 - xx1);
            float h = std::max(0.0f, yy2 - yy1);
            float inter = w * h;

            float test_area = (test.x2 - test.x1) * (test.y2 - test.y1);
            float iou = inter / (current_area + test_area - inter + 1e-5f);

            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    __android_log_print(ANDROID_LOG_INFO, "hard_nms", "NMS completed: %zu detections from %zu original",
        picked.size(), detections.size());
    return picked;
}

// Process face detection output
std::vector<Detection> process_face_output(ncnn::Mat& conf_mat, ncnn::Mat& box_mat,
    int img_width, int img_height) {
    __android_log_print(ANDROID_LOG_INFO, "FaceOutput", "Starting face output processing");
    std::vector<Detection> detections;

    const float* conf_data = conf_mat.channel(0);
    const float* box_data = box_mat.channel(0);

    int num_anchors = conf_mat.h;
    __android_log_print(ANDROID_LOG_INFO, "FaceOutput", "Processing %d anchors", num_anchors);

    // Add timeout protection for face output processing
    auto face_output_start = std::chrono::high_resolution_clock::now();
    const int max_face_output_time = 3000; // 3 seconds max for face output processing

    for (int i = 0; i < num_anchors; ++i) {
        // Check timeout periodically
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - face_output_start);
        if (elapsed.count() > max_face_output_time) {
            __android_log_print(ANDROID_LOG_WARN, "FaceOutput", "Timeout during anchor processing, returning partial results");
            break;
        }

        // Get confidence for face class (index 1)
        float conf = conf_data[i * 2 + 1];

        if (conf > FACE_THRESHOLD) {
            Detection det;
            det.x1 = box_data[i * 4 + 0] * img_width;
            det.y1 = box_data[i * 4 + 1] * img_height;
            det.x2 = box_data[i * 4 + 2] * img_width;
            det.y2 = box_data[i * 4 + 3] * img_height;
            det.score = conf;
            detections.push_back(det);

            // Log every 100th detection to avoid too much logging
            if (detections.size() % 100 == 0) {
                __android_log_print(ANDROID_LOG_INFO, "FaceOutput", "Processed %d anchors, found %zu detections", i + 1, detections.size());
            }
        }
    }

    __android_log_print(ANDROID_LOG_INFO, "FaceOutput", "Found %zu detections before NMS", detections.size());
    std::vector<Detection> result = hard_nms(detections, IOU_THRESHOLD);
    __android_log_print(ANDROID_LOG_INFO, "FaceOutput", "NMS completed, %zu detections remaining", result.size());
    return result;
}

// Inline softmax for 2 classes
inline void softmax2(float& val0, float& val1) {
    float max_val = std::max(val0, val1);
    float exp0 = std::exp(val0 - max_val);
    float exp1 = std::exp(val1 - max_val);
    float sum = exp0 + exp1;
    val0 = exp0 / sum;
    val1 = exp1 / sum;
}

// Process a single face for gender classification
bool processSingleFace(const Detection& face, const cv::Mat& rgb_src, ncnn::Net& gender_net) {
    int orig_width = rgb_src.cols;
    int orig_height = rgb_src.rows;

    int x1 = std::max(0, static_cast<int>(face.x1));
    int y1 = std::max(0, static_cast<int>(face.y1));
    int x2 = std::min(orig_width, static_cast<int>(face.x2));
    int y2 = std::min(orig_height, static_cast<int>(face.y2));

    // Validate face crop dimensions
    if (x2 <= x1 || y2 <= y1) {
        __android_log_print(ANDROID_LOG_WARN, "SingleFace", "Invalid face crop dimensions: %d,%d,%d,%d", x1, y1, x2, y2);
        return false;
    }

    cv::Mat face_crop = rgb_src(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    cv::Mat face_resized;
    cv::resize(face_crop, face_resized, cv::Size(GENDER_MODEL_WIDTH, GENDER_MODEL_HEIGHT));

    ncnn::Mat gender_input = ncnn::Mat::from_pixels(face_resized.data, ncnn::Mat::PIXEL_BGR2RGB, GENDER_MODEL_WIDTH, GENDER_MODEL_HEIGHT);

    const float gender_norm[3] = { 1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f };
    const float gender_mean[3] = { 0.0f, 0.0f, 0.0f };
    gender_input.substract_mean_normalize(gender_mean, gender_norm);

    ncnn::Mat gender_input_chw;
    ncnn::convert_packing(gender_input, gender_input_chw, 1);

    ncnn::Extractor gender_ex = gender_net.create_extractor();
    gender_ex.set_light_mode(true);
    gender_ex.input("in0", gender_input_chw);

    ncnn::Mat gender_output;
    gender_ex.extract("out0", gender_output);

    if (gender_output.w >= 2) {
        float female_score = gender_output[0];
        float male_score = gender_output[1];

        // Apply softmax
        float sum = std::exp(female_score) + std::exp(male_score);
        female_score = std::exp(female_score) / sum;
        male_score = std::exp(male_score) / sum;

        __android_log_print(ANDROID_LOG_INFO, "SingleFace", "Female=%.3f, Male=%.3f", female_score, male_score);

        if (female_score >= male_score) {
            __android_log_print(ANDROID_LOG_INFO, "SingleFace", "Female detected");
            return true;  // Female detected
        }
    }
    else {
        __android_log_print(ANDROID_LOG_WARN, "SingleFace", "Invalid gender output: width=%d", gender_output.w);
    }

    return false;  // No female detected in this face
}

// Process image with detailed result
ProcessingResult process_image_with_gender_count(cv::Mat& src, AAssetManager* mgr) {
    // Add timeout protection
    auto start_time = std::chrono::high_resolution_clock::now();
    const int max_processing_time = 10000; // 10 seconds max processing time

    __android_log_print(ANDROID_LOG_INFO, "ImageProcessing", "Step 1 - Checking models loaded status");

    // Initialize models if needed
    ModelManager& modelManager = ModelManager::getInstance();
    if (!modelManager.areModelsLoaded()) {
        __android_log_print(ANDROID_LOG_INFO, "ImageProcessing", "Step 1a - Models not loaded, initializing...");
        std::string init_result = modelManager.initialize_models(mgr);
        if (init_result != "SUCCESS") {
            __android_log_print(ANDROID_LOG_ERROR, "ImageProcessing", "Step 1a - Model initialization failed: %s", init_result.c_str());
            return ProcessingResult::MODEL_LOAD_FAILED;
        }
        __android_log_print(ANDROID_LOG_INFO, "ImageProcessing", "Step 1a - Models initialized successfully");
    }
    else {
        __android_log_print(ANDROID_LOG_INFO, "ImageProcessing", "Step 1a - Models already loaded");
    }

    // Check timeout
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
    if (elapsed.count() > max_processing_time) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageProcessing", "Step 1b - Timeout during model initialization");
        return ProcessingResult::TIMEOUT;
    }

    __android_log_print(ANDROID_LOG_INFO, "ImageProcessing", "Step 2 - Converting image to RGB");
    // Convert to RGB
    cv::Mat rgb_src;
    if (src.channels() == 3) {
        cv::cvtColor(src, rgb_src, cv::COLOR_BGR2RGB);
        __android_log_print(ANDROID_LOG_INFO, "ImageProcessing", "Step 2a - Converted 3-channel BGR to RGB");
    }
    else if (src.channels() == 4) {
        cv::cvtColor(src, rgb_src, cv::COLOR_BGRA2RGB);
        __android_log_print(ANDROID_LOG_INFO, "ImageProcessing", "Step 2a - Converted 4-channel BGRA to RGB");
    }
    else {
        __android_log_print(ANDROID_LOG_ERROR, "ImageProcessing", "Step 2a - Unsupported image format: %d channels", src.channels());
        return ProcessingResult::IMAGE_PROCESSING_ERROR;
    }

    int orig_width = rgb_src.cols;
    int orig_height = rgb_src.rows;
    __android_log_print(ANDROID_LOG_INFO, "ImageProcessing", "Step 2b - Processing image: %dx%d", orig_width, orig_height);

    __android_log_print(ANDROID_LOG_INFO, "ImageProcessing", "Step 3 - Starting face detection");
    // Face detection
    cv::Mat resized_fd;
    cv::resize(rgb_src, resized_fd, cv::Size(FD_MODEL_WIDTH, FD_MODEL_HEIGHT));
    __android_log_print(ANDROID_LOG_INFO, "ImageProcessing", "Step 3a - Resized image for face detection");

    ncnn::Mat fd_input = ncnn::Mat::from_pixels(resized_fd.data, ncnn::Mat::PIXEL_RGB, FD_MODEL_WIDTH, FD_MODEL_HEIGHT);
    __android_log_print(ANDROID_LOG_INFO, "ImageProcessing", "Step 3b - Created ncnn input tensor");

    const float mean_vals[3] = { 127.0f, 127.0f, 127.0f };
    const float norm_vals[3] = { 1.0f / 160.0f, 1.0f / 160.0f, 1.0f / 160.0f };
    fd_input.substract_mean_normalize(mean_vals, norm_vals);
    __android_log_print(ANDROID_LOG_INFO, "ImageProcessing", "Step 3c - Normalized input tensor");

    ncnn::Extractor face_ex = modelManager.getFaceNet().create_extractor();
    face_ex.set_light_mode(true);
    face_ex.input("in0", fd_input);
    __android_log_print(ANDROID_LOG_INFO, "ImageProcessing", "Step 3d - Created face extractor and set input");

    ncnn::Mat conf_mat, box_mat;
    face_ex.extract("out0", conf_mat);
    face_ex.extract("out1", box_mat);
    __android_log_print(ANDROID_LOG_INFO, "ImageProcessing", "Step 3e - Extracted face detection outputs");

    // Check timeout before processing face outputs
    auto face_process_time = std::chrono::high_resolution_clock::now();
    auto face_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(face_process_time - start_time);
    if (face_elapsed.count() > 7000) {  // 7 seconds timeout before face processing
        __android_log_print(ANDROID_LOG_ERROR, "ImageProcessing", "Step 3e - Timeout before face processing, elapsed: %lld ms",
            static_cast<long long>(face_elapsed.count()));
        return ProcessingResult::TIMEOUT;
    }

    std::vector<Detection> faces = process_face_output(conf_mat, box_mat, orig_width, orig_height);
    __android_log_print(ANDROID_LOG_INFO, "ImageProcessing", "Step 3f - Processed face outputs, detected %zu faces", faces.size());

    if (faces.empty()) {
        __android_log_print(ANDROID_LOG_INFO, "ImageProcessing", "Step 4 - No faces detected");
        return ProcessingResult::NO_FACES;
    }

    __android_log_print(ANDROID_LOG_INFO, "ImageProcessing", "Step 5 - Starting face classification");
    // Process each detected face
    for (size_t i = 0; i < faces.size(); ++i) {
        // Check timeout periodically
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
        if (elapsed.count() > max_processing_time) {
            __android_log_print(ANDROID_LOG_ERROR, "ImageProcessing", "Step 5a - Timeout during face processing");
            return ProcessingResult::TIMEOUT;
        }

        const auto& face = faces[i];
        __android_log_print(ANDROID_LOG_INFO, "ImageProcessing", "Step 5b - Processing face %zu at (%.1f,%.1f)-(%.1f,%.1f)",
            i, face.x1, face.y1, face.x2, face.y2);

        if (processSingleFace(face, rgb_src, modelManager.getGenderNet())) {
            __android_log_print(ANDROID_LOG_INFO, "ImageProcessing", "Step 5c - Female detected in face %zu", i);
            return ProcessingResult::FEMALE_DETECTED;
        }
    }

    __android_log_print(ANDROID_LOG_INFO, "ImageProcessing", "Step 6 - No females detected");
    return ProcessingResult::SUCCESS;
}

// Convert bitmap to Mat with smart pointer
std::unique_ptr<cv::Mat> Bitmap2Mat(JNIEnv* env, jobject bitmap, jboolean needUnPremultiplyAlpha) {
    if (bitmap == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "Bitmap2Mat", "Error: bitmap is null");
        return nullptr;
    }

    AndroidBitmapInfo info;
    void* pixels = 0;

    int ret = AndroidBitmap_getInfo(env, bitmap, &info);
    if (ret < 0) {
        __android_log_print(ANDROID_LOG_ERROR, "Bitmap2Mat", "AndroidBitmap_getInfo() failed, error code: %d", ret);
        return nullptr;
    }

    __android_log_print(ANDROID_LOG_INFO, "Bitmap2Mat", "Bitmap info: width=%d, height=%d, format=%d",
        info.width, info.height, info.format);

    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 && info.format != ANDROID_BITMAP_FORMAT_RGB_565) {
        __android_log_print(ANDROID_LOG_ERROR, "Bitmap2Mat", "Unsupported bitmap format: %d", info.format);
        return nullptr;
    }

    ret = AndroidBitmap_lockPixels(env, bitmap, &pixels);
    if (ret < 0) {
        __android_log_print(ANDROID_LOG_ERROR, "Bitmap2Mat", "AndroidBitmap_lockPixels() failed, error code: %d", ret);
        return nullptr;
    }

    if (pixels == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "Bitmap2Mat", "Error: pixels is null after lock");
        AndroidBitmap_unlockPixels(env, bitmap);
        return nullptr;
    }

    auto mat = std::make_unique<cv::Mat>();
    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        *mat = cv::Mat(info.height, info.width, CV_8UC4, pixels);
        if (needUnPremultiplyAlpha) {
            cvtColor(*mat, *mat, cv::COLOR_mRGBA2RGBA);
        }
    }
    else {
        cv::Mat tmp(info.height, info.width, CV_8UC2, pixels);
        cvtColor(tmp, *mat, cv::COLOR_BGR5652RGBA);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
    return mat;
}

// Simple hash function for image content (for caching)
std::string generateImageHash(const cv::Mat& image) {
    std::stringstream ss;
    ss << image.cols << "x" << image.rows << "-" << image.channels();

    // Use a simple hash based on image dimensions and first few pixels
    int sample_points = std::min(10, image.cols * image.rows);
    for (int i = 0; i < sample_points; i++) {
        int x = i % image.cols;
        int y = i / image.cols;
        if (y < image.rows) {
            if (image.channels() == 1) {
                ss << "-" << static_cast<int>(image.at<uchar>(y, x));
            }
            else if (image.channels() == 3) {
                auto pixel = image.at<cv::Vec3b>(y, x);
                ss << "-" << static_cast<int>(pixel[0])
                    << "," << static_cast<int>(pixel[1])
                    << "," << static_cast<int>(pixel[2]);
            }
            else if (image.channels() == 4) {
                auto pixel = image.at<cv::Vec4b>(y, x);
                ss << "-" << static_cast<int>(pixel[0])
                    << "," << static_cast<int>(pixel[1])
                    << "," << static_cast<int>(pixel[2])
                    << "," << static_cast<int>(pixel[3]);
            }
        }
    }

    return ss.str();
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_haram_block_ImageViewAccessibilityService_ImageClassification(
    JNIEnv* env,
    jobject,
    jobject bitmapIn,
    jobject assetManager) {

    // Add timeout protection to prevent infinite loops
    auto start_time = std::chrono::high_resolution_clock::now();

    if (bitmapIn == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageClassification", "Error: bitmapIn is null");
        return env->NewStringUTF("false");
    }

    __android_log_print(ANDROID_LOG_INFO, "ImageClassification", "Step 1: Converting bitmap to Mat");
    auto src = Bitmap2Mat(env, bitmapIn, false);

    if (!src || src->empty()) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageClassification", "Error: Source image is empty");
        return env->NewStringUTF("false");
    }
    __android_log_print(ANDROID_LOG_INFO, "ImageClassification", "Step 1: Bitmap conversion successful, size: %dx%d", src->cols, src->rows);

    // Check cache first
    std::string image_hash = generateImageHash(*src);
    bool cached_result = resultCache.getResult(image_hash);
    __android_log_print(ANDROID_LOG_INFO, "ImageClassification", "Step 1a: Image hash: %s", image_hash.c_str());

    __android_log_print(ANDROID_LOG_INFO, "ImageClassification", "Step 2: Getting asset manager");
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    if (mgr == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageClassification", "Error: Asset manager is null");
        return env->NewStringUTF("false");
    }
    __android_log_print(ANDROID_LOG_INFO, "ImageClassification", "Step 2: Asset manager obtained");

    __android_log_print(ANDROID_LOG_INFO, "ImageClassification", "Step 3: Starting image classification processing");

    // Add a timeout check before starting processing
    auto pre_process_time = std::chrono::high_resolution_clock::now();
    auto pre_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(pre_process_time - start_time);
    if (pre_elapsed.count() > 8000) {  // 8 seconds timeout before processing
        __android_log_print(ANDROID_LOG_ERROR, "ImageClassification", "Step 3: Timeout before processing, elapsed: %lld ms",
            static_cast<long long>(pre_elapsed.count()));
        return env->NewStringUTF("false");
    }

    ProcessingResult result = process_image_with_gender_count(*src, mgr);
    bool final_result = (result == ProcessingResult::SUCCESS || result == ProcessingResult::NO_FACES);

    // Cache the result
    resultCache.storeResult(image_hash, final_result);

    __android_log_print(ANDROID_LOG_INFO, "ImageClassification", "Step 3: Classification processing completed, result: %s", final_result ? "true" : "false");

    // Check if processing took too long
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    __android_log_print(ANDROID_LOG_INFO, "ImageClassification",
        "Step 4: Classification completed in %lld ms, result: %s",
        static_cast<long long>(duration.count()), final_result ? "true" : "false");

    __android_log_print(ANDROID_LOG_INFO, "ImageClassification", "Step 5: Returning result to Java");
    return env->NewStringUTF(final_result ? "true" : "false");
}
