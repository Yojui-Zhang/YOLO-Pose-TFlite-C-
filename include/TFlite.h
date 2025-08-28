#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/ocl.hpp>  // OpenCL 支援

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/delegates/external/external_delegate.h>

#include "SortTracking.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;

class classifyDetector{
public:
    static constexpr const char* class_name_classify[classify_NUM_CLASS] = {"100km", "110km", "30km", "40km", "50km", 
                                                                            "60km", "70km", "80km", "90km", "car_left", 
                                                                            "car_normal", "car_right", "car_warning", "light_green", "light_other", 
                                                                            "light_red", "light_yellow", "sign_other"};

    void classify_init(char* classify_model_path);
    cv::Mat cropObjects(const Mat& frame, const TrackingBox &obj, int classify_model_width, int classify_model_height);
};

class PoseDetector {
public:
    static constexpr const char* class_names[NUM_CLASS] = {"roadlane", "car", "rider", "person", "light", "signC", "signT"};

    // 模型與資料
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    float* input_data = nullptr;
    float* yolov8_output = nullptr;

    // 預處理 scale 結果
    float scale_factor;
    int new_width, new_height;
    int top, bottom, left, right;
    float mean[3] = {0, 0, 0};
    float scale[3] = {0.003921f, 0.003921f, 0.003921f};

    // Function
    static float sigmoid(float x);
    float intersection_over_union(const cv::Rect &rect1, const cv::Rect &rect2);
    void nms(std::vector<Object> &objects, float nms_threshold_bbox, float nms_threshold_lane);
    void generate_proposals(const float *data, float prob_threshold, std::vector<Object> &objects, float scale, int top, int left);
    void get_input_data_fp32(const  cv::Mat &sample,
                             float          *input_data,
                             int            input_height, 
                             int            input_width, 
                             const float    *mean, 
                             const float    *scale, 
                             int            new_width, 
                             int            new_height, 
                             int            top, 
                             int            bottom, 
                             int            left, 
                             int            right);

    cv::Mat draw_objects(const cv::Mat &img, const std::vector<Object> &objects, int classify_model_width, int classify_model_height);

    bool Set_TFlite(const char* model_path);
    void Calculate_Scale(const cv::Mat& frame, int input_width, int input_height);
};

Net net;
SORTTRACKING sorttracking;
classifyDetector classifydetector__;

void classifyDetector::classify_init(char* classify_model_path)
{
    #ifdef _GPU_delegate
    // ================== 啟用 OpenCL 加速 ===================
    if (cv::ocl::haveOpenCL()) {
        cv::ocl::setUseOpenCL(true);
        cout << "OpenCL is enabled!" << endl;
    } else {
        cout << "OpenCL is not supported on this device." << endl;
    }

#endif
    // ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

    net = readNetFromONNX(classify_model_path);

    if (net.empty()) {
        cerr << "Failed to load ONNX model!" << endl;
    }
#ifdef _GPU_delegate
    // 使用 OpenCL 進行推論加速
    net.setPreferableBackend(DNN_BACKEND_DEFAULT);  // 讓 OpenCV 自動選擇最佳後端
    net.setPreferableTarget(DNN_TARGET_OPENCL);     // 指定使用 OpenCL 進行加速
#endif
}

// 裁剪偵測到的物件
cv::Mat classifyDetector::cropObjects(const Mat& frame, const TrackingBox &obj, int classify_model_width, int classify_model_height ) {

    Mat crop_image;

    int width = classify_model_width;
    int height = classify_model_height;

    // 取得偵測物件的邊界框
    cv::Rect roi = obj.box;

    // 確保裁剪區域不超過影像邊界
    roi &= Rect(0, 0, frame.cols, frame.rows);

    // 裁剪影像
    Mat croppedObject = frame(roi).clone(); // 需要 clone() 避免引用原圖

    resize(croppedObject, crop_image, cv::Size(width, height));

    return crop_image;
}

void PoseDetector::Calculate_Scale(const cv::Mat& frame, int input_width, int input_height) {
    float scale_x = static_cast<float>(frame.cols) / input_width;
    float scale_y = static_cast<float>(frame.rows) / input_height;
    scale_factor = std::max(scale_x, scale_y);

    new_width = static_cast<int>(frame.cols / scale_factor);
    new_height = static_cast<int>(frame.rows / scale_factor);

    top = (input_height - new_height) / 2;
    bottom = input_height - new_height - top;
    left = (input_width - new_width) / 2;
    right = input_width - new_width - left;

    std::cout << "[Scale Info]" << std::endl;
    std::cout << "scale_factor: " << scale_factor << std::endl;
    std::cout << "new_width: " << new_width << ", new_height: " << new_height << std::endl;
    std::cout << "top: " << top << ", bottom: " << bottom << std::endl;
    std::cout << "left: " << left << ", right: " << right << std::endl;
}


bool PoseDetector::Set_TFlite(const char* model_path) {
    model = tflite::FlatBufferModel::BuildFromFile(model_path);
    if (!model) {
        std::cerr << "Failed to mmap model\n";
        return false;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to construct interpreter\n";
        return false;
    }

#ifdef _GPU_delegate
    const char* vx_delegate_library_path = "/usr/lib/libvx_delegate.so";
    TfLiteExternalDelegateOptions delegate_options = TfLiteExternalDelegateOptionsDefault(vx_delegate_library_path);
    TfLiteDelegate* vx_delegate = TfLiteExternalDelegateCreate(&delegate_options);
    if (interpreter->ModifyGraphWithDelegate(vx_delegate) != kTfLiteOk) {
        std::cerr << "Fail to create vx delegate\n";
        return false;
    }
#endif

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors!" << std::endl;
        return false;
    }

    int input_index = interpreter->inputs()[0];
    input_data = interpreter->typed_tensor<float>(input_index);

    int output_index = interpreter->outputs()[0];
    yolov8_output = interpreter->tensor(output_index)->data.f;

    // 顯示模型資訊（可選）
    std::cout << "Input tensor type: " << TfLiteTypeGetName(interpreter->tensor(input_index)->type) << std::endl;
    TfLiteIntArray* input_dims = interpreter->tensor(input_index)->dims;
    for (int i = 0; i < input_dims->size; ++i)
        std::cout << "Input[" << i << "]: " << input_dims->data[i] << std::endl;

    TfLiteIntArray* output_dims = interpreter->tensor(output_index)->dims;
    std::cout << "Output tensor type: " << TfLiteTypeGetName(interpreter->tensor(output_index)->type) << std::endl;
    for (int i = 0; i < output_dims->size; ++i)
        std::cout << "Output[" << i << "]: " << output_dims->data[i] << std::endl;

    return true;
}

float PoseDetector::sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

float PoseDetector::intersection_over_union(const cv::Rect &rect1, const cv::Rect &rect2)
{
    float area1 = rect1.area();
    float area2 = rect2.area();

    cv::Rect intersection = rect1 & rect2;
    float intersection_area = intersection.area();
    float union_area = area1 + area2 - intersection_area;

    return intersection_area / union_area;
}

void PoseDetector::nms(std::vector<Object> &objects, float nms_threshold_bbox, float nms_threshold_lane)
{
    std::sort(objects.begin(), objects.end(), [](const Object &a, const Object &b)
              { return a.score > b.score; });

    std::vector<int> picked;
    for (int i = 0; i < objects.size(); ++i)
    {
        const Object &a = objects[i];
        bool keep = true;
        for (int j = 0; j < picked.size(); ++j)
        {
            const Object &b = objects[picked[j]];
            float iou = intersection_over_union(a.box, b.box);
            if (iou > nms_threshold_bbox && (a.class_id > 0 && b.class_id > 0))
            {
                keep = false;
                break;
            }
            else if(iou > nms_threshold_lane )
            {
                keep = false;
                break;
            }
        }
        if (keep)
        {
            picked.push_back(i);
        }
    }

    std::vector<Object> nms_objects;
    for (int i = 0; i < picked.size(); ++i)
    {
        nms_objects.push_back(objects[picked[i]]);
    }
    objects = nms_objects;
}


void PoseDetector::generate_proposals(const float *data, float prob_threshold, std::vector<Object> &objects, float scale, int top, int left)
{

    const int num_keypoints = Keypoint_NUM;
    const int class_start_channel = 4;
    const int kpt_start_channel = 4 + NUM_CLASS;

    for (int i = 0; i < NUM_BOXES; ++i)
    {

        float x = (data[0 * NUM_BOXES + i] * INPUT_WIDTH - left) * scale;
        float y = (data[1 * NUM_BOXES + i] * INPUT_HEIGHT - top) * scale;
        float w = (data[2 * NUM_BOXES + i]) * INPUT_WIDTH * scale;
        float h = (data[3 * NUM_BOXES + i]) * INPUT_HEIGHT * scale;
        float x1 = x - w / 2;
        float y1 = y - h / 2;
        cv::Rect rect(cv::Point(x1, y1), cv::Size(w, h));

        int class_id = -1;
        float max_prob = prob_threshold;
        for (int j = 0; j < NUM_CLASS; ++j)
        {
            float prob = data[(class_start_channel + j) * NUM_BOXES + i];
            if (prob > max_prob)
            {
                class_id = j;
                max_prob = prob;
            }
        }

        if (class_id >= 0)
        {
            Object obj;
            obj.class_id = class_id;
            obj.score = max_prob;
            obj.box = rect;

            obj.kpts.clear();
            for (int k = 0; k < num_keypoints; k++)
            {
                
                float kpt_x = (data[(kpt_start_channel + k*3 + 0) * NUM_BOXES + i] * INPUT_WIDTH  - left) * scale;
                float kpt_y = (data[(kpt_start_channel + k*3 + 1) * NUM_BOXES + i] * INPUT_HEIGHT - top ) * scale;
                float kpt_v =  data[(kpt_start_channel + k*3 + 2) * NUM_BOXES + i];

                obj.kpts.push_back(cv::Point3f(kpt_x, kpt_y, kpt_v));
            
            }

            objects.push_back(obj);
        }
    }

}


void PoseDetector::get_input_data_fp32(const cv::Mat &sample, float *input_data, int input_height, int input_width, const float *mean, const float *scale, int new_width, int new_height, int top, int bottom, int left, int right)
{

    cv::Mat img;
    cv::resize(sample, img, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB); // BGR → RGB
    img.convertTo(img, CV_32FC3, 1.0 / 255.0);  // Normalize once

    // 使用與 YOLOv8 相同的 padding 值（0.447 ≈ 114 / 255）
    cv::Mat img_new(input_height, input_width, CV_32FC3, cv::Scalar(0.447059, 0.447059, 0.447059));
    img.copyTo(img_new(cv::Rect(left, top, new_width, new_height)));

    // 直接複製資料（不再乘 scale）
    std::memcpy(input_data, img_new.data, input_height * input_width * 3 * sizeof(float));

}

cv::Mat PoseDetector::draw_objects(const cv::Mat &img, const std::vector<Object> &objects, int classify_model_width, int classify_model_height)
{


    cv::Mat image = img.clone();

    std::vector<TrackingBox> TrackingResult = sorttracking.TrackingResult(objects);

    for (const auto &obj : TrackingResult)
    {
        std::string label_txt;
        bool classify_light__ = false;
        int traffic_class_num;

        if(obj.class_id != 0){

            if((obj.class_id == 1 && obj.box.x >= 550 && obj.box.x <= 730) || obj.class_id == 4 || obj.class_id == 5 || obj.class_id == 6 ){

                Mat crop_image;
                crop_image = classifydetector__.cropObjects(image, obj, classify_model_width, classify_model_height);  //to crop_image

                // 調整輸入大小 (根據 ONNX 模型需求)
                Mat blob;
                Size inputSize(classify_model_width, classify_model_height);  // 根據模型需求調整
                blobFromImage(crop_image, blob, 1.0 / 255, inputSize, Scalar(), true, false);

                // 設定模型輸入
                net.setInput(blob);

                Mat classify_output = net.forward();

                // 解析結果
                Point classId;
                double confidence;
                minMaxLoc(classify_output, nullptr, &confidence, nullptr, &classId);

                // cout << "Predicted Class: " << class_name_classify[classId.x] << ", Confidence: " << confidence << endl;

                classify_light__ = true;
                traffic_class_num = classId.x;

            }

            // Draw bbox
            cv::rectangle(image, obj.box, GREEN, 2);

            // Draw class label
            if(classify_light__ == false){
                label_txt = cv::format("%s", class_names[obj.class_id]);
            }
            else if(classify_light__ == true){
                label_txt = cv::format("%s", classifydetector__.class_name_classify[traffic_class_num]);
            }


            int baseline = 0;
            cv::Size textSize = cv::getTextSize(label_txt, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::Point textOrg(obj.box.x, obj.box.y - 5);
            cv::putText(image, label_txt, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1);
        }
        else{
            // Draw keypoints (if any)
            for (const auto& kpt : obj.kpts)
            {
                int x = static_cast<int>(kpt.x);
                int y = static_cast<int>(kpt.y);
                int v = static_cast<int>(kpt.z);  // visibility

                cv::circle(image, cv::Point(x, y), 3, RED, -1);   
            }
        }
    }

    return image;
}