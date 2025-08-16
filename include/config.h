// ============= GPU Accelerate =============
// #define _GPU_delegate

// ============= write Video ================
// #define Write_Video__

// ============= Camera Choose ==============
// #define _v4l2cap
#define _openCVcap

// ============= model channel size =========
#define _640640
// #define _640384
// #define _512288
// #define _480480

// ============= read image size ============
#define input_video_width 1280
#define input_video_height 720

// ============= write image size ===========
#define output_video_width 1280
#define output_video_height 720
#define output_video_fps 30

// ============= V4L2 Set ===================
#define V4L2_cap_num 8
cv::Mat rgbImg1(input_video_height, input_video_width, CV_8UC3);
cv::Mat rgbImg(output_video_height, output_video_width, CV_8UC3); 
cv::Rect rect(0, 0, output_video_width, output_video_height);

// ================ Tensorflow Set ==========
#define NUM_CLASS 7
#define PROB_THRESHOLD 0.4
#define NMS_THRESHOLD_BBOX 0.4
#define NMS_THRESHOLD_LANE 0.85
#define Keypoint_NUM 15

#ifdef _640640
    #define INPUT_WIDTH 640
    #define INPUT_HEIGHT 640
    #define NUM_BOXES 8400
#endif

#ifdef _640384
    #define INPUT_WIDTH 640
    #define INPUT_HEIGHT 384
    #define NUM_BOXES 5040
#endif

#ifdef _512288
    #define INPUT_WIDTH 512
    #define INPUT_HEIGHT 288
    #define NUM_BOXES 3024
#endif

#ifdef _480480
    #define INPUT_WIDTH 480
    #define INPUT_HEIGHT 480
    #define NUM_BOXES 4725
#endif


// ================ System Set ================
const cv::Scalar RED(255, 0, 0);       // 紅色
const cv::Scalar GREEN(0, 255, 0);     // 綠色
const cv::Scalar BLUE(0, 0, 255);      // 藍色
const cv::Scalar YELLOW(255, 255, 0);  // 黃色
const cv::Scalar CYAN(0, 255, 255);    // 青色
const cv::Scalar MAGENTA(255, 0, 255); // 品紅色
const cv::Scalar WHITE(255, 255, 255); // 白色
const cv::Scalar BLACK(0, 0, 0);       // 黑色
const cv::Scalar GRAY(128, 128, 128);  // 灰色
const cv::Scalar ORANGE(255, 165, 0);  // 橙色
const cv::Scalar PINK(255, 192, 203);  // 粉色
const cv::Scalar PURPLE(128, 0, 128);  // 紫色
const cv::Scalar BROWN(165, 42, 42);   // 棕色

cv::Mat frame(input_video_height, input_video_width, CV_8UC3); 
cv::Mat Output_frame(input_video_height, input_video_width, CV_8UC3); 


