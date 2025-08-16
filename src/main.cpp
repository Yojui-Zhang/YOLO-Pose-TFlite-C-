// 基本輸入輸出
#include <iostream>
#include <fstream>
#include <sstream>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// TensorFlow Lite 核心
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/delegates/external/external_delegate.h"

#ifdef _GPU_delegate
    #include "tensorflow/lite/delegates/gpu/delegate.h"
    #include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#endif

// 時間函式
#include <ctime>

// 信號處理（如 ctrl+c 可選）
#include <csignal>

// 自訂頭檔
#include "config.h"
#include "TFlite.h"
#include "write_video.h"
#ifdef _v4l2cap
    #include "V4L2_define.h"        
    int v4l2res = v4l2init(V4L2_cap_num);
#endif

using namespace std;
using namespace cv;

PoseDetector pose;


int main(int argc, char **argv)
{

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path> " << std::endl;
        return 1;
    }

    const char* model_path = argv[1];

// ==============================================================
#ifdef _openCVcap
    // 初始化影片路徑
    const char *inputVideoPath = "../video/vecow-demo.mp4";
    // const char *inputVideoPath = "../video/test.jpg";
    // const char *outputVideoPath = "output.mp4";

    // 打開輸入影片
    cv::VideoCapture cap(inputVideoPath);
    // cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        printf("can't open openCV camera\n");
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, input_video_width);       //Setting the width of the video
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, input_video_height);     //Setting the height of the video//

    int codec = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));

    cap >> frame;
    cv::resize(frame, frame, cv::Size(input_video_width, input_video_height));
#endif
#ifdef _v4l2cap
    frame = v4l2Cam();
    // rgbImg = rgbImg1(rect);
#endif
// ===============================================================

#ifdef Write_Video__
    write_video(output_video_width, output_video_height, output_video_fps, "Output_video.mp4");
#endif

    if (!pose.Set_TFlite(model_path))
        return -1;

    pose.Calculate_Scale(frame, INPUT_WIDTH, INPUT_HEIGHT);

    clock_t start, end, start_invok, end_invok;
    double TF_invoke_time_used;


    while (1) {

        start = clock();


#ifdef _openCVcap
        cap >> frame;
#endif
#ifdef _v4l2cap
        frame = v4l2Cam();
        // rgbImg = rgbImg1(rect);
#endif

        pose.get_input_data_fp32(frame,
                         pose.input_data,
                         INPUT_HEIGHT, INPUT_WIDTH,
                         pose.mean, pose.scale,
                         pose.new_width, pose.new_height,
                         pose.top, pose.bottom, pose.left, pose.right);
        

        start_invok = clock();

        if (pose.interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Invoke failed!" << std::endl;
            return -1;
        }

        end_invok = clock();
        std::cout << "After invoke: " << ((double) (end_invok - start_invok)) / CLOCKS_PER_SEC << std::endl;

        std::vector<Object> yolov8_objects;
        pose.generate_proposals(pose.yolov8_output,
                        PROB_THRESHOLD,
                        yolov8_objects,
                        pose.scale_factor,
                        pose.top, pose.left);

        pose.nms(yolov8_objects, NMS_THRESHOLD_BBOX, NMS_THRESHOLD_LANE);

        Output_frame = pose.draw_objects(frame, yolov8_objects);

        end = clock();
        TF_invoke_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        cout << "Time taken: " << TF_invoke_time_used << "seconds"  << "\n\n" << endl;

#ifdef Write_Video__
        video_writer.write(Output_frame);
#endif

        cv::imshow ("window",Output_frame);

	    int key = cv::waitKey(30);              // 等待 30 毫秒
        if (key == 32) {                        // 空格鍵的 ASCII 代碼為 32
            std::cout << "jump out" << std::endl;
            break;
        }
    }

    // 關閉資源
#ifdef _openCVcap
    cap.release();
#endif
#ifdef Write_Video__
    video_writer.release();
#endif

    return 0;
}