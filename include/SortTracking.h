#pragma once
#ifndef SORTTRACKING_H_
#define SORTTRACKING_H_
#include "Hungarian.h"
#include "kalman.h"
#include "opencv2/opencv.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <iomanip> // to format image names using setw() and setfill()
//#include <io.h> // to check file existence using POSIX function access(). On Linux include <unistd.h>.
#include <unistd.h>
#include <set>

struct Object {
    int class_id;
    float score;
    cv::Rect box;
    std::vector<cv::Point3f> kpts;  // x, y, visibility
};

typedef struct TrackingBox
{
    int frame;
    int id;
    int class_id;
    float score;

    cv::Rect box;
    std::vector<cv::Point3f> kpts;

} TrackingBox;

class SORTTRACKING {

public:

    // global variables for counting
    #define CNUM 20

    SORTTRACKING();
    ~SORTTRACKING();
    std::vector<TrackingBox> TrackingResult(const std::vector<Object> &bboxes);

    std::vector<KalmanTracker> trackers;
private:

    int frame_count = 0;

    // 消失時間，未被偵測幾幀後刪除
	int max_age = 5;

    // 最小連續命中次數，才輸出
	int min_hits = 3;

    // IOU
    // 調大（例如 0.5）→ 要求 IOU 更高才算配對，ID 比較穩，但容易丟失/斷軌
    // 調小（例如 0.2~0.3）→ 容易配對，不易斷軌，但誤配風險↑
	double iouThreshold = 0.3;
	
	
	// variables used in the for-loop
	vector<Rect_<float>> predictedBoxes;
	vector<vector<double>> iouMatrix;
	vector<int> assignment;

	set<int> unmatchedDetections;
	set<int> unmatchedTrajectories;
	set<int> allItems;
	set<int> matchedItems;
	vector<cv::Point> matchedPairs;
	vector<TrackingBox> frameTrackingResult;
	unsigned int trkNum = 0;
	unsigned int detNum = 0;
    std::vector<TrackingBox> detData;
    cv::Mat h=Mat::ones(3,3,CV_64FC1);
    cv::Mat m=Mat::ones(3,1,CV_64FC1);
    double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt);
    double vehicle_dis(double x_data, double y_data, double z_data);
};

#endif // !MOBILEFACENET_H_