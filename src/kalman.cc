///////////////////////////////////////////////////////////////////////////////
// KalmanTracker.cpp: KalmanTracker Class Implementation Declaration

#include "kalman.h"
#include <cmath>

int KalmanTracker::kf_count = 0;

// 如果你的物體移動比較平穩且測量值噪聲較小，可以減少過程噪聲協方差矩陣的值，增加測量噪聲協方差矩陣的值。
// 如果你的物體移動不規則且測量值噪聲較大，可以增加過程噪聲協方差矩陣的值，減少測量噪聲協方差矩陣的值
// 調整測量噪聲協方差矩陣
// setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1));  // 增加測量噪聲
// setIdentity(kf.measurementNoiseCov, Scalar::all(1e-3));  // 減少測量噪聲
// 調整過程噪聲協方差矩陣
// setIdentity(kf.processNoiseCov, Scalar::all(1e-1));  // 增加過程噪聲
// setIdentity(kf.processNoiseCov, Scalar::all(1e-3));  // 減少過程噪聲


// initialize Kalman filter
void KalmanTracker::init_kf(StateType stateMat)
{
	int stateNum = 7;
	int measureNum = 4;
	kf = KalmanFilter(stateNum, measureNum, 0);

	measurement = Mat::zeros(measureNum, 1, CV_32F);

//	kf.transitionMatrix = *(Mat_<float>(stateNum, stateNum) <<
	kf.transitionMatrix = (Mat_<float>(stateNum, stateNum) <<
		1, 0, 0, 0, 1, 0, 0,
		0, 1, 0, 0, 0, 1, 0,
		0, 0, 1, 0, 0, 0, 1,
		0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 1);

	setIdentity(kf.measurementMatrix);
	//初始不確定度
	setIdentity(kf.errorCovPost, Scalar::all(1));

	// *過程噪聲*
	// 調大 → 更相信「運動模型」，更快追新位置，更靈敏、較不平滑
	// 調小 → 更不相信模型，更平滑、反應慢一點
	setIdentity(kf.processNoiseCov, Scalar::all(1e-2));
	
	// *量測噪聲*
	// 調大 → 更不信偵測，更平滑、反應慢
	// 調小 → 更信偵測，更靈敏
	setIdentity(kf.measurementNoiseCov, Scalar::all(1e-4));

	
	// initialize state vector with bounding box in [cx,cy,s,r] style
	kf.statePost.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
	kf.statePost.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
	kf.statePost.at<float>(2, 0) = stateMat.area();
	kf.statePost.at<float>(3, 0) = stateMat.width / stateMat.height;
}


// Predict the estimated bounding box.
StateType KalmanTracker::predict()
{
	// predict
	Mat p = kf.predict();
	m_age += 1;

	if (m_time_since_update > 0)
		m_hit_streak = 0;
	m_time_since_update += 1;

	StateType predictBox = get_rect_xysr(p.at<float>(0, 0), p.at<float>(1, 0), p.at<float>(2, 0), p.at<float>(3, 0));

	m_history.push_back(predictBox);
	return m_history.back();
}


// Update the state vector with observed bounding box.
void KalmanTracker::update(StateType stateMat, float score)
{
    this->score = score;  // 更新 score
    m_time_since_update = 0;
    m_history.clear();
    m_hits += 1;
    m_hit_streak += 1;

    // measurement
    measurement.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
    measurement.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
    measurement.at<float>(2, 0) = stateMat.area();
    measurement.at<float>(3, 0) = stateMat.width / stateMat.height;

    // update
    kf.correct(measurement);
}


// Return the current state vector
StateType KalmanTracker::get_state()
{
	Mat s = kf.statePost;
	return get_rect_xysr(s.at<float>(0, 0), s.at<float>(1, 0), s.at<float>(2, 0), s.at<float>(3, 0));
}


// Convert bounding box from [cx,cy,s,r] to [x,y,w,h] style.
StateType KalmanTracker::get_rect_xysr(float cx, float cy, float s, float r)
{
	float w = sqrt(s * r);
	float h = s / w;
	float x = (cx - w / 2);
	float y = (cy - h / 2);

	if (x < 0 && cx > 0)
		x = 0;
	if (y < 0 && cy > 0)
		y = 0;

	return StateType(x, y, w, h);
}
