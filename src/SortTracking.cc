#include "SortTracking.h"

#include <unordered_map>
#include <algorithm>
#include <cmath>

using namespace std;

// ==============================
// Keypoint EMA filter (本檔案私有)
// - 以 track_id 為鍵，維護每條軌跡的關鍵點平滑狀態
// - 僅在 SortTracking.cc 內部使用，不修改任何 .h
// ==============================

struct _KptEMA {
    cv::Point2f pt{0.f, 0.f};
    bool initialized{false};
};

struct _TrackKptState {
    vector<_KptEMA> kpts;   // 每個關鍵點的 EMA 狀態
    int num_kpts{0};
    int missed{0};
};

// 追蹤 id（使用 KalmanTracker::m_id 原始值） -> 關鍵點平滑狀態
static std::unordered_map<int, _TrackKptState> __kpt_states;
// 當前幀輸出的平滑後 kpts，供輸出時讀取
static std::unordered_map<int, std::vector<cv::Point3f>> __kpt_outputs;

// 夠用又安全的 clamp
template <typename T> static inline T _clamp(T v, T lo, T hi) {
    return std::max(lo, std::min(hi, v));
}

// 以 conf 做自適應 EMA：conf 高 -> 反應快 (alpha 大)；conf 低 -> 更平滑 (alpha 小)
static inline float _conf_to_alpha(float conf, float conf_thr, float alpha_lo, float alpha_hi) {
    conf = _clamp(conf, 0.0f, 1.0f);
    float t = (conf <= conf_thr) ? 0.f : (conf - conf_thr) / (1.f - conf_thr);
    return alpha_lo + (alpha_hi - alpha_lo) * t;
}

// 更新/取得某條 track 的平滑後 kpts；
// - meas == nullptr 代表此幀沒有新觀測（unmatched），則輸出沿用上次平滑值（z 會降為 0）
// - meas != nullptr 則用 EMA 更新
// alpha_lo / alpha_hi：平滑強度上下限（小→更平滑、大→更靈敏）
// conf_thr：低於此分數視為低可信 → 使用較小 alpha（更平滑）
static std::vector<cv::Point3f> _update_kpts_ema(int track_id, const std::vector<cv::Point3f>* meas,
                                                float conf_thr=0.2f, float alpha_lo=0.15f, float alpha_hi=0.7f)
{
    auto &st = __kpt_states[track_id];

    // 初始化/調整長度
    if (meas && (int)meas->size() != st.num_kpts) {
        st.kpts.assign(meas->size(), _KptEMA{});
        st.num_kpts = (int)meas->size();
    } else if (!meas && st.num_kpts == 0) {
        // 沒有量測也沒有狀態，回空
        return {};
    }

    std::vector<cv::Point3f> out;
    out.resize(st.num_kpts, cv::Point3f(0.f, 0.f, 0.f));

    if (meas) {
        st.missed = 0;
        for (int i = 0; i < st.num_kpts; ++i) {
            const float mx = (*meas)[i].x;
            const float my = (*meas)[i].y;
            const float mc = (*meas)[i].z;  // keypoint conf

            // 僅對有效量測做更新（可依需求加入邊界檢查）
            if (mc > 0.f && mx > 0.f && my > 0.f) {
                float alpha = _conf_to_alpha(mc, conf_thr, alpha_lo, alpha_hi);
                if (!st.kpts[i].initialized) {
                    st.kpts[i].pt = cv::Point2f(mx, my);
                    st.kpts[i].initialized = true;
                } else {
                    st.kpts[i].pt.x = alpha * mx + (1.f - alpha) * st.kpts[i].pt.x;
                    st.kpts[i].pt.y = alpha * my + (1.f - alpha) * st.kpts[i].pt.y;
                }
                out[i] = cv::Point3f(st.kpts[i].pt.x, st.kpts[i].pt.y, mc);
            } else {
                // 量測無效，沿用上一幀（若尚未初始化則保持 (0,0,0)）
                if (st.kpts[i].initialized) {
                    out[i] = cv::Point3f(st.kpts[i].pt.x, st.kpts[i].pt.y, 0.f);
                }
            }
        }
    } else {
        // 沒有量測：保持 last（z=0），missed++
        st.missed += 1;
        for (int i = 0; i < st.num_kpts; ++i) {
            if (st.kpts[i].initialized) {
                out[i] = cv::Point3f(st.kpts[i].pt.x, st.kpts[i].pt.y, 0.f);
            }
        }
    }
    return out;
}

// 當追蹤器被刪除時，清掉對應的 kpt 狀態
static inline void _erase_kpt_state(int track_id) {
    __kpt_states.erase(track_id);
    __kpt_outputs.erase(track_id);
}

// ==============================
// 原有 SORTTRACKING 實作，僅插入 kpts 的處理
// ==============================

SORTTRACKING::SORTTRACKING() {}
SORTTRACKING::~SORTTRACKING() {}

// Computes IOU between two bounding boxes
double SORTTRACKING::GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;
    if (un < DBL_EPSILON) return 0;
    return (double)(in / un);
}

std::vector<TrackingBox> SORTTRACKING::TrackingResult(const std::vector<Object>& bboxes)
{
    int input_i = 0;

    // 將檢測結果轉為 TrackingBox，並把 kpts 一併放入 detData（原本沒放）
    for (const auto& obj : bboxes)
    {
        TrackingBox temp;
        temp.id = input_i;
        temp.class_id = obj.class_id;
        temp.box = Rect_<float>(Point_<float>(obj.box.x, obj.box.y),
                                Point_<float>(obj.box.width + obj.box.x, obj.box.height + obj.box.y));
        temp.score = obj.score;
        temp.kpts  = obj.kpts;  // <<< 新增：帶入該檢測的 keypoints
        detData.push_back(temp);
        input_i++;
    }

    input_i = 0;
    frame_count++;

    if (trackers.size() == 0) // 第一幀
    {
        // 用第一次檢測初始化 Kalman 追蹤器 + 初始化 kpt EMA 狀態
        for (unsigned int i = 0; i < detData.size(); i++)
        {
            KalmanTracker trk = KalmanTracker(detData[i].box, detData[i].score);
            trk.m_class = detData[i].class_id;
            // 先建立 tracker
            trackers.push_back(trk);

            // 初始化對應的 kpts 狀態（使用 KalmanTracker::m_id 作為 key）
            int tid = trk.m_id;
            auto out_k = _update_kpts_ema(tid, &detData[i].kpts);
            __kpt_outputs[tid] = std::move(out_k);
        }
    }
    else
    {
        // === 原有 predict ===
        predictedBoxes.clear();
        for (auto it = trackers.begin(); it != trackers.end();)
        {
            Rect_<float> pBox = (*it).predict();
            if (pBox.x >= 0 && pBox.y >= 0)
            {
                predictedBoxes.push_back(pBox);
                it++;
            }
            else
            {
                // 追蹤器無效 -> 一併清掉 kpt 狀態
                _erase_kpt_state((*it).m_id);
                it = trackers.erase(it);
            }
        }

        // === 原有 Hungarian 匹配（以 1 - IOU 作成本）===
        trkNum = predictedBoxes.size();
        detNum = detData.size();
        iouMatrix.clear();
        iouMatrix.resize(trkNum, vector<double>(detNum, 0));
        for (unsigned int i = 0; i < trkNum; i++)
            for (unsigned int j = 0; j < detNum; j++)
                iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detData[j].box);

        HungarianAlgorithm HungAlgo;
        assignment.clear();
        HungAlgo.Solve(iouMatrix, assignment);

        unmatchedTrajectories.clear();
        unmatchedDetections.clear();
        allItems.clear();
        matchedItems.clear();

        if (detNum > trkNum)
        {
            for (unsigned int n = 0; n < detNum; n++) allItems.insert(n);
            for (unsigned int i = 0; i < trkNum; ++i) matchedItems.insert(assignment[i]);
            set_difference(allItems.begin(), allItems.end(),
                           matchedItems.begin(), matchedItems.end(),
                           insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
        }
        else if (detNum < trkNum)
        {
            for (unsigned int i = 0; i < trkNum; ++i)
                if (assignment[i] == -1)
                    unmatchedTrajectories.insert(i);
        }

        matchedPairs.clear();
        for (unsigned int i = 0; i < trkNum; ++i)
        {
            if (assignment[i] == -1) continue;
            if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
            {
                unmatchedTrajectories.insert(i);
                unmatchedDetections.insert(assignment[i]);
            }
            else
                matchedPairs.push_back(cv::Point(i, assignment[i]));
        }

        // === 原有：更新匹配到的追蹤器 ===
        for (unsigned int i = 0; i < matchedPairs.size(); i++)
        {
            int trkIdx = matchedPairs[i].x;
            int detIdx = matchedPairs[i].y;

            trackers[trkIdx].update(detData[detIdx].box, detData[detIdx].score);

            // 新增：用此 detection 的 kpts 更新 EMA，並保存當前幀輸出
            int tid = trackers[trkIdx].m_id;
            auto out_k = _update_kpts_ema(tid, &detData[detIdx].kpts);
            __kpt_outputs[tid] = std::move(out_k);
        }

        // === 原有：未匹配的檢測 -> 新增追蹤器 ===
        for (auto umd : unmatchedDetections)
        {
            KalmanTracker tracker = KalmanTracker(detData[umd].box, detData[umd].score);
            tracker.m_class = detData[umd].class_id;
            int tid = tracker.m_id;  // 在 push_back 前即可取到 id（見 kalman.h 建構子）

            // 先初始化 kpts EMA，確保本幀就有輸出
            auto out_k = _update_kpts_ema(tid, &detData[umd].kpts);
            __kpt_outputs[tid] = std::move(out_k);

            trackers.push_back(tracker);
        }

        // === 新增：未匹配的軌跡（本幀沒有量測）也要生成 kpts 輸出（沿用 last, z=0）===
        for (auto utrk : unmatchedTrajectories)
        {
            if (utrk >= 0 && utrk < (int)trackers.size()) {
                int tid = trackers[utrk].m_id;
                auto out_k = _update_kpts_ema(tid, nullptr);
                __kpt_outputs[tid] = std::move(out_k);
            }
        }
    }

    // === 原有：彙整輸出（這裡把平滑後的 kpts 帶進去）===
    frameTrackingResult.clear();
    for (auto it = trackers.begin(); it != trackers.end();)
    {
        if (((*it).m_time_since_update < 1) &&
            ((*it).m_hit_streak >= min_hits || frame_count <= min_hits))
        {
            TrackingBox res;
            res.box = (*it).get_state();
            res.id = (*it).m_id + 1;     // 保持原有 +1 輸出
            res.class_id = (*it).m_class;
            res.frame = frame_count;
            res.score = (*it).score;

            // 取出上一段流程存好的平滑後 kpts
            auto found = __kpt_outputs.find((*it).m_id);
            if (found != __kpt_outputs.end())
                res.kpts = found->second;   // 平滑後
            else
                res.kpts.clear();

            frameTrackingResult.push_back(res);
            it++;
        }
        else
        {
            it++;
        }

        // 原有：刪除過時追蹤器；新增：同步刪除 kpt 狀態
        if (it != trackers.end() && (*it).m_time_since_update > max_age) {
            _erase_kpt_state((*it).m_id);
            it = trackers.erase(it);
        }
    }

    detData.clear();
    return frameTrackingResult;
}
