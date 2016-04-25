#include <opencv2/opencv.hpp>
#define PI 3.14159265f
#include <chrono>

cv::Mat edge_nms(cv::Mat &E, cv::Mat &O, int r, int s, float m, int nThreads);
bool edge_detect(const cv::Mat &im, cv::Mat &E, cv::Mat &O, std::string model_path);

// util functions
void vis_matrix(cv::Mat &m, char *window_name);
void gradient(cv::Mat &I, cv::Mat &gx, cv::Mat &gy);
cv::Mat signum(cv::Mat &src);

enum class TimeResolution
{
    MILLI_SEC,
    MICRO_SEC,
    INVALID,
};

/**
 * @brief VTimer Utility Class
 *
 */

class VTimer
{
public:
    // Classic API
    VTimer();
    void Restart();
    int64_t TimeSpan();

    // New API
    void Stop();
    int64_t GetDuration(TimeResolution tres);

private:
    std::chrono::time_point<std::chrono::system_clock> beg_;
    std::chrono::time_point<std::chrono::system_clock> end_;

    VTimer(const VTimer&);
    VTimer& operator=(const VTimer&);
};

inline VTimer::VTimer()
{
    Restart();
}

inline void VTimer::Restart()
{
    beg_ = std::chrono::high_resolution_clock::now();
}

inline int64_t VTimer::TimeSpan()
{
    Stop();
    return GetDuration(TimeResolution::MILLI_SEC);
}

inline void VTimer::Stop()
{
    end_ = std::chrono::high_resolution_clock::now();
}

inline int64_t VTimer::GetDuration(TimeResolution tres)
{
    switch (tres)
    {
    case TimeResolution::MILLI_SEC:
        return std::chrono::duration_cast<std::chrono::milliseconds>(end_ - beg_).count();
    case TimeResolution::MICRO_SEC:
        return std::chrono::duration_cast<std::chrono::microseconds>(end_ - beg_).count();
    default:
        return -1;
    }
}

