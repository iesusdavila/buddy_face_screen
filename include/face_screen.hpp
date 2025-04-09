#ifndef VIDEO_SYNCHRONIZER_HPP
#define VIDEO_SYNCHRONIZER_HPP

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <ament_index_cpp/get_package_share_directory.hpp>

class VideoSynchronizer : public rclcpp::Node
{
public:
    VideoSynchronizer();
    ~VideoSynchronizer();

private:
    // Funciones del nodo
    void loadFrames(const std::string& framesDir, std::vector<cv::Mat>& frames);
    void audioPlayingCallback(const std_msgs::msg::Bool::SharedPtr msg);
    std::string getEyesState();
    cv::Mat getCurrentEyeFrame(const std::string& eyesState);
    cv::Mat getCurrentMouthFrame();
    cv::Mat combineFrames(const cv::Mat& eyesFrame, const cv::Mat& mouthFrame);
    double easeInOut(double x);
    void renderLoop();
    void shutdown();

    // Imágenes de base
    cv::Mat eyesOpenImg;
    cv::Mat eyesClosedImg;
    cv::Mat mouthClosedImg;
    cv::Mat mouthOpenImg;
    
    // Frames de transición
    std::vector<cv::Mat> eyesFrames;
    std::vector<cv::Mat> mouthFrames;
    
    // Variables de estado
    bool ttsActive;
    std::chrono::time_point<std::chrono::system_clock> lastBlinkTime;
    double blinkInterval;
    bool running;
    cv::Mat currentFrame;
    std::mutex frameMutex;
    
    // Hilo de renderizado
    std::thread renderThread;
    
    // Publicadores y suscriptores ROS
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr ttsSubscription;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr faceScreenPublisher;
    
    // Bridge para conversión entre OpenCV y ROS
    cv_bridge::CvImage cvBridge;
};

#endif // VIDEO_SYNCHRONIZER_HPP