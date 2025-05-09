#include "face_screen.hpp"
#include <filesystem>
#include <algorithm>
#include <cmath>

namespace fs = std::filesystem;

VideoSynchronizer::VideoSynchronizer() : Node("face_screen")
{
    std::string cocoShareDir = ament_index_cpp::get_package_share_directory("coco_face_screen");

    eyesOpenImg = cv::imread(cocoShareDir + "/imgs/ojos_abiertos.png", cv::IMREAD_UNCHANGED);
    eyesClosedImg = cv::imread(cocoShareDir + "/imgs/ojos_cerrados.png", cv::IMREAD_UNCHANGED);
    mouthClosedImg = cv::imread(cocoShareDir + "/imgs/boca_cerrada.png", cv::IMREAD_UNCHANGED);
    mouthOpenImg = cv::imread(cocoShareDir + "/imgs/boca_abierta.png", cv::IMREAD_UNCHANGED);
    
    loadFrames(cocoShareDir + "/imgs_transition/parpadear", eyesFrames);
    loadFrames(cocoShareDir + "/imgs_transition/hablar", mouthFrames);
    
    ttsActive = false;
    lastBlinkTime = std::chrono::system_clock::now();
    running = true;
    
    ttsSubscription = this->create_subscription<std_msgs::msg::Bool>(
        "/audio_playing",
        10,
        std::bind(&VideoSynchronizer::audioPlayingCallback, this, std::placeholders::_1));

    faceScreenPublisher = this->create_publisher<sensor_msgs::msg::Image>("/face_screen", 10);
    
    renderThread = std::thread(&VideoSynchronizer::renderLoop, this);
    
    RCLCPP_INFO(this->get_logger(), "Video Synchronizer iniciado");
}

VideoSynchronizer::~VideoSynchronizer()
{
    shutdown();
}

void VideoSynchronizer::loadFrames(const std::string& framesDir, std::vector<cv::Mat>& frames)
{
    frames.clear();
    
    for (const auto& entry : fs::directory_iterator(framesDir))
    {
        if (entry.path().extension() == ".png" && entry.path().filename().string().find("frame_") != std::string::npos)
        {
            cv::Mat frame = cv::imread(entry.path().string(), cv::IMREAD_UNCHANGED);
            frames.push_back(std::move(frame));
        }
    }
}

void VideoSynchronizer::audioPlayingCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
    ttsActive = msg->data;
    RCLCPP_INFO(this->get_logger(), "TTS estado: %s", ttsActive ? "activo" : "inactivo");
}

std::string VideoSynchronizer::getEyesState()
{
    auto currentTime = std::chrono::system_clock::now();
    double timeSinceLastBlink = std::chrono::duration<double>(currentTime - lastBlinkTime).count();
    
    if (timeSinceLastBlink >= BLINK_INTERVAL)
    {
        lastBlinkTime = currentTime;
        return "blinking";
    }
    
    if (timeSinceLastBlink < 0.5)
    {
        return "blinking";
    }
    
    return "open";
}

cv::Mat VideoSynchronizer::getCurrentEyeFrame(const std::string& eyesState)
{
    if (eyesState == "open")
    {
        return eyesOpenImg;
    }
    else if (eyesState == "blinking")
    {
        auto currentTime = std::chrono::system_clock::now();
        double timeInBlink = std::chrono::duration<double>(currentTime - lastBlinkTime).count();
        int fps = 60;
        int frameCount = static_cast<int>(eyesFrames.size());

        double blinkDuration = frameCount * 1.25 / fps; 
        
        if (timeInBlink >= blinkDuration)
        {
            return eyesOpenImg;
        }
        
        double progress = timeInBlink / blinkDuration;
        
        if (progress < 0.5)
        {
            int frameIdx = static_cast<int>(progress * 2 * frameCount);
            frameIdx = std::min(frameIdx, frameCount - 1);
            return eyesFrames[frameIdx];
        }
        else
        {
            int frameIdx = static_cast<int>((1.0 - progress) * 2 * frameCount);
            frameIdx = std::min(frameIdx, frameCount - 1);
            return eyesFrames[frameIdx];
        }
    }
    
    return eyesOpenImg;
}

cv::Mat VideoSynchronizer::getCurrentMouthFrame()
{
    if (!ttsActive)
    {
        return mouthClosedImg;
    }

    double transitionDuration = 0.6;  
    double holdOpenDuration = 0.2;    
    double holdClosedDuration = 0.2;  

    double mouthCycleTime = (2 * transitionDuration) + holdOpenDuration + holdClosedDuration;
    auto currentTime = std::chrono::system_clock::now();
    double secondsSinceEpoch = std::chrono::duration<double>(currentTime.time_since_epoch()).count();
    double timeInCycle = std::fmod(secondsSinceEpoch, mouthCycleTime) / mouthCycleTime;

    int frameCount = static_cast<int>(mouthFrames.size());
    if (frameCount == 0)
    {
        return mouthClosedImg;
    }

    double openingEnd = transitionDuration / mouthCycleTime;
    double holdOpenEnd = (transitionDuration + holdOpenDuration) / mouthCycleTime;
    double closingEnd = (transitionDuration + holdOpenDuration + transitionDuration) / mouthCycleTime;

    if (timeInCycle < openingEnd) 
    {
        double phaseProgress = timeInCycle / openingEnd;
        double smoothProgress = easeInOut(phaseProgress);
        int frameIdx = static_cast<int>(smoothProgress * (frameCount - 1));
        frameIdx = std::min(frameIdx, frameCount - 1);
        return mouthFrames[frameIdx];
    }
    else if (timeInCycle < holdOpenEnd) 
    {
        return mouthOpenImg;
    }
    else if (timeInCycle < closingEnd)
    {
        double phaseProgress = (timeInCycle - holdOpenEnd) / (closingEnd - holdOpenEnd);
        double smoothProgress = easeInOut(phaseProgress);
        int frameIdx = static_cast<int>((1 - smoothProgress) * (frameCount - 1));
        frameIdx = std::min(std::max(frameIdx, 0), frameCount - 1);
        return mouthFrames[frameIdx];
    }
    else 
    {
        return mouthClosedImg;
    }
}

double VideoSynchronizer::easeInOut(double x)
{
    return (x < 0.5) ? (2 * x * x) : (1 - std::pow(-2 * x + 2, 2) / 2);
}

cv::Mat VideoSynchronizer::combineFrames(const cv::Mat& eyesFrame, const cv::Mat& mouthFrame) {
    cv::Mat result;
    cv::addWeighted(eyesFrame, 1.0, mouthFrame, 1.0, 0.0, result);  // Mezcla alfa eficiente
    return result;
}

void VideoSynchronizer::renderLoop()
{
    std::lock_guard<std::mutex> lock(frameMutex);
    rclcpp::Rate rate(30); 
    
    while (running && rclcpp::ok())
    {
        std::string eyesState = getEyesState();
        cv::Mat eyesFrame = getCurrentEyeFrame(eyesState);
        cv::Mat mouthFrame = getCurrentMouthFrame();
        
        cv::Mat combinedFrame = combineFrames(eyesFrame, mouthFrame);
                
        try
        {
            sensor_msgs::msg::Image::SharedPtr imgMsg = cv_bridge::CvImage(
                std_msgs::msg::Header(), "bgra8", combinedFrame).toImageMsg();
            
            imgMsg->header.stamp = this->now();
            imgMsg->header.frame_id = "face_frame";
            
            faceScreenPublisher->publish(*imgMsg);
            RCLCPP_DEBUG(this->get_logger(), "Frame publicado en face_screen");
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "Error al publicar imagen: %s", e.what());
        }
        
        rate.sleep();
    }
}

void VideoSynchronizer::shutdown()
{
    running = false;
    
    if (renderThread.joinable())
    {
        renderThread.join();
    }
}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VideoSynchronizer>();
    
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    return 0;
}