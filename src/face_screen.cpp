#include "face_screen.hpp"
#include <filesystem>
#include <algorithm>
#include <cmath>

namespace fs = std::filesystem;

VideoSynchronizer::VideoSynchronizer() : Node("face_screen")
{
    std::string cocoShareDir = ament_index_cpp::get_package_share_directory("coco_face_screen");

    eyesOpenImg = cv::imread(cocoShareDir + "/imgs/open_eyes.png", cv::IMREAD_UNCHANGED);
    eyesClosedImg = cv::imread(cocoShareDir + "/imgs/close_eyes.png", cv::IMREAD_UNCHANGED);
    mouthClosedImg = cv::imread(cocoShareDir + "/imgs/close_mouth.png", cv::IMREAD_UNCHANGED);
    mouthOpenImg = cv::imread(cocoShareDir + "/imgs/open_mouth.png", cv::IMREAD_UNCHANGED);
    
    loadFrames(cocoShareDir + "/imgs_transition/blinking_frames", eyesFrames);
    loadFrames(cocoShareDir + "/imgs_transition/talking", mouthFrames);
    
    ttsActive = false;
    lastBlinkTime = std::chrono::system_clock::now();
    isBlinking = false;
    blinkStartTime = std::chrono::system_clock::now();
    running = true;
    
    ttsSubscription = this->create_subscription<std_msgs::msg::Bool>(
        "/audio_playing",
        10,
        std::bind(&VideoSynchronizer::audioPlayingCallback, this, std::placeholders::_1));

    faceScreenPublisher = this->create_publisher<sensor_msgs::msg::Image>("/face_screen", 10);
    
    renderThread = std::thread(&VideoSynchronizer::renderLoop, this);
    
    RCLCPP_INFO(this->get_logger(), "Initialized Video Synchronizer Node");
}

VideoSynchronizer::~VideoSynchronizer()
{
    shutdown();
}

void VideoSynchronizer::loadFrames(const std::string& framesDir, std::vector<cv::Mat>& frames)
{
    frames.clear();
    
    std::vector<std::string> filenames;
    for (const auto& entry : fs::directory_iterator(framesDir))
    {
        if (entry.path().extension() == ".png" && entry.path().filename().string().find("frame_") != std::string::npos)
        {
            filenames.push_back(entry.path().string());
        }
    }
    
    std::sort(filenames.begin(), filenames.end());
    
    for (const auto& filename : filenames)
    {
        cv::Mat frame = cv::imread(filename, cv::IMREAD_UNCHANGED);
        if (!frame.empty())
        {
            frames.push_back(std::move(frame));
        }
    }
    
    RCLCPP_INFO(this->get_logger(), "Loaded %zu frames from %s", frames.size(), framesDir.c_str());
}

void VideoSynchronizer::audioPlayingCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
    ttsActive = msg->data;
    RCLCPP_INFO(this->get_logger(), "TTS estado: %s", ttsActive ? "active" : "inactive");
}

std::string VideoSynchronizer::getEyesState()
{
    auto currentTime = std::chrono::system_clock::now();
    double timeSinceLastBlink = std::chrono::duration<double>(currentTime - lastBlinkTime).count();
    
    if (!isBlinking && timeSinceLastBlink >= BLINK_INTERVAL)
    {
        isBlinking = true;
        blinkStartTime = currentTime;
        lastBlinkTime = currentTime;
        return "blinking";
    }
    
    if (isBlinking)
    {
        double timeInBlink = std::chrono::duration<double>(currentTime - blinkStartTime).count();
        if (timeInBlink < BLINK_DURATION)
        {
            return "blinking";
        }
        else
        {
            isBlinking = false;
            return "open";
        }
    }
    
    return "open";
}

cv::Mat VideoSynchronizer::getCurrentEyeFrame(const std::string& eyesState)
{
    if (eyesState == "open" || eyesFrames.empty()) 
    {
        return eyesOpenImg;
    }
    
    auto currentTime = std::chrono::system_clock::now();
    double timeInBlink = std::chrono::duration<double>(currentTime - blinkStartTime).count();
    
    if (timeInBlink >= BLINK_DURATION)
    {
        return eyesOpenImg;
    }
    
    double progress = timeInBlink / BLINK_DURATION;
    progress = std::max(0.0, std::min(progress, 1.0));
    
    double animationProgress;
    if (progress < 0.5)
    {
        animationProgress = progress * 2.0;
    }
    else
    {
        animationProgress = 2.0 - (progress * 2.0);
    }
    
    animationProgress = easeInOut(animationProgress);
    
    int frameIdx = static_cast<int>(animationProgress * (eyesFrames.size() - 1));
    frameIdx = std::max(0, std::min(frameIdx, static_cast<int>(eyesFrames.size() - 1)));
    
    return eyesFrames[frameIdx];
}

cv::Mat VideoSynchronizer::getCurrentMouthFrame()
{
    if (!ttsActive || mouthFrames.empty()) 
    {
        return mouthOpenImg;
    }

    auto currentTime = std::chrono::system_clock::now();
    double secondsSinceEpoch = std::chrono::duration<double>(currentTime.time_since_epoch()).count();
    double timeInCycle = std::fmod(secondsSinceEpoch, mouthCycleTime);
    double cycleProgress = timeInCycle / mouthCycleTime;

    int frameCount = static_cast<int>(mouthFrames.size());
    
    if (cycleProgress < openingEnd) 
    {
        double phaseProgress = cycleProgress / openingEnd;
        double smoothProgress = easeInOut(phaseProgress);
        int frameIdx = static_cast<int>(smoothProgress * (frameCount - 1));
        frameIdx = std::min(frameIdx, frameCount - 1);
        return mouthFrames[frameIdx];
    }
    else if (cycleProgress < holdOpenEnd) 
    {
        return mouthClosedImg;
    }
    else if (cycleProgress < closingEnd)
    {
        double phaseProgress = (cycleProgress - holdOpenEnd) / (closingEnd - holdOpenEnd);
        double smoothProgress = easeInOut(phaseProgress);
        int frameIdx = static_cast<int>((1.0 - smoothProgress) * (frameCount - 1));
        frameIdx = std::max(0, std::min(frameIdx, frameCount - 1));
        return mouthFrames[frameIdx];
    }
    else
    {
        return mouthOpenImg;
    }
}

double VideoSynchronizer::easeInOut(double x)
{
    return x < 0.5 ? 2 * x * x : 1 - std::pow(-2 * x + 2, 2) / 2;
}

cv::Mat VideoSynchronizer::combineFrames(const cv::Mat& eyesFrame, const cv::Mat& mouthFrame) 
{
    if (eyesFrame.empty() || mouthFrame.empty())
    {
        RCLCPP_WARN(this->get_logger(), "Empty frame detected");
        return eyesFrame.empty() ? mouthFrame : eyesFrame;
    }
    
    cv::Mat result;
    if (eyesFrame.size() != mouthFrame.size())
    {
        cv::Mat mouthResized;
        cv::resize(mouthFrame, mouthResized, eyesFrame.size());
        cv::addWeighted(eyesFrame, 1.0, mouthResized, 1.0, 0.0, result);
    }
    else
    {
        cv::addWeighted(eyesFrame, 1.0, mouthFrame, 1.0, 0.0, result);
    }
    
    return result;
}

void VideoSynchronizer::renderLoop()
{
    rclcpp::Rate rate(30);
    
    while (running && rclcpp::ok())
    {
        std::lock_guard<std::mutex> lock(frameMutex);
        
        std::string eyesState = getEyesState();
        cv::Mat eyesFrame = getCurrentEyeFrame(eyesState);
        cv::Mat mouthFrame = getCurrentMouthFrame();
        
        cv::Mat combinedFrame = combineFrames(eyesFrame, mouthFrame);
        
        if (!combinedFrame.empty())
        {
            try
            {
                sensor_msgs::msg::Image::SharedPtr imgMsg = cv_bridge::CvImage(
                    std_msgs::msg::Header(), "bgra8", combinedFrame).toImageMsg();
                
                imgMsg->header.stamp = this->now();
                imgMsg->header.frame_id = "face_frame";
                
                faceScreenPublisher->publish(*imgMsg);
            }
            catch (const std::exception& e)
            {
                RCLCPP_ERROR(this->get_logger(), "Error to convert OpenCV image to ROS message: %s", e.what());
            }
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