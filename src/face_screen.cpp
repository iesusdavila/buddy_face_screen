#include "face_screen.hpp"
#include <filesystem>
#include <algorithm>
#include <cmath>

namespace fs = std::filesystem;

VideoSynchronizer::VideoSynchronizer() : Node("face_screen")
{
    std::string buddyShareDir = ament_index_cpp::get_package_share_directory("buddy_face_screen");
    
    std::string eyesFramesDir = buddyShareDir + "/imgs_transition/parpadear";
    std::string mouthFramesDir = buddyShareDir + "/imgs_transition/hablar";
    
    // Imágenes iniciales y finales
    std::string eyesOpenImgDir = buddyShareDir + "/imgs/ojos_abiertos.png";
    std::string eyesClosedImgDir = buddyShareDir + "/imgs/ojos_cerrados.png";
    std::string mouthClosedImgDir = buddyShareDir + "/imgs/boca_cerrada.png";
    std::string mouthOpenImgDir = buddyShareDir + "/imgs/boca_abierta.png";

    // Cargar imágenes
    eyesOpenImg = cv::imread(eyesOpenImgDir, cv::IMREAD_UNCHANGED);
    eyesClosedImg = cv::imread(eyesClosedImgDir, cv::IMREAD_UNCHANGED);
    mouthClosedImg = cv::imread(mouthClosedImgDir, cv::IMREAD_UNCHANGED);
    mouthOpenImg = cv::imread(mouthOpenImgDir, cv::IMREAD_UNCHANGED);
    
    // Convertir imágenes a RGBA si es necesario
    cv::cvtColor(eyesOpenImg, eyesOpenImg, cv::COLOR_BGRA2RGBA);
    cv::cvtColor(eyesClosedImg, eyesClosedImg, cv::COLOR_BGRA2RGBA);
    cv::cvtColor(mouthClosedImg, mouthClosedImg, cv::COLOR_BGRA2RGBA);
    cv::cvtColor(mouthOpenImg, mouthOpenImg, cv::COLOR_BGRA2RGBA);
    
    // Cargar frames de ojos y boca
    loadFrames(eyesFramesDir, eyesFrames);
    loadFrames(mouthFramesDir, mouthFrames);
    
    // Variables de control
    ttsActive = false;
    lastBlinkTime = std::chrono::system_clock::now();
    blinkInterval = 8.0;
    running = true;
    
    // Suscripción al topic de audio
    ttsSubscription = this->create_subscription<std_msgs::msg::Bool>(
        "/audio_playing",
        10,
        std::bind(&VideoSynchronizer::audioPlayingCallback, this, std::placeholders::_1));

    // Publicador para la pantalla facial
    faceScreenPublisher = this->create_publisher<sensor_msgs::msg::Image>("/face_screen", 10);
    
    // Iniciar el hilo de renderizado
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
    
    // Obtener todos los archivos .png en el directorio
    std::vector<std::string> framePaths;
    for (const auto& entry : fs::directory_iterator(framesDir))
    {
        if (entry.path().extension() == ".png" && entry.path().filename().string().find("frame_") != std::string::npos)
        {
            framePaths.push_back(entry.path().string());
        }
    }
    
    // Ordenar por nombre
    std::sort(framePaths.begin(), framePaths.end());
    
    // Cargar cada frame
    for (const auto& path : framePaths)
    {
        cv::Mat frame = cv::imread(path, cv::IMREAD_UNCHANGED);
        cv::Mat rgbaFrame;
        cv::cvtColor(frame, rgbaFrame, cv::COLOR_BGRA2RGBA);
        frames.push_back(rgbaFrame);
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
    
    // Si han pasado 8 segundos desde el último parpadeo, inicia un nuevo parpadeo
    if (timeSinceLastBlink >= blinkInterval)
    {
        lastBlinkTime = currentTime;
        return "blinking";
    }
    // Si estamos dentro de 0.5 segundos después del parpadeo, seguimos en secuencia de parpadeo
    else if (timeSinceLastBlink < 0.5)
    {
        return "blinking";
    }
    
    // En otro caso, ojos abiertos
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
        // Determinar en qué parte de la secuencia de parpadeo estamos
        auto currentTime = std::chrono::system_clock::now();
        double timeInBlink = std::chrono::duration<double>(currentTime - lastBlinkTime).count();
        int fps = 60;
        int frameCount = static_cast<int>(eyesFrames.size());

        double blinkDuration = frameCount * 1.25 / fps; // Duración total del parpadeo en segundos
        
        if (timeInBlink >= blinkDuration)
        {
            return eyesOpenImg;
        }
        
        double progress = timeInBlink / blinkDuration;
        
        // Secuencia: abierto -> cerrado -> abierto
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
    
    // Por defecto retornar ojos abiertos
    return eyesOpenImg;
}

cv::Mat VideoSynchronizer::getCurrentMouthFrame()
{
    if (!ttsActive)
    {
        return mouthClosedImg;
    }

    // Configuración de tiempos (ajusta estos valores según necesites)
    double transitionDuration = 0.6;  // Duración de apertura/cierre en segundos (X)
    double holdOpenDuration = 0.2;    // Tiempo con boca abierta
    double holdClosedDuration = 0.2;  // Tiempo con boca cerrada al final

    // Calcular ciclo total
    double mouthCycleTime = (2 * transitionDuration) + holdOpenDuration + holdClosedDuration;
    auto currentTime = std::chrono::system_clock::now();
    double secondsSinceEpoch = std::chrono::duration<double>(currentTime.time_since_epoch()).count();
    double timeInCycle = std::fmod(secondsSinceEpoch, mouthCycleTime) / mouthCycleTime;

    int frameCount = static_cast<int>(mouthFrames.size());
    if (frameCount == 0)
    {
        return mouthClosedImg;
    }

    // Definir límites de las fases
    double openingEnd = transitionDuration / mouthCycleTime;
    double holdOpenEnd = (transitionDuration + holdOpenDuration) / mouthCycleTime;
    double closingEnd = (transitionDuration + holdOpenDuration + transitionDuration) / mouthCycleTime;

    if (timeInCycle < openingEnd)  // Fase de apertura
    {
        double phaseProgress = timeInCycle / openingEnd;
        double smoothProgress = easeInOut(phaseProgress);
        int frameIdx = static_cast<int>(smoothProgress * (frameCount - 1));
        frameIdx = std::min(frameIdx, frameCount - 1);
        return mouthFrames[frameIdx];
    }
    else if (timeInCycle < holdOpenEnd)  // Boca abierta
    {
        return mouthOpenImg;
    }
    else if (timeInCycle < closingEnd)  // Fase de cierre
    {
        double phaseProgress = (timeInCycle - holdOpenEnd) / (closingEnd - holdOpenEnd);
        double smoothProgress = easeInOut(phaseProgress);
        int frameIdx = static_cast<int>((1 - smoothProgress) * (frameCount - 1));
        frameIdx = std::min(std::max(frameIdx, 0), frameCount - 1);
        return mouthFrames[frameIdx];
    }
    else  // Boca cerrada al final
    {
        return mouthClosedImg;
    }
}

double VideoSynchronizer::easeInOut(double x)
{
    // Función de suavizado cúbica: más lenta al inicio y al final, más rápida en el medio
    if (x < 0.5)
    {
        return 2 * x * x;
    }
    else
    {
        return 1 - std::pow(-2 * x + 2, 2) / 2;
    }
}

cv::Mat VideoSynchronizer::combineFrames(const cv::Mat& eyesFrame, const cv::Mat& mouthFrame)
{
    cv::Mat result = eyesFrame.clone();
    
    // Combinar solo donde la máscara alpha de la boca es > 0
    for (int y = 0; y < mouthFrame.rows; y++)
    {
        for (int x = 0; x < mouthFrame.cols; x++)
        {
            cv::Vec4b mouthPixel = mouthFrame.at<cv::Vec4b>(y, x);
            
            // Si el canal alpha de la boca es > 0, usar el píxel de la boca
            if (mouthPixel[3] > 0)
            {
                result.at<cv::Vec4b>(y, x) = mouthPixel;
            }
        }
    }
    
    return result;
}

void VideoSynchronizer::renderLoop()
{
    rclcpp::Rate rate(30);  // 30 Hz
    
    while (running && rclcpp::ok())
    {
        // Obtener estados actuales
        std::string eyesState = getEyesState();
        cv::Mat eyesFrame = getCurrentEyeFrame(eyesState);
        cv::Mat mouthFrame = getCurrentMouthFrame();
        
        // Combinar frames
        cv::Mat combinedFrame = combineFrames(eyesFrame, mouthFrame);
        
        // Guardar el frame actual para posible grabación
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            currentFrame = combinedFrame.clone();
        }
        
        // Convertir RGBA a BGR para ROS
        cv::Mat rosFrame;
        cv::cvtColor(combinedFrame, rosFrame, cv::COLOR_RGBA2BGR);
        
        // Convertir a mensaje de imagen ROS
        try
        {
            sensor_msgs::msg::Image::SharedPtr imgMsg = cv_bridge::CvImage(
                std_msgs::msg::Header(), "bgr8", rosFrame).toImageMsg();
            
            imgMsg->header.stamp = this->now();
            imgMsg->header.frame_id = "face_frame";
            
            // Publicar la imagen
            faceScreenPublisher->publish(*imgMsg);
            RCLCPP_DEBUG(this->get_logger(), "Frame publicado en face_screen");
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "Error al publicar imagen: %s", e.what());
        }
        
        // Control de FPS
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