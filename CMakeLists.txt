cmake_minimum_required(VERSION 3.8)
project(coco_face_screen)

# ========== Configuración para C++ ==========
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

# Configuración del ejecutable C++
add_executable(face_screen src/face_screen.cpp)
ament_target_dependencies(face_screen 
  rclcpp 
  sensor_msgs 
  std_msgs
  cv_bridge
)

target_include_directories(face_screen PUBLIC
  include/
)

# ========== Instalación común ==========
install(DIRECTORY
  imgs
  imgs_transition
  points
  scripts
  src
  videos
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS 
  face_screen
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
)

ament_export_include_directories(include)
ament_export_libraries(face_screen)
ament_package()
