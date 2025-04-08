cmake_minimum_required(VERSION 3.8)
project(buddy_face_screen)

# ========== Configuración para Python ==========
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Instalar el script Python
install(PROGRAMS
  src/face_screen.py
  DESTINATION lib/${PROJECT_NAME}
)

# ========== Instalación común ==========
install(DIRECTORY
  imgs
  imgs_transition
  points
  src
  videos
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(
  rclpy
  sensor_msgs
  std_msgs
)

ament_package()
