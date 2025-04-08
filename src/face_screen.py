#!/usr/bin/env python3

import cv2
import numpy as np
import os
import time
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import glob
from ament_index_python.packages import get_package_share_directory

class VideoSynchronizer(Node):
    def __init__(self):
        super().__init__('video_synchronizer')

        buddy_share_dir = get_package_share_directory('buddy_face_screen')
        
        eyes_frames_dir = os.path.join(buddy_share_dir, "imgs_transition", "parpadear")
        mouth_frames_dir = os.path.join(buddy_share_dir, "imgs_transition", "hablar")
        
        # Imágenes iniciales y finales
        eyes_open_img_dir = os.path.join(buddy_share_dir, "imgs", "ojos_abiertos.png")
        eyes_closed_img_dir = os.path.join(buddy_share_dir, "imgs", "ojos_cerrados.png")
        mouth_closed_img_dir = os.path.join(buddy_share_dir, "imgs", "boca_cerrada.png")
        mouth_open_img_dir = os.path.join(buddy_share_dir, "imgs", "boca_abierta.png")

        self.eyes_open_img = cv2.imread(eyes_open_img_dir, cv2.IMREAD_UNCHANGED)
        self.eyes_closed_img = cv2.imread(eyes_closed_img_dir, cv2.IMREAD_UNCHANGED)
        self.mouth_closed_img = cv2.imread(mouth_closed_img_dir, cv2.IMREAD_UNCHANGED)
        self.mouth_open_img = cv2.imread(mouth_open_img_dir, cv2.IMREAD_UNCHANGED)
        
        # Convertir imágenes a RGBA si es necesario
        self.eyes_open_img = cv2.cvtColor(self.eyes_open_img, cv2.COLOR_BGRA2RGBA)
        self.eyes_closed_img = cv2.cvtColor(self.eyes_closed_img, cv2.COLOR_BGRA2RGBA)
        self.mouth_closed_img = cv2.cvtColor(self.mouth_closed_img, cv2.COLOR_BGRA2RGBA)
        self.mouth_open_img = cv2.cvtColor(self.mouth_open_img, cv2.COLOR_BGRA2RGBA)
        
        # Cargar frames de ojos y boca
        self.eyes_frames = self.load_frames(eyes_frames_dir)
        self.mouth_frames = self.load_frames(mouth_frames_dir)
        
        # Variables de control
        self.tts_active = False
        self.last_blink_time = time.time()
        self.blink_interval = 8.0 
        self.running = True
        self.current_frame = None
        
        self.tts_subscription = self.create_subscription(
            Bool,
            '/audio_playing',
            self.audio_playing_callback,
            10)

        self.cv_bridge = CvBridge()

        self.face_screen = self.create_publisher(Image, '/face_screen', 10)
        
        # Iniciar el hilo de renderizado
        self.render_thread = threading.Thread(target=self.render_loop)
        self.render_thread.daemon = True
        self.render_thread.start()
        
        self.get_logger().info('Video Synchronizer iniciado')
    
    def load_frames(self, frames_dir):
        """Cargar todos los frames desde un directorio ordenados por nombre"""
        frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
        frames = []
        
        for path in frame_paths:
            frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
            frames.append(frame)
            
        return frames
    
    def audio_playing_callback(self, msg):
        """Callback para el topic /stt_terminado"""
        self.tts_active = msg.data 
        self.get_logger().info(f'TTS estado: {"activo" if msg.data else "inactivo"}')
    
    def get_eyes_state(self):
        """Determina el estado actual de los ojos basado en el tiempo"""
        current_time = time.time()
        time_since_last_blink = current_time - self.last_blink_time
        
        # Si han pasado 8 segundos desde el último parpadeo, inicia un nuevo parpadeo
        if time_since_last_blink >= self.blink_interval:
            self.last_blink_time = current_time
            return "blinking"
        
        # Si estamos dentro de 0.5 segundos después del parpadeo, seguimos en secuencia de parpadeo
        elif time_since_last_blink < 0.5:
            return "blinking"
        
        # En otro caso, ojos abiertos
        return "open"
    
    def get_current_eye_frame(self, eyes_state):
        """Obtiene el frame actual para los ojos"""
        if eyes_state == "open":
            return self.eyes_open_img
        elif eyes_state == "blinking":
            # Determinar en qué parte de la secuencia de parpadeo estamos
            time_in_blink = time.time() - self.last_blink_time
            fps = 60
            frame_count = len(self.eyes_frames)

            blink_duration = frame_count*1.25/(fps) # Duración total del parpadeo en segundos
            
            if time_in_blink >= blink_duration:
                return self.eyes_open_img
            
            progress = time_in_blink / blink_duration
            
            # Secuencia: abierto -> cerrado -> abierto
            if progress < 0.5:
                frame_idx = int(progress * 2 * frame_count)
                frame_idx = min(frame_idx, frame_count - 1)
                return self.eyes_frames[frame_idx]
            else:
                frame_idx = int((1.0 - progress) * 2 * frame_count)
                frame_idx = min(frame_idx, frame_count - 1)
                return self.eyes_frames[frame_idx]
    
    def get_current_mouth_frame(self):
        """Obtiene el frame actual para la boca con animación más fluida"""
        if not self.tts_active:
            return self.mouth_closed_img
        
        # Si TTS está activo, animar la boca con más fluidez
        mouth_cycle_time = 1.8 
        
        # Calcular en qué parte del ciclo estamos (valor entre 0 y 1)
        current_time = time.time()
        time_in_cycle = (current_time % mouth_cycle_time) / mouth_cycle_time
        
        frame_count = len(self.mouth_frames)
        
        if time_in_cycle < 0.3:  
            # Normalizar el progreso en esta fase (0-1)
            phase_progress = time_in_cycle / 0.3
            # Usar curva suavizada para movimiento más natural
            smooth_progress = self._ease_in_out(phase_progress)
            # Calcular el índice del frame
            frame_idx = int(smooth_progress * frame_count)
            frame_idx = min(frame_idx, frame_count - 1)
            return self.mouth_frames[frame_idx]
            
        elif time_in_cycle < 0.5: 
            return self.mouth_open_img
            
        elif time_in_cycle < 0.8: 
            # Normalizar el progreso en esta fase (0-1)
            phase_progress = (time_in_cycle - 0.5) / 0.3
            # Usar curva suavizada para movimiento más natural
            smooth_progress = self._ease_in_out(phase_progress)
            # Calcular el índice del frame (en orden inverso)
            frame_idx = int((1.0 - smooth_progress) * frame_count)
            frame_idx = min(max(frame_idx, 0), frame_count - 1)
            return self.mouth_frames[frame_idx]
            
        else: 
            return self.mouth_closed_img

    def _ease_in_out(self, x):
        """Función de suavizado para crear transiciones más naturales"""
        # Función de suavizado cúbica: más lenta al inicio y al final, más rápida en el medio
        if x < 0.5:
            return 2 * x * x
        else:
            return 1 - pow(-2 * x + 2, 2) / 2
    
    def combine_frames(self, eyes_frame, mouth_frame):
        """Combina los frames de los ojos y la boca en una sola imagen"""        
        result = np.copy(eyes_frame)
        mask = mouth_frame[:, :, 3] > 0
        result[mask] = mouth_frame[mask]
        return result
    
    def render_loop(self):
        """Bucle principal de renderizado que combina ojos y boca"""
        rate = self.create_rate(30) 
        
        while self.running and rclpy.ok():
            # Obtener estados actuales
            eyes_state = self.get_eyes_state()
            eyes_frame = self.get_current_eye_frame(eyes_state)
            mouth_frame = self.get_current_mouth_frame()
            
            # Combinar frames
            combined_frame = self.combine_frames(eyes_frame, mouth_frame)
            
            # Guardar el frame actual para posible grabación
            self.current_frame = combined_frame
            
            ros_frame = cv2.cvtColor(combined_frame, cv2.COLOR_RGBA2BGR)
            
            # Convertir a mensaje de imagen ROS
            try:
                img_msg = self.cv_bridge.cv2_to_imgmsg(ros_frame, encoding="bgr8")
                img_msg.header.stamp = self.get_clock().now().to_msg()
                img_msg.header.frame_id = "face_frame"
                img_msg.encoding = "bgr8"
                
                # Publicar la imagen
                self.face_screen.publish(img_msg)
                self.get_logger().debug('Frame publicado en face_screen')
            except Exception as e:
                self.get_logger().error(f'Error al publicar imagen: {str(e)}')
            
            # Control de FPS usando rate.sleep() de ROS
            rate.sleep()
    
    def shutdown(self):
        """Limpia los recursos antes de cerrar"""
        self.running = False
        if self.render_thread.is_alive():
            self.render_thread.join(timeout=1.0)
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    
    synchronizer = VideoSynchronizer()
    
    try:
        rclpy.spin(synchronizer)
    except KeyboardInterrupt:
        pass
    finally:
        synchronizer.shutdown()
        synchronizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()