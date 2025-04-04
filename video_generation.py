import cv2
import numpy as np
from united_morph_transition import morph_transition
from transition import save_frames_to_folder
from moviepy import ImageClip, concatenate_videoclips
import os

def generar_puntos_control(img_path, puntos_salida):
    """
    Función auxiliar para crear manualmente los puntos de control
    Ejecutar por separado para cada imagen necesaria
    """
    img = cv2.imread(img_path)
    puntos = []
    
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            puntos.append([x, y])
            cv2.circle(img, (x, y), 3, (0,0,255), -1)
            cv2.imshow('Imagen', img)
    
    cv2.imshow('Imagen', img)
    cv2.setMouseCallback('Imagen', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    np.savetxt(puntos_salida, np.array(puntos, dtype=int), fmt="%d")
    
    print(f"Puntos guardados en {puntos_salida}")
    return np.array(puntos)


def generar_transicion_ojos(imagen_inicial, imagen_final, puntos_inicial, puntos_final, 
                           num_frames=10, fps=24, tiempo_exposicion=0.5, 
                           output_video="transicion_ojos.mp4"):
    # Cargar imágenes originales
    img_inicial = cv2.imread(imagen_inicial, cv2.IMREAD_UNCHANGED)
    img_inicial = cv2.cvtColor(img_inicial, cv2.COLOR_BGRA2RGBA)
    img_final = cv2.imread(imagen_final, cv2.IMREAD_UNCHANGED)
    img_final = cv2.cvtColor(img_final, cv2.COLOR_BGRA2RGBA)

    # Generar frames de transición
    frames_transicion = morph_transition(
        imagen_inicial,
        imagen_final,
        puntos_inicial,
        puntos_final,
        num_images=num_frames
    )

    print(frames_transicion)
    # Guardar frames en una carpeta
    save_frames_to_folder(frames_transicion, "imagenes_transicion/parpadear")

    # Calcular duraciones
    duracion_transicion = len(frames_transicion)/fps
    frame_duration = 1/fps

    print(img_inicial)
    # Crear clipsimg_inicial
    clips = [
        ImageClip(np.array(img_inicial), duration=tiempo_exposicion*3),  # Imagen inicial
        *[ImageClip(np.array(frame), duration=frame_duration/2) for frame in frames_transicion],  # Transición
        ImageClip(np.array(img_final), duration=tiempo_exposicion),  # Imagen final
        *[ImageClip(np.array(frame), duration=frame_duration/2) for frame in reversed(frames_transicion)],  # Transición inversa,
        ImageClip(np.array(img_inicial), duration=tiempo_exposicion*3),  # Imagen inicial
    ]

    # Ensamblar video
    video_final = concatenate_videoclips(clips, method="compose")
    
    # Escribir archivo
    video_final.write_videofile(
        output_video,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        logger=None
    )

    print(f"Video generado: {output_video}\nDuración total: {video_final.duration:.2f}s")
    return video_final

# Configuración
imagen_ojos_abiertos = "./imgs/normal_ojos_abiertos.png"
imagen_ojos_cerrados = "./imgs/normal_ojos_cerrados.png"

puntos_abiertos = os.path.join("puntos/puntos_abiertos.txt")
puntos_cerrados = os.path.join("puntos/puntos_cerrados.txt")

# Generar puntos de control (ejecutar solo una vez)
if not os.path.exists(puntos_abiertos):
    generar_puntos_control(imagen_ojos_abiertos, puntos_abiertos)

if not os.path.exists(puntos_cerrados):
    generar_puntos_control(imagen_ojos_cerrados, puntos_cerrados)

generar_transicion_ojos(
    imagen_ojos_abiertos,
    imagen_ojos_cerrados,
    puntos_abiertos,
    puntos_cerrados,
    num_frames=6,
    fps=24,
    output_video="parpadeo_suave.mp4"
)