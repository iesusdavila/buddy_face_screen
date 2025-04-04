import cv2
import numpy as np
from PIL import Image
import imageio
from united_morph_transition import morph_transition
from transition import save_frames_to_folder
from transformation import read_points_from_file
from moviepy import ImageClip, concatenate_videoclips

# def generar_transicion_ojos(imagen_inicial, imagen_final, puntos_inicial, puntos_final, 
#                            num_frames=10, fps=24, output_video="transicion_ojos.mp4"):    # Cargar imágenes con transparencia
#     img_abiertos = cv2.imread(imagen_inicial, cv2.IMREAD_UNCHANGED)
#     img_cerrados = cv2.imread(imagen_final, cv2.IMREAD_UNCHANGED)

#     clips = []

#     # Generar frames de transición
#     frames = morph_transition(
#         imagen_inicial,
#         imagen_final,
#         puntos_inicial,
#         puntos_final,
#         num_images=num_frames
#     )

#     print(frames)

#     save_frames_to_folder(frames, "video_generado")
#     clips.extend([ImageClip(np.array(frame), duration=10) for frame in frames])

#     transition_clip = concatenate_videoclips(clips, method="compose")
#     reverse_clip = concatenate_videoclips(clips[::-1], method="compose")
#     final_clip = concatenate_videoclips([transition_clip, reverse_clip], method="compose")
    
#     # Escribir el video final
#     final_clip.write_videofile(
#         output_video,
#         fps=fps,
#         codec="libx264",
#         audio_codec="aac",
#         logger=None
#     )
    
#     print(f"Video generado exitosamente: {output_video}")
#     return final_clip
#     # return concatenate_videoclips(clips, method="compose")


# def generar_transicion_ojos(imagen_inicial, imagen_final, puntos_inicial, puntos_final, 
#                            num_frames=10, fps=24, output_video="transicion_ojos.mp4"):
#     # Generar frames de transición
#     frames = morph_transition(
#         imagen_inicial,
#         imagen_final,
#         puntos_inicial,
#         puntos_final,
#         num_images=num_frames
#     )
    
#     # Calcular duración total deseada (segundos)
#     duracion_total = (num_frames * 2) / fps  # Ida y vuelta
    
#     # Calcular duración por frame
#     frame_duration = duracion_total / (len(frames))
    
#     # Crear clips con duración precisa
#     clips = [ImageClip(np.array(frame), duration=frame_duration) for frame in frames]
    
#     # Crear transición de ida y vuelta
#     transicion = concatenate_videoclips(clips, method="compose")
#     transicion_inversa = concatenate_videoclips(clips[::-1], method="compose")
#     video_final = concatenate_videoclips([transicion, transicion_inversa], method="compose")
    
#     # Ajustar FPS y escribir video
#     # video_final = video_final.set_fps(fps)
#     video_final.write_videofile(
#         output_video,
#         fps=fps,
#         codec="libx264", 
#         audio_codec="aac",
#         logger=None,
#         # preset='fast'
#     )
    
#     print(f"Video generado: {output_video} - Duración: {video_final.duration:.2f}s")

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
    save_frames_to_folder(frames_transicion, "imagenes_transicion")

    # Calcular duraciones
    duracion_transicion = len(frames_transicion)/fps
    frame_duration = 1/fps

    print(img_inicial)
    # Crear clips
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
        # preset='fast',
        logger=None
    )

    print(f"Video generado: {output_video}\nDuración total: {video_final.duration:.2f}s")
    return video_final

# Configuración
imagen_ojos_abiertos = "./hola/normal_ojos_abiertos.png"
imagen_ojos_cerrados = "./hola/normal_ojos_cerrados.png"
puntos_abiertos = "puntos_abiertos.txt"
puntos_cerrados = "puntos_cerrados.txt"

generar_transicion_ojos(
    
    imagen_ojos_abiertos,
    imagen_ojos_cerrados,
    puntos_abiertos,
    puntos_cerrados,
    num_frames=6,
    fps=24,
    output_video="parpadeo_suave.mp4"
)