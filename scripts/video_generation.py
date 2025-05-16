import cv2
import numpy as np
from united_morph_transition import MorphTransition
from moviepy import ImageClip, concatenate_videoclips
import os

def save_frames_to_folder(frames, output_frames_dir):
    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir)

    for i in range(len(frames)):
        frame = frames[i]
        frame_output_path = os.path.join(output_frames_dir, f"frame_{i:04d}.png")
        frame.save(frame_output_path, format="PNG")
        
    print(f"Frames guardados en la carpeta: {output_frames_dir}")

def generar_puntos_control(img_path, puntos_salida):
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
    
    np.savetxt(puntos_salida, np.array(puntos, dtype=int), fmt="%d", delimiter=", ")
    
    print(f"Puntos guardados en {puntos_salida}")
    return np.array(puntos)

def generar_transicion_ojos(imagen_inicial, imagen_final, puntos_inicial, puntos_final, 
                           num_frames=10, fps=24, tiempo_exposicion=0.5, 
                           output_video=None, folder_frames=None):
    img_inicial = cv2.imread(imagen_inicial, cv2.IMREAD_UNCHANGED)
    img_inicial = cv2.cvtColor(img_inicial, cv2.COLOR_BGRA2RGBA)
    img_final = cv2.imread(imagen_final, cv2.IMREAD_UNCHANGED)
    img_final = cv2.cvtColor(img_final, cv2.COLOR_BGRA2RGBA)

    frames_transicion = MorphTransition(
        imagen_inicial,
        imagen_final,
        puntos_inicial,
        puntos_final,
        num_images=num_frames
    ).morph_transition()

    if folder_frames is not None:
        save_frames_to_folder(frames_transicion, folder_frames)

    if output_video is not None:
        duracion_transicion = len(frames_transicion)/fps
        frame_duration = 1/fps

        clips = [
            ImageClip(np.array(img_inicial), duration=tiempo_exposicion*3),  
            *[ImageClip(np.array(frame), duration=frame_duration) for frame in frames_transicion],  
            ImageClip(np.array(img_final), duration=tiempo_exposicion*0.75),  
            *[ImageClip(np.array(frame), duration=frame_duration) for frame in reversed(frames_transicion)],
            ImageClip(np.array(img_inicial), duration=tiempo_exposicion*3),  
        ]

        video_final = concatenate_videoclips(clips, method="compose")
        
        video_final.write_videofile(
            output_video,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            logger=None
        )

        print(f"Video generado: {output_video}\nDuración total: {video_final.duration:.2f}s")

imagen_boca_abierta = "../imgs/boca_abierta.png"
imagen_boca_cerrada = "../imgs/boca_cerrada.png"

puntos_abiertos = os.path.join("../points/boca_abierta.txt")
puntos_cerrados = os.path.join("../points/boca_cerrada.txt")

name_output_video = "hablar_suave.mp4"
folder_frames = "imagenes_transicion/parpadear"

if not os.path.exists(puntos_abiertos):
    generar_puntos_control(imagen_boca_abierta, puntos_abiertos)

if not os.path.exists(puntos_cerrados):
    generar_puntos_control(imagen_boca_cerrada, puntos_cerrados)

generar_transicion_ojos(
    imagen_boca_cerrada,
    imagen_boca_abierta,
    puntos_cerrados,
    puntos_abiertos,
    num_frames=5,
    fps=60,
    output_video=name_output_video,
    folder_frames=folder_frames
)
