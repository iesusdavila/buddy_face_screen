import cv2
import numpy as np
import os
from moviepy import ImageClip, concatenate_videoclips
from united_morph_transition import morph_transition


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
    
    # np.savetxt(puntos_salida, np.array(puntos))
    np.savetxt(puntos_salida, np.array(puntos, dtype=int), fmt="%d")
    
    print(f"Puntos guardados en {puntos_salida}")
    return np.array(puntos)

def morph_avanzado(img1, img2, puntos1, puntos2, num_frames=30, piramide_niveles=3):
    # Alinear imágenes
    transform, _ = cv2.estimateAffinePartial2D(puntos2, puntos1)
    img2_aligned = cv2.warpAffine(img2, transform, (img1.shape[1], img1.shape[0]))
    
    # Construir pirámides
    gaussiana1 = [img1.astype(np.float32)]
    gaussiana2 = [img2_aligned.astype(np.float32)]
    
    for _ in range(piramide_niveles):
        img1_pyr = cv2.pyrDown(gaussiana1[-1])
        img2_pyr = cv2.pyrDown(gaussiana2[-1])
        gaussiana1.append(img1_pyr)
        gaussiana2.append(img2_pyr)
    
    # Generar frames
    frames = []
    for t in np.linspace(0, 1, num_frames):
        blended_pyr = []
        for nivel in range(piramide_niveles, -1, -1):
            img1_resized = gaussiana1[nivel]
            img2_resized = gaussiana2[nivel]
            
            # Calcular pesos
            peso1 = np.cos(t * np.pi/2)  # Función de easing
            peso2 = np.sin(t * np.pi/2)
            
            # Mezclar nivel de la pirámide
            blended = cv2.addWeighted(img1_resized, peso1, img2_resized, peso2, 0)
            blended_pyr.append(blended)
        
        # Reconstruir imagen
        reconstruida = blended_pyr[0]
        for i in range(1, len(blended_pyr)):
            reconstruida = cv2.pyrUp(reconstruida)
            h, w = blended_pyr[i].shape[:2]
            reconstruida = cv2.resize(reconstruida, (w, h)) + blended_pyr[i]
        
        # Normalizar y convertir
        frame = np.clip(reconstruida, 0, 255).astype(np.uint8)
        frames.append(frame)
    
    return frames

def crear_transicion_suave(imagen_a, imagen_b, puntos_a, puntos_b, 
                          duracion=2.0, fps=30, output_file="transicion.mp4"):
    # Cargar imágenes
    img1 = cv2.imread(imagen_a, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(imagen_b, cv2.IMREAD_UNCHANGED)
    
    # Verificar dimensiones
    if img1.shape != img2.shape:
        raise ValueError("Las imágenes deben tener las mismas dimensiones")
    
    # Cargar puntos de control
    pts1 = np.loadtxt(puntos_a)
    pts2 = np.loadtxt(puntos_b)
    
    # Generar frames de transición
    num_frames = int(duracion * fps)
    frames = morph_avanzado(img1, img2, pts1, pts2, num_frames)
    
    # Crear clips de video
    clips = []
    for frame in frames:
        # Convertir BGR a RGB para MoviePy
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        clips.append(ImageClip(frame_rgb, duration=1/fps))
    
    # Crear video con ciclo continuo
    video = concatenate_videoclips(clips + clips[::-1], method="compose")
    video.write_videofile(
        output_file,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        ffmpeg_params=['-crf', '18']
    )
    return output_file

if __name__ == "__main__":
    # Configuración
    DIR_BASE = "ruta/a/tus/archivos"
    
    # Archivos de entrada
    img1_path = "./hola/normal_ojos_abiertos.png"
    img2_path = "./hola/normal_ojos_cerrados.png"
    PUNTOS_ABIERTOS = os.path.join("puntos_abiertos.txt")
    PUNTOS_CERRADOS = os.path.join("puntos_cerrados.txt")
    
    # Generar puntos de control (ejecutar solo una vez)
    if not os.path.exists(PUNTOS_ABIERTOS):
        generar_puntos_control(img1_path, PUNTOS_ABIERTOS)
    
    if not os.path.exists(PUNTOS_CERRADOS):
        generar_puntos_control(img2_path, PUNTOS_CERRADOS)
    
    # Crear transición
    crear_transicion_suave(
        imagen_a=img1_path,
        imagen_b=img2_path,
        puntos_a=PUNTOS_ABIERTOS,
        puntos_b=PUNTOS_CERRADOS,
        duracion=1.5,
        fps=60,
        output_file="parpadeo_suave.mp4"
    )