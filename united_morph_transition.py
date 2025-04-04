import cv2
import numpy as np
import os
import sys
# Añade el directorio src a sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from transformation import read_points_from_file
from transformation import read_points_from_file, apply_transformation, smooth_image, interpolate_points, fill_holes
from transition import create_fade_transition_frames,save_frames_to_folder

def morph_transition(src_image_path, dst_image_path ,src_filename ,dst_filename, num_images):
    src_image = cv2.imread(src_image_path, cv2.IMREAD_UNCHANGED)

    # Verificar si la imagen se cargó correctamente
    if src_image is None:
        print(f"Error: No se pudo cargar la imagen {src_image_path}. Verifica el path.")
    # elif src_image.shape[2] != 4:
    #     print("Error: La imagen no tiene un canal alfa (RGBA). Usa una imagen con fondo transparente.")
    else:
        print(f"Imagen {src_image_path} cargada exitosamente.")

        # Convertir la imagen de OpenCV (BGRA) a formato RGBA (RGBA)
        src_image = cv2.cvtColor(src_image, cv2.COLOR_BGRA2RGBA)

        # Cargar la imagen de destino con canal alfa

        dst_image = cv2.imread(dst_image_path, cv2.IMREAD_UNCHANGED)

        # Verificar si la imagen se cargó correctamente
        if dst_image is None:
            print(f"Error: No se pudo cargar la imagen {dst_image_path}. Verifica el path.")
        # elif dst_image.shape[2] != 4:
        #     print("Error: La imagen no tiene un canal alfa (RGBA). Usa una imagen con fondo transparente.")
        else:
            print(f"Imagen {dst_image_path} cargada exitosamente.")

            # Convertir la imagen de OpenCV (BGRA) a formato RGBA (RGBA)
            dst_image = cv2.cvtColor(dst_image, cv2.COLOR_BGRA2RGBA)

            # Leer los puntos desde los archivos .txt
            src_points = read_points_from_file(src_filename)
            dst_points = read_points_from_file(dst_filename)

            # Verificar si los puntos fueron leídos correctamente
            if len(src_points) == 0 or len(dst_points) == 0:
                print("Error: No se pudieron leer los puntos desde los archivos.")
            else:

                deformed_images_1 = []
                deformed_images_2 = []
                
                # Generar imágenes deformadas para diferentes valores de interpolación (t = 0, 0.2, 0.4, ..., 1)
                for i in range(1, num_images + 1 ):
                    t = i / (num_images + 1)  # Factor de interpolación entre 0 y 1, menos 0 y 1
                    print(f"Generando imagen para t = {t}")
                    
                    # Interpolamos los puntos
                    interpolated_points_1 = interpolate_points(src_points, dst_points, t)
                    interpolated_points_2 = interpolate_points(dst_points, src_points, t)

                    # Deformamos la imagen usando los puntos interpolados
                    deformed_image_1 = apply_transformation(src_image, src_points, interpolated_points_1)
                    deformed_image_2 = apply_transformation(dst_image, dst_points, interpolated_points_2)
                    
                    #Rellenar huecos
                    holes_filled_1 = fill_holes(deformed_image_1)
                    holes_filled_2 = fill_holes(deformed_image_2)

                    # Suavizar la imagen resultante
                    smoothed_image_1 = smooth_image(holes_filled_1)
                    smoothed_image_2 = smooth_image(holes_filled_2)

                    deformed_images_1.append(smoothed_image_1)
                    deformed_images_2.append(smoothed_image_2)
                
                print("Generación de imágenes completada.")

                deformed_images_2 = deformed_images_2[::-1]

                frames= create_fade_transition_frames(deformed_images_1, deformed_images_2)

                return frames

"""

# Cargar la imagen con canal alfa
src_image_path = 'caritas/partes separadas sin fondo/bocas/6.png'  # Asegúrate de que la imagen tenga un canal alfa

dst_image_path = 'caritas/partes separadas sin fondo/bocas/1.png'  # Asegúrate de que la imagen tenga un canal alfa

src_filename = 'caritas/partes separadas sin fondo/bocas_points/6_points.txt'  # Puntos de referencia
dst_filename = 'caritas/partes separadas sin fondo/bocas_points/1_points.txt'  # Puntos destino

output_frames_dir = "transition_frames_main_6 a 1"

num_images = 6  # Número de imágenes a generar

frames= morph_transition(src_image_path, dst_image_path ,src_filename ,dst_filename, num_images)

save_frames_to_folder(frames, output_frames_dir)

"""
