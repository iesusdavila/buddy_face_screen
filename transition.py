import os
from PIL import Image
import numpy as np

def open_folder(folder_1, folder_2):
    """
    Carga las imágenes de dos carpetas y las convierte en arreglos NumPy.
    """
    def load_images(folder):
        images = []
        for file_name in sorted(os.listdir(folder)):
            if file_name.endswith((".png", ".jpg")):
                image_path = os.path.join(folder, file_name)
                img = Image.open(image_path).convert("RGBA")  # Asegúrate de que sea RGBA
                images.append(np.array(img))
        return images

    images_1 = load_images(folder_1)
    images_2 = load_images(folder_2)
    return images_1, images_2

def save_frames_to_folder(frames, output_frames_dir):
    # Crear la carpeta para guardar los frames si no existe
    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir)

    for i in range(len(frames)):
        frame = frames[i]
        # Guardar el frame como imagen en la carpeta
        frame_output_path = os.path.join(output_frames_dir, f"frame_{i:04d}.png")
        frame.save(frame_output_path, format="PNG")
        
    print(f"Frames guardados en la carpeta: {output_frames_dir}")

def create_fade_transition_frames(deformed_images, deformed_images_2):
    """
    Crea los frames de transición entre dos secuencias de imágenes.
    Recibe las imágenes ya cargadas como arrays NumPy.
    """
    # Asegurar que ambas secuencias tengan el mismo número de imágenes
    num_frames = min(len(deformed_images), len(deformed_images_2))
    deformed_images = deformed_images[:num_frames]
    deformed_images_2 = deformed_images_2[:num_frames]

    # Obtener las dimensiones de las imágenes
    width, height = deformed_images[0].shape[1], deformed_images[0].shape[0]

    frames=[]

    for i, (img_array_1, img_array_2) in enumerate(zip(deformed_images, deformed_images_2)):
        # Crear una imagen base (fondo transparente)
        frame = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        # Convertir las imágenes de arrays NumPy a imágenes de PIL
        img1 = Image.fromarray(img_array_1)
        img2 = Image.fromarray(img_array_2)

        # Asegurar que ambas imágenes sean RGBA
        img1 = img1.convert("RGBA")
        img2 = img2.convert("RGBA")

        # Calcular las opacidades
        alpha_1 = 1  # Disminuye de 1 a 0
        #alpha_2 = 1  # Aumenta de 0 a 1


        # Calcular las opacidades
        #alpha_1 = max(0, 1 - (i / (num_frames - 1)))  # Disminuye de 1 a 0
        alpha_2 = min(1, i / (num_frames - 1))        # Aumenta de 0 a 1


        # Ajustar opacidad de ambas imágenes
        img1 = Image.blend(Image.new("RGBA", (width, height), (0, 0, 0, 0)), img1, alpha_1)
        img2 = Image.blend(Image.new("RGBA", (width, height), (0, 0, 0, 0)), img2, alpha_2)

        # Superponer ambas imágenes
        frame = Image.alpha_composite(frame, img1)
        frame = Image.alpha_composite(frame, img2)

        frames.append(frame)

    return frames
