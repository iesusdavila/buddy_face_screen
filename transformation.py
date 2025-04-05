import cv2
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import numpy as np

# Función para leer los puntos desde un archivo .txt
def read_points_from_file(filename):
    points = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                x, y = map(int, line.strip().split(','))
                points.append((x, y))
    except Exception as e:
        print(f"Error al leer el archivo {filename}: {e}")
    return np.array(points)

def generate_fine_grid(src_points, dst_points, density=3):
    """
    Genera una malla más densa entre los puntos de control.
    """
    src_points = np.array(src_points)
    dst_points = np.array(dst_points)

    t = np.linspace(0, 1, density, endpoint=False)[1:]  # Evitar duplicar puntos originales
    new_src_points = []
    new_dst_points = []

    for i in range(len(src_points) - 1):
        for t_val in t:
            new_src_points.append((1 - t_val) * src_points[i] + t_val * src_points[i + 1])
            new_dst_points.append((1 - t_val) * dst_points[i] + t_val * dst_points[i + 1])

    new_src_points = np.vstack([src_points, new_src_points])
    new_dst_points = np.vstack([dst_points, new_dst_points])
    
    return new_src_points, new_dst_points

def apply_transformation(image, src_points, dst_points, density=3):
    """
    Optimiza la transformación de la imagen para ser más rápida y mantener la calidad.
    """
    src_points, dst_points = generate_fine_grid(src_points, dst_points, density)
    tri = Delaunay(dst_points)  # Triangulación rápida usando los puntos destino
    rows, cols = image.shape[:2]
    new_image = np.zeros((rows, cols, image.shape[2]), dtype=image.dtype)

    # Matrices inversas de transformación para cada triángulo
    affine_matrices = [
        cv2.getAffineTransform(np.float32(dst_points[simplex]), np.float32(src_points[simplex]))
        for simplex in tri.simplices
    ]

    # Generar una máscara por triángulo directamente en paralelo
    for i, simplex in enumerate(tri.simplices):
        mask = np.zeros((rows, cols), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dst_points[simplex]), 255)
        mask_indices = np.argwhere(mask > 0)

        if mask_indices.size == 0:
            continue  # Si no hay píxeles en este triángulo, omitir

        # Transformar los píxeles destino a coordenadas fuente
        affine_matrix = affine_matrices[i]
        dst_pixels = np.hstack((mask_indices[:, ::-1], np.ones((mask_indices.shape[0], 1))))
        src_pixels = (dst_pixels @ affine_matrix.T).astype(int)

        # Filtrar píxeles fuera de los límites
        valid_mask = (0 <= src_pixels[:, 0]) & (src_pixels[:, 0] < cols) & \
                     (0 <= src_pixels[:, 1]) & (src_pixels[:, 1] < rows)
        src_pixels = src_pixels[valid_mask]
        mask_indices = mask_indices[valid_mask]

        # Copiar píxeles de la imagen fuente
        new_image[mask_indices[:, 0], mask_indices[:, 1]] = image[src_pixels[:, 1], src_pixels[:, 0]]

    return new_image

def fill_holes(image):
    """
    Optimiza el relleno de huecos transparentes en la imagen.
    """
    mask = (image[:, :, 3] == 0).astype(np.uint8)  # Detectar píxeles transparentes
    if not np.any(mask):
        return image  # No hay huecos

    # Aplicar relleno únicamente donde hay transparencia
    rgb_filled = cv2.inpaint(image[:, :, :3], mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)

    # Combinar el resultado con el canal alfa original
    filled_image = np.dstack((rgb_filled, image[:, :, 3]))
    return filled_image

# Función para suavizar la imagen usando un filtro gaussiano
def smooth_image(image, kernel_size=5):
    """
    Suaviza únicamente los bordes de las áreas no transparentes de la imagen.
    """
    # Separar el canal alfa y la imagen RGB
    alpha_channel = image[:, :, 3]
    rgb_image = image[:, :, :3]

    # Crear una máscara de bordes donde el alfa cambia (es decir, la transición entre transparente y no transparente)
    edges = cv2.Canny(alpha_channel, 50, 150)
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # Suavizar la región de bordes en la imagen RGB
    smoothed_rgb = cv2.GaussianBlur(rgb_image, (kernel_size, kernel_size), 0)

    # Combinar la imagen suavizada en los bordes con la original
    smoothed_image = rgb_image.copy()
    smoothed_image[dilated_edges > 0] = smoothed_rgb[dilated_edges > 0]

    # Reconstruir la imagen con el canal alfa original
    result = np.dstack((smoothed_image, alpha_channel))
    return result

# Función para interpolar entre dos conjuntos de puntos
def interpolate_points(src_points, dst_points, t):
    return (1 - t) * src_points + t * dst_points