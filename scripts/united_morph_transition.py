import cv2
import numpy as np
from scipy.spatial import Delaunay
from PIL import Image

class MorphTransition:
    def __init__(self, src_image_path, dst_image_path, src_filename, dst_filename, num_images):
        self.src_image_path = src_image_path
        self.dst_image_path = dst_image_path
        self.src_filename = src_filename
        self.dst_filename = dst_filename
        self.num_images = num_images
    
    def read_points_from_file(self):
        """
        Load control point coordinates from text files.
        """
        points_src = []
        points_dst = []
        try:
            with open(self.src_filename, 'r') as file:
                for line in file:
                    x, y = map(int, line.strip().split(','))
                    points_src.append((x, y))
            with open(self.dst_filename, 'r') as file:
                for line in file:
                    x, y = map(int, line.strip().split(','))
                    points_dst.append((x, y))
        except Exception as e:
            print(f"Error reading points from file: {e}")
        return np.array(points_src), np.array(points_dst)

    def linear_interpolation(self, src_points, dst_points, t):
        """
        Calculate intermediate points between origin and destination points using linear interpolation.
        """
        return (1 - t) * src_points + t * dst_points
    
    def generate_fine_grid(self, points, interpolate_points, density=3):
        """
        Generate a denser mesh between control points.
        """
        points = np.array(points)
        interpolate_points = np.array(interpolate_points)

        t = np.linspace(0, 1, density, endpoint=False)[1:] 
        new_points = []
        new_interpolate_points = []

        for i in range(len(points) - 1):
            for t_val in t:
                new_points.append(self.linear_interpolation(points[i], points[i + 1], t_val))
                new_interpolate_points.append(self.linear_interpolation(interpolate_points[i], interpolate_points[i + 1], t_val))

        new_points = np.vstack([points, new_points])
        new_interpolate_points = np.vstack([interpolate_points, new_interpolate_points])
        
        return new_points, new_interpolate_points

    def apply_transformation(self, image, points, interpolate_points, density=3):
        """
        Optimize the image transformation to be faster and maintain quality.
        """
        points, interpolate_points = self.generate_fine_grid(points, interpolate_points, density)

        tri = Delaunay(interpolate_points) 
        rows, cols = image.shape[:2]
        new_image = np.zeros((rows, cols, image.shape[2]), dtype=image.dtype)

        affine_matrices = [
            cv2.getAffineTransform(np.float32(interpolate_points[simplex]), np.float32(points[simplex]))
            for simplex in tri.simplices
        ]

        for i, simplex in enumerate(tri.simplices):
            mask = np.zeros((rows, cols), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(interpolate_points[simplex]), 255)
            mask_indices = np.argwhere(mask > 0)

            if mask_indices.size == 0:
                continue 

            affine_matrix = affine_matrices[i]
            interpolate_pixels = np.hstack((mask_indices[:, ::-1], np.ones((mask_indices.shape[0], 1))))
            natural_pixels = (interpolate_pixels @ affine_matrix.T).astype(int)

            valid_mask = (0 <= natural_pixels[:, 0]) & (natural_pixels[:, 0] < cols) & \
                        (0 <= natural_pixels[:, 1]) & (natural_pixels[:, 1] < rows)
            natural_pixels = natural_pixels[valid_mask]
            mask_indices = mask_indices[valid_mask]

            new_image[mask_indices[:, 0], mask_indices[:, 1]] = image[natural_pixels[:, 1], natural_pixels[:, 0]]

        return new_image

    def smooth_image(self, image, kernel_size=5):
        """
        Smooth only the edges of non-transparent areas of the image.
        """
        alpha_channel = image[:, :, 3]
        rgb_image = image[:, :, :3]

        edges = cv2.Canny(alpha_channel, 50, 150)
        dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        smoothed_rgb = cv2.GaussianBlur(rgb_image, (kernel_size, kernel_size), 0)

        smoothed_image = rgb_image.copy()
        smoothed_image[dilated_edges > 0] = smoothed_rgb[dilated_edges > 0]

        result = np.dstack((smoothed_image, alpha_channel))
        return result

    def fill_holes(self, image):
        """
        Optimize the filling of transparent holes in the image.
        """
        mask = (image[:, :, 3] == 0).astype(np.uint8) 
        if not np.any(mask):
            return image 

        rgb_filled = cv2.inpaint(image[:, :, :3], mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)

        filled_image = np.dstack((rgb_filled, image[:, :, 3]))
        return filled_image
 
    def create_fade_transition_frames(self, deformed_images, deformed_images_2):
        """
        Create transition frames between two sequences of images.
        Receives the already loaded images as NumPy arrays.
        """
        num_frames = min(len(deformed_images), len(deformed_images_2))
        deformed_images = deformed_images[:num_frames]
        deformed_images_2 = deformed_images_2[:num_frames]

        width, height = deformed_images[0].shape[1], deformed_images[0].shape[0]

        frames=[]

        for i, (img_array_1, img_array_2) in enumerate(zip(deformed_images, deformed_images_2)):
            frame = Image.new("RGBA", (width, height), (0, 0, 0, 0))

            img1 = Image.fromarray(img_array_1)
            img2 = Image.fromarray(img_array_2)

            img1 = img1.convert("RGBA")
            img2 = img2.convert("RGBA")

            alpha_1 = 1  

            alpha_2 = min(1, i / (num_frames - 1))

            img1 = Image.blend(Image.new("RGBA", (width, height), (0, 0, 0, 0)), img1, alpha_1)
            img2 = Image.blend(Image.new("RGBA", (width, height), (0, 0, 0, 0)), img2, alpha_2)

            frame = Image.alpha_composite(frame, img1)
            frame = Image.alpha_composite(frame, img2)

            frames.append(frame)

        return frames

    def morph_transition(self):
        """
        Optimize the morphing transition between two images.
        """
        src_image = cv2.imread(self.src_image_path, cv2.IMREAD_UNCHANGED)

        if src_image is None:
            print(f"Error: Don't load the image {self.src_image_path}. Check the path.")
        else:
            src_image = cv2.cvtColor(src_image, cv2.COLOR_BGRA2RGBA)
            
            dst_image = cv2.imread(self.dst_image_path, cv2.IMREAD_UNCHANGED)

            if dst_image is None:
                print(f"Error: Don't load the image {self.dst_image_path}. Check the path.")
            else:
                dst_image = cv2.cvtColor(dst_image, cv2.COLOR_BGRA2RGBA)

                src_points, dst_points = self.read_points_from_file()

                deformed_images_1 = []
                deformed_images_2 = []
                
                for i in range(1, self.num_images + 1 ):
                    t = i / (self.num_images + 1)  
                    print(f"Generating frame {i}/{self.num_images} with t={t:.2f}")
                    
                    interpolated_points_1 = self.linear_interpolation(src_points, dst_points, t)
                    interpolated_points_2 = self.linear_interpolation(dst_points, src_points, t)

                    deformed_image_1 = self.apply_transformation(src_image, src_points, interpolated_points_1)
                    deformed_image_2 = self.apply_transformation(dst_image, dst_points, interpolated_points_2)
                    
                    holes_filled_1 = self.fill_holes(deformed_image_1)
                    holes_filled_2 = self.fill_holes(deformed_image_2)

                    smoothed_image_1 = self.smooth_image(holes_filled_1)
                    smoothed_image_2 = self.smooth_image(holes_filled_2)

                    deformed_images_1.append(smoothed_image_1)
                    deformed_images_2.append(smoothed_image_2)
                
                deformed_images_2 = deformed_images_2[::-1]
                frames = self.create_fade_transition_frames(deformed_images_1, deformed_images_2)

                return frames
