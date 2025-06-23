import cv2
import numpy as np
from united_morph_transition import MorphTransition
from moviepy import ImageClip, concatenate_videoclips
import os
from ament_index_python import get_package_share_directory

def save_frames_to_folder(frames, output_frames_dir):
    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir)

    for i in range(len(frames)):
        frame = frames[i]
        frame_output_path = os.path.join(output_frames_dir, f"frame_{i:04d}.png")
        frame.save(frame_output_path, format="PNG")
        
    print(f"Frames saved in the folder: {output_frames_dir}")

def generate_checkpoints(img_path, output_points):
    img = cv2.imread(img_path)
    points = []
    
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            cv2.circle(img, (x, y), 3, (0,0,255), -1)
            cv2.imshow('Image', img)
    
    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    np.savetxt(output_points, np.array(points, dtype=int), fmt="%d", delimiter=", ")
    
    print(f"Saved points in {output_points}")
    return np.array(points)

def generate_eye_transition(dir_init_img, dir_final_img, points_init_img, points_final_img, 
                           num_frames=10, fps=24, exposure_time=0.5, 
                           folder_frames=None):
    init_img = cv2.imread(dir_init_img, cv2.IMREAD_UNCHANGED)
    init_img = cv2.cvtColor(init_img, cv2.COLOR_BGRA2RGBA)
    final_img = cv2.imread(dir_final_img, cv2.IMREAD_UNCHANGED)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGRA2RGBA)

    frames_transition = MorphTransition(
        dir_init_img,
        dir_final_img,
        points_init_img,
        points_final_img,
        num_images=num_frames
    ).morph_transition()

    if folder_frames is not None:
        save_frames_to_folder(frames_transition, folder_frames)

    frame_duration = 1/fps

    clips = [
        ImageClip(np.array(init_img), duration=exposure_time*3),  
        *[ImageClip(np.array(frame), duration=frame_duration) for frame in frames_transition],  
        ImageClip(np.array(final_img), duration=exposure_time*0.75),  
        *[ImageClip(np.array(frame), duration=frame_duration) for frame in reversed(frames_transition)],
        ImageClip(np.array(init_img), duration=exposure_time*3),  
    ]

    concatenate_videoclips(clips, method="compose")

path_pkg = get_package_share_directory('coco_face_screen')

init_img = os.path.join(path_pkg, "imgs", "open_mouth.png")
final_img = os.path.join(path_pkg, "imgs", "close_mouth.png")

points_init_img = os.path.join(path_pkg,"points","open_mouth.txt")
points_final_img = os.path.join(path_pkg,"points","close_mouth.txt")

folder_frames = "../imgs_transition/talking"

if not os.path.exists(points_init_img):
    generate_checkpoints(init_img, points_init_img)

if not os.path.exists(points_final_img):
    generate_checkpoints(final_img, points_final_img)

generate_eye_transition(
    final_img,
    init_img,
    points_final_img,
    points_init_img,
    num_frames=15,
    fps=60,
    folder_frames=folder_frames
)
