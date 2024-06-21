import cv2
import os
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from depth_midas_gray import method
from yolo_video import detection
import torch
def display_results(detected_images, bounding_boxes):
    """
    Displays the detected images with bounding boxes.

    Args:
        detected_images (list): List of images with bounding boxes drawn.
        bounding_boxes (list): List of bounding boxes (label, score, and coordinates) for each frame.
    """
    for img, boxes in zip(detected_images, bounding_boxes):
        plt.figure(figsize=(12, 9))
        plt.imshow(img)
        for label, score, box in boxes:
            print(f"  Label: {label}, Score: {score}, Bounding Box Coordinates: {box}")
        plt.axis('off')
        plt.show()

model = YOLO("yolov8l-world.pt")  # Load model
video_path = "Path to video file"  
classes = ["garbage can", "pole", "curb", "car", "person", "stairs", "bench", "ledge", "door", "street light", "animal", "street sign", "table", "chair"]
interval =3

detected_images, bounding_boxes, frame_images,width,height = detection.process_video(model, video_path, classes, interval)

def save_image(foldername,idx,img):
    if not os.path.exists(foldername): 
        os.makedirs(foldername) 
    name = './'+ foldername +'/frame' + str(idx) + '.png'
    
     
    
    cv2.imwrite(name, img)
    
frame_images_depth = []
frame_images_depth_gray = []
for frame_image in frame_images:
    colored_output, output = method.depth_midas(frame_image)
    frame_images_depth.append(colored_output)
    frame_images_depth_gray.append(output)
    torch.cuda.empty_cache()


#display_results(frame_images_depth, bounding_boxes)

medians_midpoints=[]
depth_detected_images=[]
left_threshold = 0.33
right_threshold = 0.66

medians_midpoints = []
depth_detected_images = []

for i, img in enumerate(frame_images_depth):
    bbox_coords_all = bounding_boxes[i]
    height, width, _ = img.shape
    
    img_with_boxes = img.copy()
    
    image_medians = []
    midpoints = []
    for j, box_info in enumerate(bbox_coords_all):
        label = box_info[0]
        x_min, y_min, x_max, y_max = box_info[2]
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        
        midpoint = ((x_max - x_min) / 2 + x_min)
        location = None
        if midpoint < (width * left_threshold):
            location = "Left"
        elif midpoint > (width * right_threshold):
            location = "Right"
        else:
            location = "Middle"
            
        assert location is not None
        cropped_img = frame_images_depth_gray[i][int(y_min):int(y_max), int(x_min):int(x_max)]
        
        median = np.median(cropped_img)
        image_medians.append((label, median, location, midpoint))
        
        cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), (0, 0, 0), 2)
        cv2.putText(img_with_boxes, f"{label}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,1.5, (0, 0, 0), 3)
        cv2.putText(img_with_boxes, f"{location}", (x_min, y_min -60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        
    medians_midpoints.append(image_medians)
    
    depth_detected_images.append(img_with_boxes)
    save_image('DepthMapResult', i, img_with_boxes)
    
display_results(depth_detected_images, bounding_boxes)

for image_medians in medians_midpoints:
    image_medians.sort(key=lambda x: x[1], reverse=True) 


from copy import deepcopy
LEFT_START = (int(width / 2), int(height))
MIDDLE_START = (int(width / 2), int(height))
RIGHT_START = (int(width / 2), int(height))

LEFT_END = (int(width / 6), int(height / 2))
MIDDLE_END = (int(width / 2), int(height / 2))
RIGHT_END = (int(5*width / 6), int(height / 2))

def draw_img_with_arrow(idx, mode, colour, thickness=20):
    if mode == "Left":
        start = LEFT_START
        end = LEFT_END
    elif mode == "Right":
        start = RIGHT_START
        end = RIGHT_END
    else:
        start = MIDDLE_START
        end = MIDDLE_END

    test_frame = deepcopy(frame_images[idx])
    cv_im = cv2.arrowedLine(test_frame, start, end, colour, thickness)
    save_image('ArrowResult', idx, cv_im)


for i in range(len(frame_images)):
    modes = []
    for j in medians_midpoints[i]:
        modes.append((j[2], j[1]))

    modes.sort(key=lambda x: x[1])
    
    if len(set(mode[0] for mode in modes)) == 3:
        mode = modes[0][0]
    elif "Middle" not in set(mode[0] for mode in modes):
        mode = "Middle"
        
    elif "Left" not in set(mode[0] for mode in modes):
        mode = "Left"
    elif "Right" not in set(mode[0] for mode in modes):
        mode = "Right"
    else:  # Default case
        mode = "Unknown"

    draw_img_with_arrow(i, mode, (0, 255, 0))
            



  

    
