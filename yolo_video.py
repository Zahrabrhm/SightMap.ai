import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
class detection(object):
    def __init__(self, name):
        self.name=name 
    def process_video(model, video_path, classes, interval):
        """
        Processes a video, detects objects at specified frame intervals, and returns processed results.

        Args:
            model (YOLO): The YOLO model for object detection.
            video_path (str): Path to the video file.
            classes (list): List of classes to detect.
            interval (int): Number of frames to skip between detections.

        Returns:
            detected_images (list): List of images with bounding boxes drawn.
            bounding_boxes (list): List of bounding boxes for each frame.
            frame_images (list): List of original frame images.
        """
        # Set the classes for detection
        model.set_classes(classes)

        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        detected_images = []
        bounding_boxes = []
        frame_images = []
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) if cap.isOpened() else None
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) if cap.isOpened() else None

        current_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process every 'interval' frame
            if current_frame % interval == 0:
                # Convert frame to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Perform detection
                results = model.predict(rgb_frame, conf=0.5, iou=0.45)  # Adjust thresholds

                # Convert results to the required format
                boxes = []
                for det in results:
                    if det.boxes is not None:
                        for i in range(len(det.boxes)):
                            box = det.boxes[i].xyxy[0].tolist()
                            label = det.names[det.boxes[i].cls[0].item()]
                            score = det.boxes[i].conf[0].item()
                            boxes.append((label, score, box))

                # Draw bounding boxes on the frame
                fig, ax = plt.subplots(1, figsize=(12, 9))
                ax.imshow(rgb_frame)
                for label, score, box in boxes:
                    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    plt.text(box[0], box[1] - 10, f"{label} ({score:.2f})", color='red', fontsize=12, weight='bold')
                plt.axis('off')

                # Save the figure as image
                fig.canvas.draw()
                img_with_boxes = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img_with_boxes = img_with_boxes.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)

                detected_images.append(img_with_boxes)
                bounding_boxes.append(boxes)
                frame_images.append(rgb_frame)

            current_frame += 1

        cap.release()

        return detected_images, bounding_boxes, frame_images,width,height