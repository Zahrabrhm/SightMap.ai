import os
import cv2
from PIL import Image
import re

def sort_images_by_number(images):
    return sorted(images, key=lambda x: int(re.search(r'\d+', x).group()))

os.chdir("Path to images")
path = "Path to images"

mean_height = 0
mean_width = 0

num_of_images = len(os.listdir('.'))


for file in os.listdir('.'):
    im = Image.open(os.path.join(path, file))
    width, height = im.size
    mean_width += width
    mean_height += height
mean_width = int(mean_width / num_of_images)
mean_height = int(mean_height / num_of_images)
for file in os.listdir('.'):
    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith("png"):
        im = Image.open(os.path.join(path, file))
        width, height = im.size
        imResize = im.resize((mean_width, mean_height), Image.ANTIALIAS)
        imResize.save(file, 'JPEG', quality=95)
        print(f"{file} is resized")
def generate_video():
    image_folder = '.'
    video_name = 'output.avi'

    images = [img for img in os.listdir(image_folder)
              if img.endswith(".jpg") or
                 img.endswith(".jpeg") or
                 img.endswith("png")]

    images = sort_images_by_number(images)
    print(images)

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    frame_rate = 33.2  # 166 frames / 5 seconds

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
generate_video()
