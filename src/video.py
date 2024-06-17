import cv2
import os
from natsort import natsorted

folder_name = '2024_06_17-12_07_28'
# Path to the directory containing .jpg images
image_folder = './data/images/' + folder_name
# Path to save the output video
output_video_path = './data/video/' + folder_name + '.avi'
# Video settings
frame_rate = 2
video_codec = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs like 'MJPG', 'MP4V', etc.

# Get list of images
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images = natsorted(images)  # Ensure the images are sorted in the correct order

# Read the first image to get the size
first_image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_image.shape

# Initialize the video writer
video = cv2.VideoWriter(output_video_path, video_codec, frame_rate, (width, height))

# Iterate through images and write them to the video
for image in images:
    print(image)
    img_path = os.path.join(image_folder, image)
    img = cv2.imread(img_path)
    video.write(img)

# Release the video writer
video.release()
cv2.destroyAllWindows()

print(f'Video saved at {output_video_path}')
