import cv2
import os

image_folder = os.environ['HOME'] + '/imgs/'
video_name = 'eval.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

images = sorted(images, key=lambda x:int(x.split('.')[0]))
images = images[:500]

frame_width = 640
frame_height = 640
fps = 4

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_width, frame_height))

for image in images:
    image_path = os.path.join(image_folder, image)
    frame = cv2.imread(image_path)
    video.write(frame)

cv2.destroyAllWindows()
video.release()
