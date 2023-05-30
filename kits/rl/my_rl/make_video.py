import cv2
import os
os.environ['HOME'] = 'D:\\PycharmProjects\\Lux-Design-S2\\kits\\rl\\my_rl'
image_folder = os.environ['HOME'] + '/imgs/'
video_name = image_folder+'video3.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

images = sorted(images, key=lambda x:int(x.split('.')[0]))
images = images[:40]

frame_width = 640
frame_height = 640
fps = 4

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

for image in images:
    image_path = os.path.join(image_folder, image)
    frame = cv2.imread(image_path)
    video.write(frame)

cv2.destroyAllWindows()
video.release()
