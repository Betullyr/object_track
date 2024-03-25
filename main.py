from ultralytics import YOLO
import cv2
import os

# load yolov8 model
model = YOLO('yolov8n.pt')

# load video
VIDEOS_DIR = 'C:/Users/dell/Desktop/object_track/'

video_path = os.path.join(VIDEOS_DIR, 'video.mp4')
cap = cv2.VideoCapture(video_path)

video_out_path = 'out.mp4'.format(video_path)

# 384,640
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (600, 720))

ret = True
# read frames
while ret:
    ret, frame = cap.read()

    if ret:
        # detect objects
        # track objects
        results = model.track(frame, persist=True)

        obinfor = results[0].verbose()

        # plot results
        # cv2.rectangle
        # cv2.putText

        frame_ = results[0].plot()
        cv2.putText(frame_, obinfor, (400, 50), cv2.FONT_ITALIC, 2, (0, 255, 0), 2)

        # visualize
        cap_out.write(frame_)

cap.release()
cap_out.release()
cv2.destroyAllWindows()