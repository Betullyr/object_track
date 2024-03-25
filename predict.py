import cv2
from ultralytics import YOLO
import os

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')
model.predict(source='your video path/video.mp4', save=True)

"""

# Open the video file
VIDEOS_DIR = 'your video path'

video_path = os.path.join(VIDEOS_DIR, 'video.mp4')

video_out_path = 'out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)

H,W = 640,480
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))



# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        cap_out.write(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(200) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap_out.release()
cap.release()
cv2.destroyAllWindows()
"""