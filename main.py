import cv2
from ultralytics import YOLO
import time

WEIGHTS_PATH = r"./weights/best.pt"
VIDEO_PATH = r"./data/inp.mp4"

model = YOLO(WEIGHTS_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

start_time = time.time()
frame_count = 0

while cap.isOpened():
    
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        results = model(frame)
        
        annotated_frame = results[0].plot()
            
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
