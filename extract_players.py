import os

import cv2

from ultralytics import YOLO

# Load YOLOv8 pretrained
model = YOLO("football_detection/yolo_player_model/weights/best.pt")  # nano version cho nhẹ

video_path = "input_video.mp4"
output_dir = "cropped_players"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_id = 0
saved_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect
    results = model(frame)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])

            # class 2 = player in this dataset (0=ball, 1=goalkeeper, 3=referee)
            if cls == 2:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]

                if crop.size != 0:
                    cv2.imwrite(f"{output_dir}/player_{saved_id}.jpg", crop)
                    saved_id += 1

    frame_id += 1

cap.release()
print("Done extracting players.")
