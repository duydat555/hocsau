import cv2
from ultralytics import YOLO

# ================= LOAD MODEL =================
model = YOLO("football_detection/yolo_player_model/weights/best.pt")

print("Class names:", model.names)  # kiểm tra class

# ================= VIDEO =================
cap = cv2.VideoCapture("input_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]

            # ===== FILTER =====
            if conf < 0.5:
                continue

            # ===== BOX =====
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ===== COLOR THEO CLASS =====
            if class_name == "player":
                color = (0, 255, 0)
            elif class_name == "goalkeeper":
                color = (0, 0, 255)
            elif class_name == "ball":
                color = (255, 0, 0)
            else:
                color = (0, 255, 255)

            # ===== DRAW =====
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name} ({conf:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

            # DEBUG
            print(f"{class_name} | conf={conf:.2f}")

    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
        break

cap.release()
cv2.destroyAllWindows()