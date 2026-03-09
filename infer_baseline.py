import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLO
detector = YOLO("yolo26n.pt")

# Load classifier
if not os.path.exists("role_classifier.pth"):
    raise FileNotFoundError(
        "role_classifier.pth not found. "
        "Run train_classifier.py first to generate it."
    )
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, 2)  # goalkeeper, player
model.load_state_dict(torch.load("role_classifier.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class_names = ["goalkeeper", "player"]

cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector(frame)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            # yolo26n.pt is a COCO model: class 0 = person (covers all players)
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                input_tensor = transform(crop).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    pred = torch.argmax(output, dim=1).item()

                label = class_names[pred]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (0,255,0), 2)

    cv2.imshow("Baseline Demo", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()