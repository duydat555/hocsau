from ultralytics import YOLO
import torch
import os

# 1. ĐƯỜNG DẪN ĐẾN DATASET
# Thay 'path/to/your/dataset' bằng đường dẫn thực tế đến thư mục bạn đã tải về
# Ví dụ: 'C:/Users/Admin/Downloads/football-dataset'
dataset_path = 'dataset/football-player.v5i.yolo26/data.yaml'

# 2. KHỞI TẠO MÔ HÌNH (Dòng Nano cực nhẹ cho RTX 3050)
# Bạn có thể dùng 'yolov8n.pt' hoặc 'yolo11n.pt' (bản mới nhất)

if __name__ == '__main__':
    model = YOLO('yolo26n.pt') 

    # 3. CẤU HÌNH VÀ HUẤN LUYỆN
    # Tối ưu cho 4GB VRAM để tránh lỗi Out of Memory (OOM)
    results = model.train(
        data=dataset_path, 
        epochs=100,           # Số vòng lặp huấn luyện
        imgsz=640,            # Kích thước ảnh chuẩn (640x640)
        batch=8,              # RTX 3050 4GB nên chạy batch 8 hoặc 16
        workers=4,            # Tận dụng 32GB RAM để load ảnh nhanh hơn
        device=0,             # Chạy bằng GPU (0 là ID của card đồ họa)
        amp=True,             # Bật chế độ tự động tối ưu bộ nhớ (Automatic Mixed Precision)
        project='football_detection', # Tên dự án
        name='yolo_player_model',     # Tên phiên bản training
        exist_ok=True
    )

    print("--- HUẤN LUYỆN HOÀN TẤT ---")
    print("Model tốt nhất nằm tại: runs/detect/yolo_player_model/weights/best.pt")