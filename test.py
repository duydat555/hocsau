from ultralytics import YOLO


def main():
    # Sử dụng model mới nhất với performance tốt nhất
    model = YOLO('runs/detect/safe_train/weights/best.pt')  
    
    # Training với cấu hình cân bằng classes - đã hoàn thành
    # model.train(
    #     data='dataset/football-player.v5i.yolo26/data.yaml', 
    #     epochs=100,  # Tăng epochs
    #     imgsz=640, 
    #     batch=16,  # Tăng batch size nếu GPU cho phép
    #     device=0,
    #     # Focal Loss parameters để xử lý class imbalance
    #     fl_gamma=1.5,  # Focal loss gamma
    #     # Data augmentation mạnh hơn
    #     hsv_h=0.015,  # Hue augmentation
    #     hsv_s=0.7,    # Saturation augmentation  
    #     hsv_v=0.4,    # Value augmentation
    #     degrees=10,   # Rotation
    #     translate=0.1, # Translation
    #     scale=0.9,    # Scaling
    #     shear=2.0,    # Shearing
    #     flipud=0.1,   # Vertical flip
    #     fliplr=0.5,   # Horizontal flip
    #     mosaic=1.0,   # Mosaic augmentation
    #     mixup=0.1,    # Mixup augmentation
    #     copy_paste=0.1, # Copy-paste augmentation
    #     # Training settings
    #     patience=25,   # Early stopping patience
    #     save_period=5, # Save every 5 epochs
    #     val=True,
    #     plots=True,
    #     verbose=True,
    #     # Optimizer settings
    #     optimizer='AdamW',
    #     lr0=0.001,    # Initial learning rate
    #     weight_decay=0.0005
    # )
    
    # Uncomment để resume training
    # model.train(resume=True)
    
    # Test prediction với settings tối ưu cho multi-class detection
    print("🎯 Testing with optimized settings for multi-class detection...")
    result = model.predict(
        'https://transform.roboflow.com/GGqN5h3Ag5Msh1nOv8tjZvWDZpv1/395edbf209e95c4cf2fa40d3098f196e/thumb.jpg', 
        conf=0.05,      # Lower confidence để detect minority classes
        iou=0.45,       # IoU threshold cho NMS
        max_det=300,    # Tăng max detections
        save=True,      # Save results
        save_txt=True,  # Save labels
        save_conf=True  # Save confidence scores
    )
    
    # In kết quả detection
    class_names = ['ball', 'goalkeeper', 'player', 'referee']
    print("\n🏆 DETECTION RESULTS:")
    print("="*50)
    
    total_detections = 0
    class_counts = {name: 0 for name in class_names}
    
    for r in result:
        if r.boxes is not None:
            total_detections = len(r.boxes)
            print(f"📊 Total detections: {total_detections}")
            
            for box in r.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = class_names[class_id]
                class_counts[class_name] += 1
                print(f"  ✅ {class_name}: {confidence:.3f}")
    
    if total_detections == 0:
        print("❌ No detections found!")
        print("💡 Try even lower confidence: conf=0.01")
    else:
        print(f"\n📈 SUMMARY:")
        for class_name, count in class_counts.items():
            if count > 0:
                print(f"  {class_name}: {count} detections")
    
    print(f"\n💾 Results saved to: runs/detect/predict*")
    return result

if __name__ == '__main__':
    main()
