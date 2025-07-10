import cv2
from ultralytics import YOLO

# 1. YOLOv11 브랜드 모델 로드
model = YOLO("YOLO/kickboardBrand_yolov11m.pt")

# 2. 이미지 불러오기
image_path = "output/kickboard_id2_frame114_conf0.71.jpg"
image = cv2.imread(image_path)

# 3. 객체 탐지 실행
results = model(image)

# 4. 탐지된 브랜드 정보 모두 print
if results[0].boxes is not None and len(results[0].boxes) > 0:
    brand_classes = results[0].boxes.cls.cpu().numpy()
    brand_confs = results[0].boxes.conf.cpu().numpy()
    for idx, (cls, conf) in enumerate(zip(brand_classes, brand_confs)):
        brand_name = model.names[int(cls)] if hasattr(model, "names") else str(cls)
        print(f"{idx+1}. 브랜드: {brand_name} (conf: {conf:.2f})")
else:
    print("탐지된 브랜드가 없습니다.")
