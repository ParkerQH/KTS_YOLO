import os
import cv2
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv
# .env 로드
load_dotenv()

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "YOLO")

model_kickboard = YOLO(os.path.join(MODEL_PATH, "kickboard_yolov11l.pt"))
model_person = YOLO(os.path.join(MODEL_PATH, "person_yolov11l.pt"))
model_helmet = YOLO(os.path.join(MODEL_PATH, "helmet_yolov11l.pt"))
model_brand = YOLO(os.path.join(MODEL_PATH, "kickboardBrand_yolov11l.pt"))


# 킥보드 분석 모듈
def kickboard_analysis(image):
    kickboard_results = model_kickboard(image)
    kickboard_detected = (
        kickboard_results[0].boxes is not None and len(kickboard_results[0].boxes) > 0
    )
    return kickboard_detected


# 사람 분석 모듈
def person_analysis(image):
    person_results = model_person(image)
    person_detected = (
        person_results[0].boxes is not None and len(person_results[0].boxes) > 0
    )
    return person_detected


# 킥보드 브랜드 분석 모듈
def brand_analysis(image):
    brand_results = model_brand(image)

    # conf 0.7 이상 중 최고 신뢰도 브랜드 한 개만 추출
    if brand_results[0].boxes is not None and len(brand_results[0].boxes) > 0:
        brand_classes = brand_results[0].boxes.cls.cpu().numpy()
        brand_confs = brand_results[0].boxes.conf.cpu().numpy()
        filtered = [
            (cls, conf) for cls, conf in zip(brand_classes, brand_confs) if conf >= 0.7
        ]
        if filtered:
            best_cls, best_conf = max(filtered, key=lambda x: x[1])
            best_brand_name = model_brand.names[int(best_cls)]
            print(f"✅ 브랜드 감지 : {best_brand_name}")
            return best_brand_name
        else:
            print("🚫 conf 0.7 이상 브랜드 감지 없음")
            return None
    else:
        return None


# 헬멧 분석 모듈
def helmet_analysis(image):
    helmet_results = model_helmet(image)
    helmet_detected = (
        helmet_results[0].boxes is not None and len(helmet_results[0].boxes) > 0
    )
    confs = helmet_results[0].boxes.conf.cpu().numpy().tolist()
    # 제일 큰 conf 값 저장
    max_conf = float(max(confs)) if confs else 0.0
    return helmet_detected, helmet_results, max_conf


# 감지 결과 시각화
def draw_boxes(results, image, color, label=""):
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        for i, box in enumerate(boxes):
            conf = confs[i]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            if label:
                text = f"{label}: {conf:.2f}"
                cv2.putText(
                    image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )
