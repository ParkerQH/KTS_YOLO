import cv2
import numpy as np
from ultralytics import YOLO

# YOLO 모델 로드
model_kickboard = YOLO("YOLO/kickboard_yolov11s.pt")
model_person = YOLO("YOLO/person_yolov11m.pt")
model_helmet = YOLO("YOLO/helmet_yolov11m.pt")
model_brand = YOLO("YOLO/kickboardBrand_yolov11m.pt")


# 킥보드 분석 모듈
def kickboard_conclusion(image):
    kickboard_results = model_kickboard(image)
    kickboard_detected = (
        kickboard_results[0].boxes is not None and len(kickboard_results[0].boxes) > 0
    )
    return kickboard_detected


# 사람 분석 모듈
def person_conclusion(image):
    person_results = model_person(image)
    person_detected = (
        person_results[0].boxes is not None and len(person_results[0].boxes) > 0
    )
    return person_detected


# 킥보드 브랜드 분석 모듈
def brand_conclusion(image):
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
            return best_brand_name
        else:
            print("🚫 conf 0.7 이상 브랜드 감지 없음")
            return None
    else:
        return None


# 헬멧 분석 모듈
def helmet_conclusion(image):
    helmet_results = model_helmet(image)
    helmet_detected = (
        helmet_results[0].boxes is not None and len(helmet_results[0].boxes) > 0
    )
    confs = helmet_results[0].boxes.conf.cpu().numpy().tolist()
    # 제일 큰 conf 값 저장
    max_conf = float(max(confs)) if confs else 0.0
    return helmet_detected, helmet_results, max_conf


# 감지 결과 시각화
def draw_boxes(results, image, color=(0, 255, 0), label=""):
    if results[0].boxes is not None:
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            if label:
                cv2.putText(
                    image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )
