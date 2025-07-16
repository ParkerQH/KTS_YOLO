import cv2
from ultralytics import YOLO

# 각각의 모델 로드
model_kickboard = YOLO("YOLO/kickboard_yolov11l.pt")
model_person = YOLO("YOLO/person_yolov11l(2).pt")
model_helmet = YOLO("YOLO/helmet_yolov11l.pt")
model_brand = YOLO("YOLO/kickboardBrand_yolov11l.pt")

# 이미지 불러오기
image_path = "output/image.png"
image = cv2.imread(image_path)

# 1. 킥보드 감지
kickboard_results = model_kickboard(image)
kickboard_detected = (
    kickboard_results[0].boxes is not None and len(kickboard_results[0].boxes) > 0
)
kickboard_conf = (
    float(kickboard_results[0].boxes.conf.max()) if kickboard_detected else 0
)

# 2. 사람 감지
person_results = model_person(image)
person_detected = (
    person_results[0].boxes is not None and len(person_results[0].boxes) > 0
)
person_conf = (
    float(person_results[0].boxes.conf.max()) if person_detected else 0
)

print(f"킥보드 감지: {kickboard_detected} (conf: {kickboard_conf:.2f}), "
      f"사람 감지: {person_detected} (conf: {person_conf:.2f})")

# 3. 킥보드와 사람이 모두 있을 때만 헬멧/브랜드 감지
if kickboard_detected and person_detected:
    # 3-1. 헬멧 감지
    helmet_results = model_helmet(image)
    helmet_detected = (
        helmet_results[0].boxes is not None and len(helmet_results[0].boxes) > 0
    )
    helmet_conf = (
        float(helmet_results[0].boxes.conf.max()) if helmet_detected else 0
        
    )

    # 3-2. 브랜드 감지
    brand_results = model_brand(image)
    brand_detected = (
        brand_results[0].boxes is not None and len(brand_results[0].boxes) > 0
    )
    # 브랜드 감지 결과에서 conf 0.7 이상 중 최고 신뢰도 브랜드 한 개만 추출
    best_brand_name, best_brand_conf = None, 0
    if brand_detected:
        brand_classes = brand_results[0].boxes.cls.cpu().numpy()
        brand_confs = brand_results[0].boxes.conf.cpu().numpy()
        filtered = [
            (cls, conf) for cls, conf in zip(brand_classes, brand_confs) if conf >= 0.7
        ]
        if filtered:
            best_cls, best_conf = max(filtered, key=lambda x: x[1])
            best_brand_name = model_brand.names[int(best_cls)]
            best_brand_conf = best_conf
            print(f"감지된 브랜드: {best_brand_name} (conf: {best_brand_conf:.2f})")
        else:
            print("conf 0.7 이상 브랜드 감지 없음")
    else:
        print("브랜드 감지: 없음")

    print(f"헬멧 감지: {helmet_detected} (conf: {helmet_conf:.2f}), "
          f"브랜드 감지: {brand_detected}")

    # 감지된 객체 시각화 예시 (conf 값 함께 표시)
    def draw_boxes(results, image, color=(0, 255, 0), label=""):
        if results[0].boxes is not None:
            for box, conf in zip(results[0].boxes.xyxy.cpu().numpy(),
                                 results[0].boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                if label:
                    text = f"{label} {conf:.2f}"
                    cv2.putText(
                        image, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                    )

    draw_boxes(helmet_results, image, (0, 0, 255), "Helmet")

    cv2.imwrite("output/annotated_result.jpg", image)

else:
    print("킥보드와 사람이 모두 감지되지 않음")
