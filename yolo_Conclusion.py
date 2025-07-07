from ultralytics import YOLO

# 모델 로드 (yolo11n.pt 또는 yolo11.pt 등)
model = YOLO("YOLO\person_yolov11m.pt")

# 분석할 이미지 경로 지정
image_path = "output\kickboard_id6_frame261.jpg"

# 객체 감지 실행
results = model(image_path)

# 결과 시각화 및 저장
results[0].show()  # 감지 결과 이미지 창에 표시
# results[0].save(filename="output.jpg")  # 결과 이미지 파일로 저장
