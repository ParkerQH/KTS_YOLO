import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

# output 폴더 생성
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# YOLOv11 모델 로드
model = YOLO("YOLO_pt/drone_yolov11m.pt")

# 트래커 설정
tracker = "YOLO_pt/bytetrack.yaml"

# 주차 간주 임계값
STATIONARY_FRAMES = 10
MOVE_THRESHOLD = 5

# 객체별 이동 기록 및 유지 시간
track_history = defaultdict(lambda: deque(maxlen=STATIONARY_FRAMES))
track_duration = defaultdict(int)
captured_ids = set()  # 중복 저장 방지

# 동영상 파일 열기
video_path = "Data/DJI_0086.MP4"
cap = cv2.VideoCapture(video_path)

frame_count = 0
FPS = cap.get(cv2.CAP_PROP_FPS)
CAPTURE_FRAMES = int(FPS * 2)  # 2초 유지 기준

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1

    # YOLO 트래킹 실행
    results = model.track(
        frame,
        persist=True,
        tracker=tracker,
        verbose=False,
        conf=0.3,
        iou=0.3
    )

    annotated_frame = frame.copy()

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else [None]*len(boxes)
        clss = results[0].boxes.cls.cpu().numpy()

        # 사람 박스도 추출
        person_boxes = [box for box, cls in zip(boxes, clss) if int(cls) == 1]  # 예: 클래스 1이 사람

        for box, obj_id, cls in zip(boxes, ids, clss):
            if int(cls) != 0:  # 킥보드 클래스만
                continue

            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            track_history[obj_id].append((cx, cy))

            # 주차된 킥보드 제외
            if len(track_history[obj_id]) == STATIONARY_FRAMES:
                coords = np.array(track_history[obj_id])
                dists = np.linalg.norm(coords - coords[0], axis=1)
                max_move = np.max(dists)
                if max_move < MOVE_THRESHOLD:
                    continue

            # 유지 시간 기록
            track_duration[obj_id] += 1

            # 3초 유지 시 캡쳐 (중복 저장 방지)
            if track_duration[obj_id] == CAPTURE_FRAMES and obj_id not in captured_ids:
                x1, y1, x2, y2 = box
                # 바운딩 박스 확장
                pad = 40  # 픽셀 단위 확장량(조절 가능)
                x1 = max(int(x1) - pad, 0)
                y1 = max(int(y1) - pad, 0)
                x2 = min(int(x2) + pad, frame.shape[1])
                y2 = min(int(y2) + pad, frame.shape[0])

                # 사람 박스가 겹치면 합집합으로 확장
                for pbox in person_boxes:
                    px1, py1, px2, py2 = map(int, pbox)
                    if not (px2 < x1 or px1 > x2 or py2 < y1 or py1 > y2):
                        x1 = min(x1, px1)
                        y1 = min(y1, py1)
                        x2 = max(x2, px2)
                        y2 = max(y2, py2)

                crop_img = frame[y1:y2, x1:x2]
                save_path = os.path.join(output_dir, f"kickboard_id{int(obj_id)}_frame{frame_count}.jpg")
                cv2.imwrite(save_path, crop_img)
                captured_ids.add(obj_id)

            # 시각화
            cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)
            cv2.putText(annotated_frame, f"Kickboard(ID:{int(obj_id)})", (int(box[0]), int(box[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("YOLOv11 Kickboard Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
