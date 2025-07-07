import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

model_kickboard = YOLO("YOLO/drone_yolov11m.pt")
model_person = YOLO("YOLO/person_yolov11m.pt")
tracker = "YOLO/bytetrack.yaml"
STATIONARY_FRAMES = 10
MOVE_THRESHOLD = 5

track_history = defaultdict(lambda: deque(maxlen=STATIONARY_FRAMES))
track_duration = defaultdict(int)
captured_ids = set()

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

    results = model_kickboard.track(
        frame,
        persist=True,
        tracker=tracker,
        verbose=False,
        conf=0.3,
        iou=0.4
    )

    annotated_frame = frame.copy()

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else [None]*len(boxes)
        clss = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for box, obj_id, cls, conf in zip(boxes, ids, clss, confs):
            if int(cls) != 0:  # 킥보드 클래스만
                continue
            if obj_id is None:
                continue

            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            track_history[obj_id].append((cx, cy))

            # 멈춰있는 객체는 더 이상 감지 제외
            if len(track_history[obj_id]) == STATIONARY_FRAMES:
                coords = np.array(track_history[obj_id])
                dists = np.linalg.norm(coords - coords[0], axis=1)
                max_move = np.max(dists)
                if max_move < MOVE_THRESHOLD:
                    continue

            track_duration[obj_id] += 1

            # 2초 이상 유지 & conf 0.7 이상 & 중복 저장 방지
            if (
                track_duration[obj_id] >= CAPTURE_FRAMES
                and obj_id not in captured_ids
                and conf >= 0.7
            ):
                x1, y1, x2, y2 = box
                pad = 80  # 넓은 영역으로 확장
                x1 = max(int(x1) - pad, 0)
                y1 = max(int(y1) - pad, 0)
                x2 = min(int(x2) + pad, frame.shape[1])
                y2 = min(int(y2) + pad, frame.shape[0])

                # 1. 해당 프레임에서 사람 모델로 한번만 추가 추론
                person_results = model_person(frame)
                person_boxes = []
                if person_results[0].boxes is not None:
                    p_boxes = person_results[0].boxes.xyxy.cpu().numpy()
                    p_clss = person_results[0].boxes.cls.cpu().numpy()
                    for p_box, p_cls in zip(p_boxes, p_clss):
                        if int(p_cls) == 0:  # 사람 클래스 번호(모델에 따라 다를 수 있음)
                            person_boxes.append(p_box)

                # 2. 킥보드 박스와 겹치는 사람 박스가 있으면 합집합으로 확장
                for pbox in person_boxes:
                    px1, py1, px2, py2 = map(int, pbox)
                    pw, ph = px2 - px1, py2 - py1
                    # 너무 큰 사람 박스는 무시
                    if pw > frame.shape[1] * 0.5 or ph > frame.shape[0] * 0.5:
                        continue
                    # 킥보드와 겹치는 경우만 확장
                    if not (px2 < x1 or px1 > x2 or py2 < y1 or py1 > y2):
                        x1 = min(x1, px1)
                        y1 = min(y1, py1)
                        x2 = max(x2, px2)
                        y2 = max(y2, py2)
                # 확장 영역이 너무 크면 킥보드 박스만 사용
                crop_w, crop_h = x2 - x1, y2 - y1
                if crop_w > frame.shape[1] * 0.7 or crop_h > frame.shape[0] * 0.7:
                    x1, y1, x2, y2 = box

                crop_img = frame[y1:y2, x1:x2]
                save_path = os.path.join(output_dir, f"kickboard_id{int(obj_id)}_frame{frame_count}_conf{conf:.2f}.jpg")
                cv2.imwrite(save_path, crop_img)
                captured_ids.add(obj_id)

            # confidence 점수 표시 (소수점 둘째자리)
            label = f"Kickboard(ID:{int(obj_id)}) conf:{conf:.2f}"
            cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)
            cv2.putText(annotated_frame, label, (int(box[0]), int(box[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("YOLOv11 Kickboard Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
