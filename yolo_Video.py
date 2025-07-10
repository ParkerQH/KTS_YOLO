import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import uuid
from datetime import datetime
import pytz

import firebase_config
from firebase_admin import storage, firestore

db_fs = firestore.client()
bucket = storage.bucket()

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

model_kickboard = YOLO("YOLO/drone_yolov11l(2).pt")
model_person = YOLO("YOLO/person_yolov11l(2).pt")
tracker = "YOLO/bytetrack.yaml"
STATIONARY_FRAMES = 10
MOVE_THRESHOLD = 5

track_history = defaultdict(lambda: deque(maxlen=STATIONARY_FRAMES))
track_duration = defaultdict(int)
captured_ids = set()

video_path = "Data/DJI_0089.MP4"
cap = cv2.VideoCapture(video_path)

frame_count = 0
FPS = cap.get(cv2.CAP_PROP_FPS)
CAPTURE_FRAMES = int(FPS * 2)  # 2초 유지 기준

def get_unique_uuid(file_extension):
    """UUID 중복 체크(스토리지, 파이어스토어 모두) 후 중복 없을 때까지 새로 생성"""
    while True:
        random_file_id = uuid.uuid4().hex
        storage_path = f"Report/{random_file_id}{file_extension}"

        # 1. Storage 중복 체크
        if bucket.blob(storage_path).exists():
            continue

        # 2. Firestore 중복 체크
        if db_fs.collection("Report").document(random_file_id).get().exists:
            continue

        return random_file_id, storage_path

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
            if int(cls) != 0:
                continue
            if obj_id is None:
                continue

            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            track_history[obj_id].append((cx, cy))

            if len(track_history[obj_id]) == STATIONARY_FRAMES:
                coords = np.array(track_history[obj_id])
                dists = np.linalg.norm(coords - coords[0], axis=1)
                max_move = np.max(dists)
                if max_move < MOVE_THRESHOLD:
                    continue

            track_duration[obj_id] += 1

            # 트랙킹 2초 이상, conf 0.7 이상
            if (
                track_duration[obj_id] >= CAPTURE_FRAMES
                and obj_id not in captured_ids
                and conf >= 0.7
            ):
                x1, y1, x2, y2 = box
                pad = 50
                x1 = max(int(x1) - pad, 0)
                y1 = max(int(y1) - pad, 0)
                x2 = min(int(x2) + pad, frame.shape[1])
                y2 = min(int(y2) + pad, frame.shape[0])

                person_results = model_person(frame)
                person_boxes = []
                if person_results[0].boxes is not None:
                    p_boxes = person_results[0].boxes.xyxy.cpu().numpy()
                    p_clss = person_results[0].boxes.cls.cpu().numpy()
                    for p_box, p_cls in zip(p_boxes, p_clss):
                        if int(p_cls) == 0:
                            person_boxes.append(p_box)

                for pbox in person_boxes:
                    px1, py1, px2, py2 = map(int, pbox)
                    pw, ph = px2 - px1, py2 - py1
                    if pw > frame.shape[1] * 0.5 or ph > frame.shape[0] * 0.5:
                        continue
                    if not (px2 < x1 or px1 > x2 or py2 < y1 or py1 > y2):
                        x1 = min(x1, px1)
                        y1 = min(y1, py1)
                        x2 = max(x2, px2)
                        y2 = max(y2, py2)
                crop_w, crop_h = x2 - x1, y2 - y1
                if crop_w > frame.shape[1] * 0.7 or crop_h > frame.shape[0] * 0.7:
                    x1, y1, x2, y2 = box

                crop_img = frame[y1:y2, x1:x2]
                save_path = os.path.join(output_dir, f"kickboard_id{int(obj_id)}_frame{frame_count}_conf{conf:.2f}.jpg")
                cv2.imwrite(save_path, crop_img)
                captured_ids.add(obj_id)

                # ---- Firebase Storage & Firestore 연동 ----
                # 파일 확장자 추출
                _, file_extension = os.path.splitext(save_path)

                # 중복 없는 UUID 및 storage_path 생성
                random_file_id, storage_path = get_unique_uuid(file_extension)

                # Storage에 파일 업로드
                blob = bucket.blob(storage_path)
                blob.upload_from_filename(save_path)
                blob.make_public()
                file_url = blob.public_url

                # GPS 정보(CCTV 고정값)
                lat, lon = "37.3005","127.0392"
                kst = pytz.timezone('Asia/Seoul')

                data = {
                    "date": datetime.now(kst),
                    "gpsInfo": f"{lat} {lon}",
                    "imageUrl": file_url,
                    "userId": "CCTV Report",
                    "violation": "CCTV 실시간 감지"
                }

                # Firestore에 저장 (컬렉션: Report, 문서: random_file_id)
                db_fs.collection("Report").document(random_file_id).set(data)

            label = f"Kickboard(ID:{int(obj_id)}) conf:{conf:.2f}"
            cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)
            cv2.putText(annotated_frame, label, (int(box[0]), int(box[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("YOLOv11 Kickboard Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
