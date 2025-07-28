import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import uuid
from datetime import datetime
import pytz

# (파이어베이스 연동 필요시 주석 해제)
# import firebase_config
# from firebase_admin import storage, firestore
# db_fs = firestore.client()
# bucket = storage.bucket()

# output 폴더 생성
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 모델 로드
base_dir = os.path.dirname(os.path.abspath(__file__))
model_kickboard = YOLO(os.path.join(base_dir, "YOLO", "drone_yolov11l(2).pt"))
model_person = YOLO(os.path.join(base_dir, "YOLO", "person_yolov11l(2).pt"))
tracker = os.path.join(base_dir, "YOLO", "bytetrack.yaml")

# 트래킹 설정
STATIONARY_FRAMES = 10
MOVE_THRESHOLD = 5
track_history = defaultdict(lambda: deque(maxlen=STATIONARY_FRAMES))
track_duration = defaultdict(int)
captured_ids = set()

# 영상 로드
video_path = "Data/DJI_0089.MP4"
cap = cv2.VideoCapture(video_path)
frame_count = 0
FPS = cap.get(cv2.CAP_PROP_FPS) or 30
CAPTURE_FRAMES = int(FPS * 2)

# ---------------- 인도 영역 설정 ----------------
sidewalk_polygon = []
polygon_done = False

ret, first_frame = cap.read()
if not ret:
    print("영상을 불러올 수 없습니다.")
    cap.release()
    exit()

def click_event(event, x, y, flags, param):
    global polygon_done
    if polygon_done:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        sidewalk_polygon.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and len(sidewalk_polygon) >= 3:
        polygon_done = True

cv2.namedWindow("Kickboard Detection")
cv2.setMouseCallback("Kickboard Detection", click_event)

# 다각형 그리기 루프
while not polygon_done:
    temp = first_frame.copy()
    for pt in sidewalk_polygon:
        cv2.circle(temp, pt, 5, (0, 0, 255), -1)
    if len(sidewalk_polygon) > 1:
        cv2.polylines(temp, [np.array(sidewalk_polygon)], False, (255, 0, 0), 2)
    cv2.putText(temp, "좌클릭: 점 추가 / 우클릭: 종료", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imshow("Kickboard Detection", temp)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC로도 종료 가능
        break

# ---------------- 분석 루프 ----------------
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    results = model_kickboard.track(frame, persist=True, tracker=tracker, verbose=False, conf=0.3, iou=0.4)
    annotated_frame = frame.copy()

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else [None]*len(boxes)
        clss = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for box, obj_id, cls, conf in zip(boxes, ids, clss, confs):
            if int(cls) != 0 or obj_id is None:
                continue

            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            track_history[obj_id].append((cx, cy))

            if len(track_history[obj_id]) == STATIONARY_FRAMES:
                coords = np.array(track_history[obj_id])
                dists = np.linalg.norm(coords - coords[0], axis=1)
                if np.max(dists) < MOVE_THRESHOLD:
                    continue

            track_duration[obj_id] += 1

            foot_point = (int((box[0] + box[2]) / 2), int(box[3]))
            is_on_sidewalk = cv2.pointPolygonTest(np.array(sidewalk_polygon), foot_point, False) >= 0

            if track_duration[obj_id] >= CAPTURE_FRAMES and obj_id not in captured_ids and conf >= 0.7 and is_on_sidewalk:
                x1, y1, x2, y2 = map(int, box)
                pad = 50
                x1 = max(x1 - pad, 0)
                y1 = max(y1 - pad, 0)
                x2 = min(x2 + pad, frame.shape[1])
                y2 = min(y2 + pad, frame.shape[0])

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
                    if px2 < x1 or px1 > x2 or py2 < y1 or py1 > y2:
                        continue
                    x1 = min(x1, px1)
                    y1 = min(y1, py1)
                    x2 = max(x2, px2)
                    y2 = max(y2, py2)

                crop_img = frame[y1:y2, x1:x2]
                save_path = os.path.join(output_dir, f"kickboard_id{int(obj_id)}_frame{frame_count}_conf{conf:.2f}.jpg")
                cv2.imwrite(save_path, crop_img)
                captured_ids.add(obj_id)

                # 파이어베이스 업로드 (필요 시 주석 해제)
                # def get_unique_uuid(file_extension):
                #     while True:
                #         random_file_id = uuid.uuid4().hex
                #         storage_path = f"Report/{random_file_id}{file_extension}"
                #         if bucket.blob(storage_path).exists():
                #             continue
                #         if db_fs.collection("Report").document(random_file_id).get().exists:
                #             continue
                #         return random_file_id, storage_path
                # _, file_extension = os.path.splitext(save_path)
                # random_file_id, storage_path = get_unique_uuid(file_extension)
                # blob = bucket.blob(storage_path)
                # blob.upload_from_filename(save_path)
                # blob.make_public()
                # file_url = blob.public_url
                # data = {
                #     "date": datetime.now(pytz.timezone("Asia/Seoul")),
                #     "gpsInfo": "37.3005 127.0392",
                #     "imageUrl": file_url,
                #     "userId": "CCTV Report",
                #     "violation": "CCTV 실시간 감지"
                # }
                # db_fs.collection("Report").document(random_file_id).set(data)

            label = f"Kickboard(ID:{int(obj_id)}) conf:{conf:.2f}"
            cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 인도 다각형 시각화
    if len(sidewalk_polygon) > 1:
        cv2.polylines(annotated_frame, [np.array(sidewalk_polygon)], isClosed=True, color=(0, 0, 255), thickness=2)

    cv2.imshow("Kickboard Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
