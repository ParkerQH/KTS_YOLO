import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import uuid
from datetime import datetime
import pytz

# Firebase 연동
import firebase_config
from firebase_admin import storage, firestore

db_fs = firestore.client()
bucket = storage.bucket()

# 결과 이미지 저장 폴더 생성
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# YOLO 모델(킥보드, 사람) 및 트래커 로드
base_dir = os.path.dirname(os.path.abspath(__file__))
model_kickboard = YOLO(os.path.join(base_dir, "YOLO", "drone_yolov11l(2).pt"))
model_person = YOLO(os.path.join(base_dir, "YOLO", "person_yolov11l(2).pt"))
tracker = os.path.join(base_dir, "YOLO", "bytetrack.yaml")

# 트래킹 조건 설정
STATIONARY_FRAMES = 10
MOVE_THRESHOLD = 5
track_history = defaultdict(lambda: deque(maxlen=STATIONARY_FRAMES))
track_duration = defaultdict(int)
captured_ids = set()

# 분석할 영상 경로
video_path = "Data/DJI_0089.MP4"
cap = cv2.VideoCapture(video_path)
frame_count = 0
FPS = cap.get(cv2.CAP_PROP_FPS) or 30
CAPTURE_FRAMES = int(FPS * 2)  # 2초 기준

# ---- 인도 영역 다각형 지정 ----
polygons = []            # 여러 개의 다각형
current_polygon = []     # 그리고 있는 중인 다각형
collecting = True

# Firebase에 중복 없는 UUID 경로 생성 함수
def get_unique_uuid(file_extension):
    while True:
        random_file_id = uuid.uuid4().hex
        storage_path = f"Report/{random_file_id}{file_extension}"
        if bucket.blob(storage_path).exists():
            continue
        if db_fs.collection("Report").document(random_file_id).get().exists:
            continue
        return random_file_id, storage_path

# 마우스 이벤트로 영역 지정
first_frame = None

def click_event(event, x, y, flags, param):
    global polygons, current_polygon
    if event == cv2.EVENT_LBUTTONDOWN:  # 좌클릭: 점 추가
        current_polygon.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:  # 우클릭: 현재 다각형 종료
        if current_polygon:
            polygons.append(current_polygon[:])
            current_polygon = []

# 첫 프레임 받아오기 및 마우스 콜백 설정
ret, first_frame = cap.read()
if not ret:
    print("영상을 불러올 수 없습니다.")
    cap.release()
    exit()

cv2.namedWindow("Kickboard Detection")
cv2.setMouseCallback("Kickboard Detection", click_event)

# 영역 수동 지정 루프 (Enter 누를 때까지)
while collecting:
    temp = first_frame.copy()
    for poly in polygons:
        if len(poly) > 1:
            cv2.polylines(temp, [np.array(poly)], True, (255, 0, 0), 2)
            cv2.fillPoly(temp, [np.array(poly)], (255, 0, 0, 50))
        for pt in poly:
            cv2.circle(temp, pt, 5, (0, 0, 255), -1)
    if current_polygon:
        cv2.polylines(temp, [np.array(current_polygon)], False, (0, 255, 255), 2)
        for pt in current_polygon:
            cv2.circle(temp, pt, 5, (0, 255, 255), -1)
    cv2.imshow("Kickboard Detection", temp)
    key = cv2.waitKey(1)
    if key == 13:  # Enter로 완료
        if current_polygon:
            polygons.append(current_polygon[:])
            current_polygon = []
        collecting = False
    elif key == 27:  # ESC로 종료
        cap.release()
        cv2.destroyAllWindows()
        exit()

# ---- 본격 감지 루프 ----
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    results = model_kickboard.track(
        frame, persist=True, tracker=tracker, verbose=False, conf=0.3, iou=0.4
    )

    annotated_frame = frame.copy()

    # 인도 영역 시각화
    overlay = annotated_frame.copy()
    for poly in polygons:
        cv2.fillPoly(overlay, [np.array(poly)], (255, 0, 0))
    annotated_frame = cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0)

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

            # 킥보드의 하단 중앙 좌표가 인도 위인지 확인
            foot_point = (int((box[0] + box[2]) / 2), int(box[3]))
            is_on_sidewalk = any(cv2.pointPolygonTest(np.array(poly), foot_point, False) >= 0 for poly in polygons)

            # 조건 만족 시 캡처 및 저장
            if (
                track_duration[obj_id] >= CAPTURE_FRAMES
                and obj_id not in captured_ids
                and conf >= 0.7
                and is_on_sidewalk
            ):
                x1, y1, x2, y2 = map(int, box)
                pad = 50
                x1 = max(x1 - pad, 0)
                y1 = max(y1 - pad, 0)
                x2 = min(x2 + pad, frame.shape[1])
                y2 = min(y2 + pad, frame.shape[0])

                # 사람 포함 여부 확인
                person_results = model_person(frame)
                person_boxes = []
                if person_results[0].boxes is not None:
                    for pbox, pcls in zip(person_results[0].boxes.xyxy.cpu().numpy(), person_results[0].boxes.cls.cpu().numpy()):
                        if int(pcls) == 0:
                            person_boxes.append(pbox)

                for pbox in person_boxes:
                    px1, py1, px2, py2 = map(int, pbox)
                    if px2 < x1 or px1 > x2 or py2 < y1 or py1 > y2:
                        continue
                    x1 = min(x1, px1)
                    y1 = min(y1, py1)
                    x2 = max(x2, px2)
                    y2 = max(y2, py2)

                crop_img = frame[y1:y2, x1:x2]
                save_path = os.path.join(
                    output_dir, f"kickboard_id{int(obj_id)}_frame{frame_count}_conf{conf:.2f}.jpg")
                cv2.imwrite(save_path, crop_img)
                captured_ids.add(obj_id)

                # ---- Firebase Storage & Firestore 연동 ----
                _, file_extension = os.path.splitext(save_path)
                random_file_id, storage_path = get_unique_uuid(file_extension)

                blob = bucket.blob(storage_path)
                blob.upload_from_filename(save_path)
                blob.make_public()
                file_url = blob.public_url

                lat, lon = "37.3005", "127.0392"
                kst = pytz.timezone("Asia/Seoul")
                data = {
                    "date": datetime.now(kst),
                    "gpsInfo": f"{lat} {lon}",
                    "imageUrl": file_url,
                    "userId": "CCTV Report",
                    "violation": "CCTV 실시간 감지"
                }
                db_fs.collection("Report").document(random_file_id).set(data)

            # 박스 및 텍스트 시각화
            label = f"Kickboard(ID:{int(obj_id)}) conf:{conf:.2f}"
            cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 결과 프레임 표시
    cv2.imshow("Kickboard Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
