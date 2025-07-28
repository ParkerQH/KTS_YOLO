import os
import cv2
import numpy as np
from ultralytics import YOLO
from mss import mss
from collections import defaultdict, deque
import uuid
from datetime import datetime
import pytz
import pygetwindow as gw

# (파이어베이스 연동 부분 필요시 활성화)
# import firebase_config
# from firebase_admin import storage, firestore
# db_fs = firestore.client()
# bucket = storage.bucket()

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

base_dir = os.path.dirname(os.path.abspath(__file__))
model_kickboard = YOLO(os.path.join(base_dir, "YOLO", "drone_yolov11l(2).pt"))
model_person = YOLO(os.path.join(base_dir, "YOLO", "person_yolov11l(2).pt"))
tracker = os.path.join(base_dir, "YOLO", "bytetrack.yaml")

STATIONARY_FRAMES = 10
MOVE_THRESHOLD = 5

track_history = defaultdict(lambda: deque(maxlen=STATIONARY_FRAMES))
track_duration = defaultdict(int)
captured_ids = set()
frame_count = 0
FPS = 30  # 화면 캡처는 FPS 자동계산이 어려워 기본 값 사용
CAPTURE_FRAMES = int(FPS * 2)

def get_unique_uuid(file_extension):
    while True:
        random_file_id = uuid.uuid4().hex
        storage_path = f"Report/{random_file_id}{file_extension}"
        # if bucket.blob(storage_path).exists():
        #     continue
        # if db_fs.collection("Report").document(random_file_id).get().exists:
        #     continue
        return random_file_id, storage_path

# 1. 현재 열려 있는 창 타이틀 표시 및 사용자 선택
def select_window():
    titles = [title for title in gw.getAllTitles() if title.strip() != ""]
    if not titles:
        print("열려있는 창이 없습니다.")
        return None
    print("열려있는 창 목록:")
    for i, title in enumerate(titles):
        print(f"{i}: {title}")
    while True:
        try:
            choice = int(input("분석할 창 번호를 선택하세요: "))
            if 0 <= choice < len(titles):
                return titles[choice]
            else:
                print("유효한 번호를 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")

selected_title = select_window()
if selected_title is None:
    exit()

window = gw.getWindowsWithTitle(selected_title)[0]
capture_region = {"top": window.top, "left": window.left, "width": window.width, "height": window.height}
print(f"분석 대상 창: {selected_title}")
print(f"창 좌표/크기: {capture_region}")

with mss() as sct:
    while True:
        img = np.array(sct.grab(capture_region))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
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
                    save_path = os.path.join(
                        output_dir, f"kickboard_id{int(obj_id)}_frame{frame_count}_conf{conf:.2f}.jpg")
                    cv2.imwrite(save_path, crop_img)
                    captured_ids.add(obj_id)

                    # Firebase 연동 파트(활성화 필요시)
                    # _, file_extension = os.path.splitext(save_path)
                    # random_file_id, storage_path = get_unique_uuid(file_extension)
                    # blob = bucket.blob(storage_path)
                    # blob.upload_from_filename(save_path)
                    # blob.make_public()
                    # file_url = blob.public_url
                    # lat, lon = "37.3005","127.0392"
                    # kst = pytz.timezone('Asia/Seoul')
                    # data = {
                    #     "date": datetime.now(kst),
                    #     "gpsInfo": f"{lat} {lon}",
                    #     "imageUrl": file_url,
                    #     "userId": "CCTV Report",
                    #     "violation": "CCTV 실시간 감지"
                    # }
                    # db_fs.collection("Report").document(random_file_id).set(data)

                label = f"Kickboard(ID:{int(obj_id)}) conf:{conf:.2f}"
                cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame,
                    label,
                    (int(box[0]), int(box[1])-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2
                )

        cv2.imshow(f"{selected_title} 분석", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
