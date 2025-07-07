import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

# YOLOv11 모델 로드
model = YOLO("YOLO_pt/drone_yolov11m.pt")

# 트래커 설정 (YOLO의 track 모드 활용)
tracker = "bytetrack.yaml"  # 기본 트래커 사용

# 주차 간주 임계값
STATIONARY_FRAMES = 10      # 몇 프레임 연속 움직임 없으면 주차로 간주
MOVE_THRESHOLD = 5          # 픽셀 단위 이동량 임계값

# 객체별 이동 기록 저장용
track_history = defaultdict(lambda: deque(maxlen=STATIONARY_FRAMES))

# 동영상 파일 열기
video_path = "Data/DJI_0086.MP4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # YOLO 트래킹 실행 (persist=True로 ID 일관성 유지)
    results = model.track(
        frame,
        persist=True,
        tracker=tracker,
        verbose=False,
        conf=0.3,   # 감지 신뢰도 임계값
        iou=0.4     # NMS IOU 임계값
    )

    annotated_frame = frame.copy()

    # 감지된 객체 정보 추출
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()   # [x1, y1, x2, y2]
        ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else [None]*len(boxes)
        clss = results[0].boxes.cls.cpu().numpy()     # 클래스 번호

        for box, obj_id, cls in zip(boxes, ids, clss):
            # 킥보드 클래스만 처리 (예: 클래스 0이 킥보드라면, 필요시 번호 확인)
            if int(cls) != 0:
                continue

            # 바운딩박스 중심 좌표
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)

            # 이동 기록 갱신
            track_history[obj_id].append((cx, cy))

            # 주차 여부 판단
            if len(track_history[obj_id]) == STATIONARY_FRAMES:
                # 최근 N프레임 중심 좌표의 최대 이동 거리 계산
                coords = np.array(track_history[obj_id])
                dists = np.linalg.norm(coords - coords[0], axis=1)
                max_move = np.max(dists)
                if max_move < MOVE_THRESHOLD:
                    continue  # 주차된 킥보드이므로 표시하지 않음

            # 움직이는 킥보드만 표시
            cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)
            cv2.putText(annotated_frame, f"Kickboard(ID:{int(obj_id)})", (int(box[0]), int(box[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # 결과 화면에 표시
    cv2.imshow("YOLOv11 Kickboard Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
