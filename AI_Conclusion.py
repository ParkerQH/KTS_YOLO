import os
import cv2
import numpy as np
import requests
import tempfile
from ultralytics import YOLO
from firebase_admin import storage, firestore

# YOLO 모델 로드
model_kickboard = YOLO('YOLO/kickboard_yolov11s.pt')
model_person    = YOLO('YOLO/person_yolov11m.pt')
model_helmet    = YOLO('YOLO/helmet_yolov11m.pt')
model_brand     = YOLO('YOLO/kickboardBrand_yolov11m.pt')

def download_image(url):
    """이미지 URL에서 이미지를 다운로드해 numpy array로 반환"""
    resp = requests.get(url, stream=True)
    if resp.status_code == 200:
        img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    else:
        print(f"🚫 이미지 다운로드 실패: {url}")
        return None

def process_image(image_url, date, user_id, violation, doc_id):
    print(f"🔥 이미지 처리 시작: {image_url}")
    image = download_image(image_url)
    if image is None:
        print("🚫 이미지 로드 실패, 건너뜀")
        return

# Firestore 실시간 리스너 설정
def on_snapshot(col_snapshot, changes, read_time):
    # 초기 스냅샷은 무시 (최초 1회 실행 시 건너뜀)
    # if not hasattr(on_snapshot, "initialized"):
    #     on_snapshot.initialized = True
    #     return

    for change in changes:
        if change.type.name == "ADDED":
            doc_id = change.document.id
            doc_data = change.document.to_dict()
            if "imageUrl" in doc_data:
                print(f"🔥 새로운 신고 감지  : {doc_id}")
                violation = doc_data.get("violation", "")
                if isinstance(violation, list):
                    # 배열이면 문자열로 합침
                    violation = ", ".join(violation)
                process_image(
                    doc_data["imageUrl"],
                    doc_data.get("date", ""),
                    doc_data.get("userId", ""),
                    violation,
                    doc_id
                )

if __name__ == "__main__":
    import time
    import firebase_config
    from firebase_admin import firestore

    db_fs = firestore.client()
    report_col = db_fs.collection("Report")
    listener = report_col.on_snapshot(on_snapshot)

    print("🔥 Firestore 실시간 감지 시작 (종료: Ctrl+C) 🔥")

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        listener.unsubscribe()
        print("\n🛑 Firestore 실시간 감지를 종료합니다.")