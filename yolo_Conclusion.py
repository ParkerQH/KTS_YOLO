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

    # 1. 킥보드 감지
    kickboard_results = model_kickboard(image)
    kickboard_detected = (
        kickboard_results[0].boxes is not None and 
        len(kickboard_results[0].boxes) > 0
    )

    # 2. 사람 감지
    person_results = model_person(image)
    person_detected = (
        person_results[0].boxes is not None and 
        len(person_results[0].boxes) > 0
    )

    print(f"🔥 킥보드 감지: {kickboard_detected}, 사람 감지: {person_detected}")

    # 3. 킥보드와 사람이 모두 있을 때만 헬멧/브랜드 감지
    if kickboard_detected and person_detected:
        # 3-1. 헬멧 감지
        helmet_results = model_helmet(image)
        helmet_detected = (
            helmet_results[0].boxes is not None and 
            len(helmet_results[0].boxes) > 0
        )
        # 3-2. 브랜드 감지
        brand_results = model_brand(image)
        brand_detected = (
            brand_results[0].boxes is not None and 
            len(brand_results[0].boxes) > 0
        )
        # conf 0.7 이상 중 최고 신뢰도 브랜드 한 개만 추출
        if brand_results[0].boxes is not None and len(brand_results[0].boxes) > 0:
            brand_classes = brand_results[0].boxes.cls.cpu().numpy()
            brand_confs   = brand_results[0].boxes.conf.cpu().numpy()
            filtered = [(cls, conf) for cls, conf in zip(brand_classes, brand_confs) if conf >= 0.7]
            if filtered:
                best_cls, best_conf = max(filtered, key=lambda x: x[1])
                best_brand_name = model_brand.names[int(best_cls)]
                print(f"🔥 감지된 브랜드: {best_brand_name} (conf: {best_conf:.2f})")
            else:
                print("🚫 conf 0.7 이상 브랜드 감지 없음")
        else:
            print("🚫 브랜드 감지: 없음")

        print(f"🔥 헬멧 감지: {helmet_detected}, 브랜드 감지: {brand_detected}")

        # (선택) 감지 결과 시각화 및 저장
        def draw_boxes(results, image, color=(0,255,0), label=''):
            if results[0].boxes is not None:
                for box in results[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    if label:
                        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        draw_boxes(helmet_results, image, (0,0,255), 'Helmet')
        # cv2.imwrite(f'output/annotated_{doc_id}.jpg', image)

        # 분석 이미지 저장 (Storage)
        bucket = storage.bucket()
        conclusion_blob = bucket.blob(f"Conclusion/{doc_id}.jpg")

        # 임시 파일 생성 (분석 이미지용)
        _, temp_annotated = tempfile.mkstemp(suffix=".jpg")
        cv2.imwrite(temp_annotated, image)
        conclusion_blob.upload_from_filename(temp_annotated)
        conclusion_url = conclusion_blob.public_url

        # 사진 지번 주소 출력
        api_key = os.getenv("VWorld_API")
        db_fs = firestore.client()
        doc_ref = db_fs.collection("Report").document(doc_id)
        doc = doc_ref.get()
        if doc.exists:
            doc_data = doc.to_dict()
            gps_info = doc_data.get("gpsInfo")
        if gps_info:
            lat_str, lon_str = gps_info.strip().split()
            lat = float(lat_str)
            lon = float(lon_str)
            parcel_addr = reverse_geocode(lat, lon, api_key)

        # Firestore에 결과 저장
        doc_id = f"conclusion_{doc_id}"  # 문서 ID 생성
        conclusion_data = {
            "date" : date,
            "userId" : user_id,
            "aiConclusion" : traffic_violation_detection,
            "violation": violation,
            "confidence": top_helmet_confidence,
            "detectedBrand": top_class,
            "imageUrl": conclusion_url,
            "region": parcel_addr,
            "gpsInfo": f"{lat} {lon}",
            "reportImgUrl": imageUrl
        }

        if traffic_violation_detection in ("사람 감지 실패", "킥보드 감지 실패"):
            conclusion_data.update({
                "result": "반려",
                "reason": traffic_violation_detection
            })
        else :
            conclusion_data.update({
                "result": "미확인"
            })

        db_fs.collection("Conclusion").document(doc_id).set(conclusion_data)

        print(f"✅ 분석된 사진 url : {imageUrl}\n")

    else:
        print("🚫 킥보드 또는 사람이 감지되지 않음")

def reverse_geocode(lat, lon, api_key):
    url = "https://api.vworld.kr/req/address"
    params = {
        "service": "address",
        "request": "getAddress",
        "crs": "epsg:4326",
        "point": f"{lon},{lat}",
        "format": "json",
        "type": "parcel",
        "key": api_key,
    }
    response = requests.get(url, params=params)

    # 반환값 단순화
    if response.status_code == 200:
        data = response.json()
        if data["response"]["status"] == "OK":
            # 첫 번째 결과에서 지번주소 추출
            result = data["response"]["result"][0]
            if "text" in result:
                return result["text"]  # 지번주소만 반환
    return None

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
                process_image(
                    doc_data["imageUrl"],
                    doc_data.get("date", ""),
                    doc_data.get("userId", ""),
                    doc_data.get("violation", ""),
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
