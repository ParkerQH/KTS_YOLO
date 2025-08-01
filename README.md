# 🛵 KTS_YOLO - 킥보드 불법 주정차 감지 시스템

YOLOv11 기반 객체 인식 + 다중 객체 추적 + 지리 정보 변환을 결합한 **킥보드 불법 주정차 자동 단속 시스템**입니다.  
CCTV 영상 분석을 통해 **킥보드가 인도에 장시간 정차한 경우**, 해당 이미지를 캡처하고 Firebase에 업로드합니다.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python"/>
  <img src="https://img.shields.io/badge/YOLOv11-ObjectDetection-red"/>
  <img src="https://img.shields.io/badge/ByteTrack-MultiObjectTracking-green"/>
  <img src="https://img.shields.io/badge/Firebase-Upload-yellow"/>
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen"/>
</p>

---

## 📅 프로젝트 정보

- **기간**: 2025.07.07 ~ 2025.07.25
- **개발자**: 박창률
- **주요 모듈**: `YOLO.py`, `geocoding.py`, `yolo_Video.py`

---

## 📌 핵심 기능

### 🧠 객체 인식 (YOLOv11)
- 킥보드, 사람, 헬멧, 브랜드 로고 인식
- 4개의 YOLOv11 모델 사용:
  - `kickboard_yolov11l.pt`
  - `person_yolov11l.pt`
  - `helmet_yolov11l.pt`
  - `kickboardBrand_yolov11l.pt`

### 🛰️ 객체 추적 (ByteTrack)
- 프레임 간 동일한 킥보드 추적
- 정지 상태 판단 후 일정 시간 이상 인도에 있을 경우만 감지

### 📍 위치 기반 기능
- 다각형으로 인도 영역 수동 지정 (마우스로 영상 클릭)
- 해당 영역 내 정지 시 위반 간주

### ☁️ Firebase 업로드
- Firestore에 위반 기록 업로드
- Storage에 감지된 이미지 저장 및 공개 URL 생성

---

## 📂 파일 구성

