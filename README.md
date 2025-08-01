# 🛵 KTS_YOLO - 안전규제 위반 전동킥보드 자동 감지 시스템

YOLOv11 기반 객체 인식 + 다중 객체 추적 + 지리 정보 변환을 결합한 **안전규제 위반 전동킥보드 자동 단속 시스템**입니다.  
CCTV 영상 분석을 통해 **킥보드가 인도에서 주행 중인 경우**, 해당 이미지를 **킥보드+사람을 그룹화하여** 캡처하고 Firebase에 업로드합니다.
AI 기반 통합 분석 시스템인 **KTS_AI_Analysis**의 하위 모듈로, **객체 탐지**와 **CCTV 영상 분석 및 자동 신고** 하는 데 활용됩니다.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python"/>
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
- **연계 모듈**: `KTS_AI_Analysis`

---

## 📌 핵심 기능

### 🧠 객체 인식 (YOLOv11)
- 킥보드, 사람, 헬멧 인식 및 킥보드 색상을 통한 브랜드 분석
- 4개의 YOLOv11 모델 사용:
  - `kickboard_yolov11l.pt` : 사진 속 전동킥보드 인식
  - `drone_kickboard_yolov11l.pt` : 영상 속 전동킥보드 인식(저화질 전용 모델)
  - `person_yolov11l.pt` : 사람 인식
  - `helmet_yolov11l.pt` : 헬멧 인식
  - `kickboardBrand_yolov11l.pt` : 킥보드 브랜드 분석

### 🛰️ 객체 추적 (ByteTrack)
- 프레임 간 동일한 킥보드 추적
- 일정 시간 이상 트래킹되고 인도 영역에 객체 하단부가 들어가면 감지

### 📍 위치 기반 기능
- 다각형으로 인도 영역 수동 지정 (마우스로 영상 클릭)
- 해당 영역 내 주행 시 위반 간주

### ☁️ Firebase 업로드
- Firestore에 위반 기록 업로드
- Storage에 감지된 이미지 저장 및 공개 URL 생성

---

## 📂 파일 구성
```
KTS_YOLO/
├── YOLO.py # YOLO 감지 모듈 (킥보드, 사람, 브랜드, 헬멧 분석)
├── geocoding.py # 위경도 좌표 → 지번 주소 변환 (VWorld API)
├── yolo_Video.py # 영상 전체 파이프라인 처리 및 Firebase 업로드
├── firebase_config.py # Firebase 인증 설정
├── YOLO/ # 학습된 YOLO 모델들 (.pt 파일)
├── output/ # 감지 이미지 저장 디렉토리
└── ...
```

---

## 🧩 주요 모듈 설명
### 📄 YOLO.py
YOLOv11로 학습된 킥보드, 사람, 헬멧, 브랜드 모델을 로드하여 이미지에서 객체를 감지하고 분석하는 모듈입니다. 킥보드, 사람, 브랜드, 헬멧 여부를 개별적으로 판단하는 함수(`kickboard_analysis`, `person_analysis`, `brand_analysis`, `helmet_analysis`)를 제공하며, `draw_boxes`를 통해 감지 결과를 시각화할 수 있습니다.
### 📄 geocoding.py
VWorld API를 사용하여 GPS 좌표를 지번 주소로 변환하는 모듈입니다. `gps()` 함수는 위도·경도를 입력받아 `reverse_geocode()`를 호출해 주소 문자열을 반환합니다. Firebase 업로드 시 위치 정보 제공에 활용됩니다.
### 📄 yolo_Video.py
CCTV 영상 데이터를 입력받아 YOLOv11 + ByteTrack 기반 객체 감지 및 추적을 수행하는 메인 파이프라인 모듈입니다. 사용자 마우스 입력으로 인도 영역(다각형)을 지정하고, 일정 조건(**시간: 2초 이상 / 위치: 인도 / 신뢰도: 0.7 이상**)을 만족하는 킥보드 및 사람 객체를 캡처한 후 Firebase Storage와 Firestore에 업로드합니다.

---

## 🎬 데모 미리보기

<p align="center">
  <img src="assets/Real-time tracking.gif" width="400"/>
  <img src="assets/Real-time tracking_1.gif" width="400"/>
</p>

> YOLOv11 + ByteTrack 기반으로 CCTV에서 킥보드를 실시간으로 감지 및 추적합니다.

<p align="center">
  <img src="assets/cctv image detection system.gif" width="500"/>
</p>

> 사용자가 마우스로 인도 영역을 지정하고, 해당 영역 내 주행하는 킥보드를 감지합니다.

<p align="center">
  <img src="assets/Photos automatically reported.jpg" style="max-height:400px; width:auto;"/>
</p>

> 감지 조건을 만족한 킥보드 + 사람 객체는 이미지로 캡처되어 Firebase에 저장됩니다.

---

## 🖥 실행 방법

### 1. 환경 설정
```bash
pip install -r requirements.txt
```
- .env 파일 생성:
```bash
VWorld_API=여기에_발급받은_API_키
```
### 2. 실행
```bash
python yolo_Video.py
```
- 마우스로 인도 영역을 다각형으로 지정
- Enter 키로 확정 → 감지 시작
- q 키로 종료

---

## 💡 향후 개선점

- 🛴 사람 + 킥보드 동시 트랙킹 로직 강화  
  동일 프레임 내 **2인 이상**이 킥보드 근처에 존재할 경우, **‘2인 탑승’으로 간주**하는 고급 감지 로직 적용

---

## 📦 통합 프로젝트 구조 안내

본 프로젝트는 전체 AI 기반 분석 시스템인 **AI_Analysis**의 하위 모듈 중 하나입니다.

`KTS_YOLO`는 **AI_Analysis** 내에서 다음과 같은 역할을 수행합니다:

- YOLOv11 기반 객체 감지 (킥보드, 사람 등)
- ByteTrack 기반 객체 추적
- CCTV 영상을 통해 인도 내 주행 판단 및 이미지 캡처
- 감지된 영상 장면을 Firebase에 업로드하여 실시간 기록 저장

👉 전체 파이프라인은 **[KTS_AI_Analysis](https://github.com/ParkerQH/KTS_AI_Analysis)** 리포지토리에서 확인할 수 있습니다.

---

## 📬 Contact

- **개발자**: 박창률  
- **GitHub**: [ParkerQH](https://github.com/ParkerQH)
