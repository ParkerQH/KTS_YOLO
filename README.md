# 🛵 KTS_YOLO - 킥보드 불법 주정차 감지 시스템

YOLOv11 기반 객체 인식 + 다중 객체 추적 + 지리 정보 변환을 결합한 **안전규제 위반 전동킥보드 자동 단속 시스템**입니다.  
CCTV 영상 분석을 통해 **킥보드가 인도에서 주행 중인 경우**, 해당 이미지를 **킥보드+사람을 그룹화하여** 캡처하고 Firebase에 업로드합니다.

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

---

## 📌 핵심 기능

### 🧠 객체 인식 (YOLOv11)
- 킥보드, 사람, 헬멧, 브랜드 로고 인식
- 4개의 YOLOv11 모델 사용:
  - `kickboard_yolov11l.pt`
  - `drone_kickboard_yolov11l.pt`
  - `person_yolov11l.pt`
  - `helmet_yolov11l.pt`
  - `kickboardBrand_yolov11l.pt`

### 🛰️ 객체 추적 (ByteTrack)
- 프레임 간 동일한 킥보드 추적
- 정지 상태 판단 후 일정 시간 이상 인도에 있을 경우만 감지

### 📍 위치 기반 기능
- 다각형으로 인도 영역 수동 지정 (마우스로 영상 클릭)
- 해당 영역 내 주행 시 위반 간주

### ☁️ Firebase 업로드
- Firestore에 위반 기록 업로드
- Storage에 감지된 이미지 저장 및 공개 URL 생성

---

## 📂 파일 구성

KTS_YOLO/
├── YOLO.py # YOLO 감지 모듈 (킥보드, 사람, 브랜드, 헬멧 분석)
├── geocoding.py # 위경도 좌표 → 지번 주소 변환 (VWorld API)
├── yolo_Video.py # 영상 전체 파이프라인 처리 및 Firebase 업로드
├── firebase_config.py # Firebase 인증 설정
├── YOLO/ # 학습된 YOLO 모델들 (.pt 파일)
├── output/ # 감지 이미지 저장 디렉토리
└── ...

---

## 🖥 실행 방법

### 1. 환경 설정
```bash
pip install -r requirements.txt
```
.env 파일 생성:
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

### 📸 예시 이미지 (Demo)

<p align="center">
  <img src="https://github.com/user-attachments/assets/67728b33-06aa-48bb-af19-6b7428145fbc" width="600"/>
</p>

> 감지된 킥보드는 이미지로 캡처되고, **Firebase**에 자동 저장됩니다.

---

### 💡 향후 개선점

- 🛴 사람 + 킥보드 동시 트랙킹 로직 강화  
  동일 프레임 내 **2인 이상**이 킥보드 근처에 존재할 경우, **‘2인 탑승’으로 간주**하는 고급 감지 로직 적용

---

### 📬 Contact

- **개발자**: 박창률  
- **GitHub**: [ParkerQH](https://github.com/ParkerQH)
