# 🏀 Basketball Shooting AI Coach

> YOLOv8 + MediaPipe를 융합한 농구 슈팅 폼 분석 AI 코칭 시스템

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)](https://github.com/ultralytics/ultralytics)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-green)](https://mediapipe.dev/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)](https://streamlit.io/)

---

## 프로젝트 개요

농구 슈팅 영상을 업로드하면 AI가 자동으로 **릴리즈 포인트(공이 손에서 떠나는 순간)** 를 감지하고, 해당 시점의 **팔꿈치 각도**와 **무릎 각도**를 분석해 코칭 피드백을 제공하는 웹 애플리케이션입니다.

---

## 주요 기능

- **농구공 실시간 탐지** — 커스텀 학습된 YOLOv8 모델로 영상 내 농구공 위치를 프레임마다 추적
- **자세 추정** — MediaPipe로 슈터의 어깨, 팔꿈치, 손목, 골반, 무릎, 발목 관절 좌표 추출
- **릴리즈 포인트 자동 감지** — 손목-공 사이 유클리디안 거리(L2 Norm)가 임계값 이상 벌어지는 순간을 릴리즈로 판정
- **각도 분석** — `acos` 벡터 연산으로 팔꿈치·무릎 각도를 0~180° 범위로 계산
- **웹 인터페이스** — Streamlit 기반 UI에서 영상 업로드 후 분석 결과 즉시 확인

---

## 기술 스택

| 역할 | 기술 |
|------|------|
| 객체 탐지 | YOLOv8 Nano (Ultralytics) |
| 자세 추정 | MediaPipe Pose |
| 영상 처리 | OpenCV |
| 수치 연산 | NumPy |
| 웹 UI | Streamlit |
| 학습 환경 | Google Colab (GPU) |
| 데이터셋 | Roboflow (농구공·골대 약 1,000장) |

---

## 분석 로직 흐름

```
슈팅 영상 입력
    │
    ▼
OpenCV 프레임 루프
    ├── YOLO 탐지 → 농구공 중심 좌표 (bx, by)
    └── MediaPipe → 관절 좌표 추출 (어깨·팔꿈치·손목·골반·무릎·발목)
    │
    ▼
융합 로직
    └── 유클리디안 거리(손목-공) ≥ 임계값 → 릴리즈 포인트 감지
    │
    ▼
각도 계산 (acos 벡터 연산)
    ├── 팔꿈치 각도: 어깨-팔꿈치-손목
    └── 무릎 각도:   골반-무릎-발목
    │
    ▼
피드백 출력
    ├── 팔꿈치 85~105° → Perfect! / Adjust Elbow
    └── 무릎 150° 이상 → Good Leg! / Bend Knees More
```

---

## 설치 및 실행

### 1. 저장소 클론

```bash
git clone https://github.com/본인계정/basketball-coach.git
cd basketball-coach
```

### 2. 가상환경 생성 및 활성화

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. 모델 파일 준비

학습된 YOLO 가중치 파일 `best.pt`를 프로젝트 루트에 위치시키세요.

> `best.pt`는 파일 크기(약 6MB)로 인해 GitHub에 포함되지 않습니다.  
> 👉 [Google Drive에서 다운로드](#) ← 본인 링크로 교체

### 5. 웹 앱 실행

```bash
streamlit run app.py
# 또는
python -m streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

---

## 파일 구조

```
basketball-coach/
├── app.py                  # Streamlit 웹 애플리케이션
├── pose_logic.py           # 핵심 분석 로직 (YOLO + MediaPipe 융합)
├── best.pt                 # YOLOv8 커스텀 학습 가중치 (GitHub 미포함)
├── requirements.txt        # 패키지 의존성
├── my_shot.mp4             # 테스트 영상 (GitHub 미포함)
├── dataset/                # 학습 데이터셋 (GitHub 미포함)
│   ├── train/
│   ├── valid/
│   └── test/
└── README.md
```

---

## requirements.txt

```
ultralytics==8.2.0
mediapipe==0.10.14
opencv-python-headless==4.9.0.80
numpy==1.26.4
streamlit==1.35.0
```

---

## 분석 결과 예시

| 항목 | 측정값 | 판정 |
|------|--------|------|
| 팔꿈치 각도 | 93.4° | Perfect! |
| 무릎 각도 | 162.1° | Good Leg! |

---

## 개발 과정

| 단계 | 내용 |
|------|------|
| 1단계 | Roboflow로 농구공·골대 데이터셋 구축, Google Colab에서 YOLOv8 Nano 100 Epoch 학습 |
| 2단계 | YOLO + MediaPipe 융합 로직 설계, 릴리즈 포인트 감지 알고리즘 구현 |
| 3단계 | Python 3.11 가상환경 구축, 라이브러리 호환성 문제 해결 |
| 4단계 | Streamlit 웹 UI 구현, 영상 업로드 및 실시간 분석 대시보드 완성 |

---

## 주의사항

- Python **3.11** 환경에서 개발 및 테스트되었습니다. 3.12 이상에서는 일부 라이브러리 호환성 문제가 발생할 수 있습니다.
- 측면에서 촬영된 영상에서 가장 정확한 분석 결과를 제공합니다.
- 슈터의 전신이 화면에 들어오도록 촬영하면 관절 감지 정확도가 높아집니다.

---

## 라이선스

This project is licensed under the MIT License.