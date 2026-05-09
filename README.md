# 3주 차 과제: YOLOv8 기반 객체 탐지(Object Detection) 모델 구축

## 📌 프로젝트 개요
본 프로젝트는 농구공(Basketball)과 농구 골대(Basketball Hoop) 데이터를 학습하여, 실제 이미지 내에서 해당 객체의 위치를 인식하고 분류하는 AI 모델을 구축하는 과정입니다. PyTorch 기반의 YOLOv8 모델을 사용하여 학습을 진행하였고, OpenCV를 통해 예측 결과를 시각화하였습니다.

## 🛠️ 개발 및 테스트 환경
* **환경:** `Python 3.14.4` / `torch-2.11.0` / CPU 기반
* **주요 패키지:** `ultralytics`, `opencv-python`, `matplotlib`

## 🚀 실행 방법 (Usage)

### 1. 모델 학습 실행 (Train)
미리 준비된 데이터셋(`data.yaml`)을 사용하여 YOLOv8 Nano 모델을 10 Epoch 동안 학습시킵니다.
```bash
python train_model.py