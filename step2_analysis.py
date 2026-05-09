import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# 1. 모델 초기화
print("모델을 로드하는 중입니다...")
yolo_model = YOLO('best.pt') # 1단계에서 학습한 농구공/골대 인식 모델
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 2. 각도 계산 함수 (수학적 로직)
def calculate_angle(a, b, c):
    a = np.array(a) # 첫 번째 점 (예: 어깨)
    b = np.array(b) # 중심 점 (예: 팔꿈치)
    c = np.array(c) # 끝 점 (예: 손목)
    
    # 아크탄젠트를 활용한 라디안 계산 후 각도(Degree)로 변환
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    # 각도가 180도를 넘어가면 보정
    if angle > 180.0:
        angle = 360 - angle
    return angle

# 3. 영상 불러오기
video_path = 'my_shot.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("영상을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

# 결과 저장을 위한 설정 (선택 사항)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('step2_result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print("영상 분석을 시작합니다...")

while cap.isOpened():
    ret, frame = cap.get()
    if not ret:
        break
        
    # YOLO 추론 (농구공 찾기)
    # stream=True를 사용하면 메모리 효율이 좋습니다.
    yolo_results = yolo_model(frame, stream=True, verbose=False)
    
    for r in yolo_results:
        boxes = r.boxes
        for box in boxes:
            # 바운딩 박스 좌표 추출
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = float(box.conf[0])
            
            # 농구공 표시 (클래스 0이 농구공이라고 가정)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Ball {conf:.2f}', (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # MediaPipe 추론 (관절 찾기)
    # MediaPipe는 RGB 이미지를 사용하므로 변환 필요
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(image_rgb)
    
    if pose_results.pose_landmarks:
        # 관절선 그리기
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # 랜드마크 좌표 추출
        landmarks = pose_results.pose_landmarks.landmark
        
        # 오른쪽 팔 각도 계산 (어깨: 12, 팔꿈치: 14, 손목: 16)
        # 이미지 크기에 맞게 정규화된 좌표를 픽셀 단위로 변환
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * width,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * height]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * width,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * height]
        
        # 오른쪽 무릎 각도 계산 (골반: 24, 무릎: 26, 발목: 28)
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * width,
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * height]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * width,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * height]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * width,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * height]
        
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        knee_angle = calculate_angle(hip, knee, ankle)
        
        # 화면에 텍스트로 각도 표시
        cv2.putText(frame, f'Elbow Angle: {int(elbow_angle)}', 
                    tuple(np.multiply(elbow, 1).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    
        cv2.putText(frame, f'Knee Angle: {int(knee_angle)}', 
                    tuple(np.multiply(knee, 1).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # 결과 영상 보여주기
    cv2.imshow('Shooting Analysis', frame)
    out.write(frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("분석이 완료되었습니다. 'step2_result.mp4'를 확인해보세요.")