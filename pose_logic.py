# pose_logic.py

import math
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import cv2


# ── 유틸리티 함수 ──────────────────────────────────────────────

def calculate_angle(a, b, c):
    """
    세 점의 좌표로 b점(꼭짓점)의 각도를 계산합니다.
    a = 첫 번째 관절 (예: 어깨, 골반)
    b = 꼭짓점 관절  (예: 팔꿈치, 무릎)
    c = 세 번째 관절 (예: 손목, 발목)
    반환값: 0~180도 사이의 각도
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle  = math.degrees(math.acos(cosine))

    return angle


def euclidean_distance(point1, point2):
    """
    두 점 사이의 직선 거리 (L2 Norm)
    point1, point2: (x, y) 튜플
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


# ── 메인 분석 함수 ─────────────────────────────────────────────

def analyze_video(video_path, model_path="best.pt", release_threshold=80, use_left=False):
    """
    농구 슈팅 영상을 분석해서 릴리즈 포인트의 팔꿈치·무릎 각도를 반환합니다.

    Parameters
    ----------
    video_path         : 분석할 영상 파일 경로
    model_path         : YOLO 가중치 파일 경로 (기본: best.pt)
    release_threshold  : 손목-공 거리 임계값(px). 이 값 이상 벌어지면 릴리즈로 판정 (기본: 80)
    use_left           : True면 왼손 기준으로 관절 추출 (기본: False = 오른손)

    Returns
    -------
    dict: detected, elbow_angle, knee_angle, elbow_feedback, knee_feedback, frame
    """

    # ── 모델 로드 ──────────────────────────────────────────────
    yolo_model = YOLO(model_path)

    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # ── 오른손 / 왼손 랜드마크 선택 ───────────────────────────
    if use_left:
        shoulder_lm = mp_pose.PoseLandmark.LEFT_SHOULDER
        elbow_lm    = mp_pose.PoseLandmark.LEFT_ELBOW
        wrist_lm    = mp_pose.PoseLandmark.LEFT_WRIST
        hip_lm      = mp_pose.PoseLandmark.LEFT_HIP
        knee_lm     = mp_pose.PoseLandmark.LEFT_KNEE
        ankle_lm    = mp_pose.PoseLandmark.LEFT_ANKLE
    else:
        shoulder_lm = mp_pose.PoseLandmark.RIGHT_SHOULDER
        elbow_lm    = mp_pose.PoseLandmark.RIGHT_ELBOW
        wrist_lm    = mp_pose.PoseLandmark.RIGHT_WRIST
        hip_lm      = mp_pose.PoseLandmark.RIGHT_HIP
        knee_lm     = mp_pose.PoseLandmark.RIGHT_KNEE
        ankle_lm    = mp_pose.PoseLandmark.RIGHT_ANKLE

    # ── 영상 열기 ──────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상 파일을 열 수 없습니다: {video_path}")

    # ── 결과 초기화 ────────────────────────────────────────────
    release_detected = False
    result_data = {
        "detected":       False,
        "elbow_angle":    None,
        "knee_angle":     None,
        "elbow_feedback": None,
        "knee_feedback":  None,
        "frame":          None,
        "angle":          None,   # 하위 호환
        "feedback":       None,   # 하위 호환
    }
    prev_distance = None

    # ── 프레임 루프 ────────────────────────────────────────────
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # ── 1) YOLO로 농구공 탐지 ─────────────────────────────
        # 클래스 ID 4 = 농구공 (best.pt 학습 결과 기준)
        ball_center  = None
        yolo_results = yolo_model(frame, verbose=False)
        for box in yolo_results[0].boxes:
            cls = int(box.cls[0])
            if cls == 4:                          # ← 농구공 클래스 ID
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                ball_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.circle(frame, ball_center, 5, (0, 165, 255), -1)
                break

        # ── 2) MediaPipe로 관절 좌표 추출 ────────────────────
        frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)

        wrist_pos   = None
        shoulder_px = elbow_px = wrist_px = None
        hip_px      = knee_px  = ankle_px = None

        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark

            shoulder_px = (int(lm[shoulder_lm].x * w), int(lm[shoulder_lm].y * h))
            elbow_px    = (int(lm[elbow_lm].x    * w), int(lm[elbow_lm].y    * h))
            wrist_px    = (int(lm[wrist_lm].x    * w), int(lm[wrist_lm].y    * h))
            wrist_pos   = wrist_px

            hip_px   = (int(lm[hip_lm].x   * w), int(lm[hip_lm].y   * h))
            knee_px  = (int(lm[knee_lm].x  * w), int(lm[knee_lm].y  * h))
            ankle_px = (int(lm[ankle_lm].x * w), int(lm[ankle_lm].y * h))

            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # ── 3) 융합 로직 — 릴리즈 포인트 감지 ───────────────
        if ball_center and wrist_pos:
            dist = euclidean_distance(wrist_pos, ball_center)

            # 이전 프레임보다 거리가 임계값 이상 벌어진 순간 = 릴리즈!
            if (not release_detected
                    and prev_distance is not None
                    and prev_distance < release_threshold
                    and dist >= release_threshold):

                release_detected = True

                elbow_angle = calculate_angle(shoulder_px, elbow_px, wrist_px)
                knee_angle  = calculate_angle(hip_px,      knee_px,  ankle_px)

                elbow_feedback = "Perfect!"       if 85 <= elbow_angle <= 105 else "Adjust Elbow"
                knee_feedback  = "Good Leg!"      if knee_angle >= 150        else "Bend Knees More"
                elbow_color    = (0, 255, 0)      if elbow_feedback == "Perfect!" else (0, 0, 255)
                knee_color     = (0, 255, 0)      if knee_feedback  == "Good Leg!" else (0, 0, 255)

                cv2.putText(frame, f"Elbow: {elbow_angle:.1f}",
                            (30, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1.2, elbow_color, 2)
                cv2.putText(frame, elbow_feedback,
                            (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.4, elbow_color, 3)
                cv2.putText(frame, f"Knee:  {knee_angle:.1f}",
                            (30, 155), cv2.FONT_HERSHEY_SIMPLEX, 1.2, knee_color,  2)
                cv2.putText(frame, knee_feedback,
                            (30, 205), cv2.FONT_HERSHEY_SIMPLEX, 1.4, knee_color,  3)

                release_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)

                result_data.update({
                    "detected":       True,
                    "elbow_angle":    elbow_angle,
                    "knee_angle":     knee_angle,
                    "elbow_feedback": elbow_feedback,
                    "knee_feedback":  knee_feedback,
                    "frame":          release_frame,
                    "angle":          elbow_angle,
                    "feedback":       elbow_feedback,
                })

            prev_distance = dist

        # 로컬 실행 시 실시간 미리보기
        # Streamlit 연동 시 아래 두 줄을 주석 처리하세요
        cv2.imshow("Basketball Coach AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    return result_data


# ── 로컬 직접 실행 테스트 ──────────────────────────────────────
if __name__ == "__main__":
    result = analyze_video(
        video_path="my_shot.mp4",
        release_threshold=80,
        use_left=False          # 왼손잡이면 True로 변경
    )

    if result["detected"]:
        print("릴리즈 포인트 감지!")
        print(f"  팔꿈치 각도 : {result['elbow_angle']:.1f}°  →  {result['elbow_feedback']}")
        print(f"  무릎 각도   : {result['knee_angle']:.1f}°  →  {result['knee_feedback']}")
    else:
        print("릴리즈 포인트를 감지하지 못했습니다.")
        print("  → release_threshold를 60으로 낮춰보세요")
        print("  → 왼손잡이라면 use_left=True로 변경하세요")
