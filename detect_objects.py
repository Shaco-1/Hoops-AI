import cv2
from ultralytics import YOLO

# 1. 방금 학습을 마친 '나만의 농구 탐지 모델' 로드 (경로에 train-4를 꼭 확인하세요!)
model = YOLO("runs/detect/train-4/weights/best.pt")

# 2. 테스트해볼 농구 사진 불러오기 (본인이 저장한 파일명으로 변경 필수)
image_path = "test_basketball.jpg" 
image = cv2.imread(image_path)

if image is None:
    print("에러: 이미지를 찾을 수 없습니다. 파일명과 경로를 다시 확인해주세요.")
else:
    # 3. AI에게 이미지 분석(객체 탐지) 시키기
    results = model(image, conf=0.1)
    
    # 4. 탐지된 객체마다 OpenCV로 네모 박스와 이름표 그리기
    for result in results:
        for box in result.boxes:
            # 좌표, 라벨(농구공/골대), 신뢰도(몇 % 확신하는지) 추출
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label_id = int(box.cls[0])
            label_name = result.names[label_id]
            confidence = box.conf[0].item()
            
            # 초록색 박스 그리기
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 라벨 텍스트 쓰기 (예: Basketball (0.85))
            text = f"{label_name} ({confidence:.2f})"
            cv2.putText(image, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 5. 결과 화면 출력!
    cv2.imshow("Basketball Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()