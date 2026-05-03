from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 가장 가벼운 YOLOv8 Nano 모델 로드
    model = YOLO("yolov8n.pt") 
    
    # 2. 모델 학습 시작 (data.yaml 경로를 본인 환경에 맞게 수정하세요)
    # 멘토님 팁 반영: 데이터 증강(augment=True)을 켜면 성능이 올라갑니다!
    print("🚀 학습을 시작합니다. (컴퓨터 사양에 따라 시간이 걸릴 수 있습니다)")
    model.train(data="dataset/data.yaml", epochs=10, imgsz=640, augment=True, fraction=0.33)