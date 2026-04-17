import cv2
import numpy as np

# 1. 이미지 로드
# 주의: 코드를 실행하기 전에 VS Code의 현재 폴더 안에 'sample.jpg'라는 이미지가 있어야 합니다!
image = cv2.imread('preprocessed_samples/tomato1_sample.jpg') 

# 이미지가 정상적으로 로드되었는지 확인하는 안전장치 (현업 꿀팁)
if image is None:
    print("에러: 이미지를 찾을 수 없습니다. 파일명과 경로를 확인해주세요.")
    exit()

# 2. BGR에서 HSV 색상 공간으로 변환
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 3. 빨간색 범위 지정 (0~10 부근과 170~180 부근 두 개로 설정)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# 4. 마스크 생성 및 병합
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 + mask2 # 두 개의 빨간색 영역 마스크를 하나로 합침

# 5. 원본 이미지에서 빨간색 부분만 추출 (마스크 씌우기)
result = cv2.bitwise_and(image, image, mask=mask)

# 6. 결과 이미지 화면에 출력
cv2.imshow('Original', image)        # 원본 이미지 창
cv2.imshow('Red Filtered', result)   # 필터링된 이미지 창

# 아무 키나 누를 때까지 창을 닫지 않고 대기
cv2.waitKey(0) 
cv2.destroyAllWindows()