import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_depth_map(image):
    """2D 이미지를 받아 가상의 깊이 맵(Depth Map)을 생성하는 함수"""
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")
    
    # 1. 흑백 이미지로 변환 (밝기 값을 깊이 정보로 가정)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. 가짜 깊이 맵 적용 (가까운 곳과 먼 곳을 색상으로 표현하는 JET 컬러맵 사용)
    depth_map = cv2.applyColorMap(grayscale, cv2.COLORMAP_JET)
    return depth_map

def generate_point_cloud(image):
    """Depth Map을 기반으로 3D 포인트 클라우드 좌표를 생성하는 심화 함수"""
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    
    # X, Y 평면의 그리드(좌표) 생성
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    
    # 흑백 이미지의 픽셀 밝기 값을 Z축(깊이)으로 사용
    Z = gray.astype(np.float32)
    
    # X, Y, Z 좌표를 합쳐서 3D 공간상의 점(Point Cloud) 데이터 완성
    points_3d = np.dstack((X, Y, Z))
    return points_3d

if __name__ == "__main__":
    # 주의: 코드를 실행하기 전에 VS Code 현재 폴더에 테스트할 이미지가 있어야 합니다.
    # 본인이 사용하는 파일명으로 변경해주세요 (예: 'sample.jpg')
    img = cv2.imread('preprocessed_samples/tomato1_sample.jpg')
    
    if img is not None:
        # --- 1단계: 2D Depth Map(열화상 느낌) 화면 출력 ---
        depth = generate_depth_map(img)
        cv2.imshow('Original 2D', img)
        cv2.imshow('3D Depth Map', depth)
        
        print("안내: 띄워진 이미지 창을 선택하고 아무 키나 누르면 창이 닫히고 3D 화면으로 넘어갑니다.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # --- 2단계: 진짜 3D Point Cloud 화면 출력 ---
        print("3D Point Cloud를 생성 중입니다. 잠시만 기다려주세요...")
        points_3d = generate_point_cloud(img)
        
        # 데이터를 그리기 쉽게 X, Y, Z축으로 분리
        X = points_3d[:, :, 0]
        Y = points_3d[:, :, 1]
        Z = points_3d[:, :, 2]
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 컴퓨터가 느려지지 않게 점을 듬성듬성(10픽셀 간격) 찍습니다.
        ax.scatter(X[::10, ::10], Y[::10, ::10], Z[::10, ::10], 
                   c=Z[::10, ::10], cmap='jet', marker='.')
        
        ax.set_title('3D Point Cloud Visualization')
        plt.show() # 이 창이 뜨면 마우스로 이리저리 드래그해서 입체감을 확인해 보세요!
        
    else:
        print("에러: 이미지를 찾을 수 없습니다. 파일명과 경로를 다시 한번 확인해주세요.")