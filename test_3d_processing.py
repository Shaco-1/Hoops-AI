import numpy as np
import pytest
import cv2
from depth_converter import generate_depth_map, generate_point_cloud

def test_generate_depth_map():
    """generate_depth_map 함수가 정상 작동하는지 검증하는 테스트"""
    # 1. 가짜 테스트용 이미지 생성 (100x100 크기의 검정색 빈 이미지)
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # 2. 테스트할 함수 실행
    depth_map = generate_depth_map(test_image)
    
    # 3. 결과 검증 (assert: 뒤의 조건이 참이 아니면 에러를 발생시킴)
    assert depth_map.shape == test_image.shape, "출력 크기가 입력 크기와 다릅니다."
    assert isinstance(depth_map, np.ndarray), "출력 데이터 타입이 ndarray가 아닙니다."

def test_empty_image_handling():
    """빈 이미지가 들어왔을 때 에러(ValueError)를 잘 뱉어내는지 검증"""
    with pytest.raises(ValueError, match="입력된 이미지가 없습니다."):
        generate_depth_map(None)