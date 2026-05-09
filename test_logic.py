import unittest
import numpy as np

# pose_logic.py 에서 각도 계산 함수만 가져오기
from pose_logic import calculate_angle

class TestPoseLogic(unittest.TestCase):
    def test_calculate_angle_90_degrees(self):
        # 직각(90도) 상황 시뮬레이션
        shoulder = [0, 100]
        elbow = [0, 0]
        wrist = [100, 0]
        angle = calculate_angle(shoulder, elbow, wrist)
        self.assertAlmostEqual(angle, 90.0)

    def test_calculate_angle_180_degrees(self):
        # 일직선(180도) 상황 시뮬레이션
        shoulder = [0, 100]
        elbow = [0, 0]
        wrist = [0, -100]
        angle = calculate_angle(shoulder, elbow, wrist)
        self.assertAlmostEqual(angle, 180.0)

if __name__ == '__main__':
    unittest.main()