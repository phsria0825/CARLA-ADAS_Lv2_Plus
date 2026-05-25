"""
SFOT (Sensor Fusion and Object Tracking) 의 전역 파라미터.

상태 차원과 모델별 노이즈 prior 는 모션 모델 자체가 소유하므로 (sfot_motion.py),
본 파일에는 모델에 비의존적인 트랙 lifecycle / 측정 노이즈 / association
gating 임계값만 둔다.
"""
import config


# ---- 모션 모델 선택자 ----
# SFOT EKF 가 사용하는 교체 가능한 예측 모델.
#   "CV"   : 6 상태 등속도 모델. 단순·빠르나 곡선에서 드리프트.
#   "CTRV" : 7 상태 평면 CTRV + 수직 CV. 차량 MOT 의 표준 선택지로, 곡선에서
#            yaw arc 위에 trajectory 를 유지해 직선 외삽 오차가 적다.
SFOT_MOTION_MODEL = getattr(config, "SFOT_MOTION_MODEL", "CTRV")


# 시간 간격 (메인 루프 tick step)
DT = config.FIXED_DELTA


# ---- 트랙 lifecycle (모델 독립) ----
CONFIRMED_SCORE = 0.7
DELETE_SCORE = 0.4
SCORE_WINDOW = 8
MAX_P_POS = 25.0
SHAPE_BLEND_WEIGHT = 0.1

# Association — 측정 차원별 chi² 99.5 % gate 임계.
GATING_THRESHOLDS = {
    2: 10.596634733096073,   # chi2.ppf(0.995, df=2)   (카메라 픽셀)
    3: 12.838156466598647,   # chi2.ppf(0.995, df=3)   (LiDAR x,y,z)
}

# 센서별 측정 잡음 (BEV 스타일 검출에 맞춰 튜닝, 모델 독립).
SIGMA_LIDAR_X = 0.5
SIGMA_LIDAR_Y = 0.3
SIGMA_LIDAR_Z = 0.3
SIGMA_CAM_PIXEL_U = 10.0
SIGMA_CAM_PIXEL_V = 10.0
