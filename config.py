"""
CARLA ADAS Lv2+ 데모의 전역 설정 모듈이다.

전 모듈이 ``import config``로 참조하므로 본 파일은 어떠한 프로젝트 내부 모듈도
import하지 않는다. numpy 등 외부 라이브러리만 사용한다.
"""
import numpy as np

# CARLA 연결
CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
CARLA_TIMEOUT = 30.0
MAP_NAME = 'Town04'
SYNC_MODE = True
FIXED_DELTA = 0.05  # 20 FPS (제어 주기 50 ms)

# 차량
VEHICLE_BP = 'vehicle.tesla.model3'
WHEELBASE = 2.875

# 조향 명령 포화
MAX_STEER = 1.0             # CARLA VehicleControl.steer는 [-1, 1] 범위이다.
MAX_STEER_ANGLE_RAD = 0.61  # Tesla Model 3 최대 휠 조향각(약 35°). rad 단위인
                            # LQR/FF 출력을 [-1, 1]로 정규화할 때 사용한다.

# 횡제어 — Ackermann 곡률 FF + 동역학 bicycle LQR FB (control/lateral_controller.py).
#
# 아래 차량 동역학 파라미터는 Ackermann FF(WHEELBASE)와 LQR 상태공간 행렬
# (mass / Iz / lf / lr / Cf / Cr) 모두에 입력으로 들어간다.
VEHICLE_MASS = 1800.0  # kg
VEHICLE_IZ = 3270.0    # kg·m²
VEHICLE_LF = 1.41      # m — 전륜축에서 CG까지의 거리
VEHICLE_LR = 1.465     # m — 후륜축에서 CG까지의 거리
VEHICLE_CF = 80000.0   # N/rad — 전륜 cornering stiffness
VEHICLE_CR = 85000.0   # N/rad — 후륜 cornering stiffness

# LQR Q/R은 정적 상수이나 K 자체는 정적이 아니다 — Ac가 ego 속도 v에 의존하므로
# K는 매 tick Q/R로부터 재계산된다 (control/lateral_controller.py의 _solve_K 참고).
# v에 대한 gain-scheduled lookup table도 가능하나, 4×4 DARE 풀이가 수십 µs라
# 20 Hz 제어 주기에선 불필요한 최적화이다.
LQR_Q = np.diag([2.5, 0.1, 5.5, 5.5])   # [e_y, e_y_dot, e_psi, e_psi_dot] 가중
LQR_R = np.array([[2.0]])                # 조향 effort cost

# 종방향 제어 (ACC / SCC PID 2단 루프)
# 상위 루프: 속도/간격 오차 → 목표 가속도
SPEED_KP = 0.75
SPEED_KI = 0.05
SPEED_KD = 0.05

GAP_KP = 0.65
GAP_KI = 0.02
GAP_KD = 0.3

# 하위 루프: 목표 가속도 → throttle/brake
ACCEL_KP = 1.5
ACCEL_KI = 0.5
ACCEL_KD = 0.0

MAX_ACCEL = 5.0
MAX_DECEL = -5.0
COMFORT_DECEL = -2.5
MAX_JERK = 5.0
CONTROL_SCALE = 30.0

# SCC / ACC 동작 파라미터
SET_SPEED = 60.0
TIME_GAP = 1.5
STANDSTILL_DIST = 5.0

LEAD_DETECT_DIST = 70.0
LEAD_LOST_DIST = 75.0
LEAD_LOST_COUNT = 8  # 20 Hz 기준 0.4 s 동안 미수신 시 lead 상실로 간주
GAP_MAX_THROTTLE = 0.50
FOLLOW_SPEED_TAPER_KMH = 10.0
FOLLOW_OVERSPEED_BRAKE_KMH = 5.0
FOLLOW_OVERSPEED_BRAKE_MAX = 0.20

# Stop-and-Go (저속 추종) 동작 파라미터
SNG_ENGAGE_SPEED_KMH = 15.0
SNG_EXIT_SPEED_KMH = 20.0
SNG_ENGAGE_DIST = 20.0
SNG_EXIT_DIST = 25.0
SNG_HOLD_ENTRY_SPEED_KMH = 3.0
SNG_HOLD_SPEED_KMH = 1.0
SNG_HOLD_MARGIN_M = 0.75
SNG_HOLD_RELEASE_GAP_M = 1.5
SNG_HOLD_RELEASE_RATE_MPS = 0.8
SNG_HOLD_BRAKE = 0.35
SNG_MAX_THROTTLE = 0.55  # cut-in 회복 시 throttle 한도
SNG_MAX_BRAKE = 0.60

# AEB 동작 파라미터
AEB_TTC_ENTER = 1.2
AEB_TTC_EXIT = 1.8
AEB_MIN_RANGE = 6.0
AEB_RELEASE_RANGE = 8.0
AEB_MIN_CLOSING_SPEED = 1.0
AEB_BRAKE_MIN = 0.65
AEB_BRAKE_MAX = 1.0
AEB_FULL_BRAKE_TTC = 0.6
AEB_HOLD_SPEED_KMH = 0.5
AEB_HOLD_BRAKE = 0.80

# Traffic Manager
TM_PORT = 8000

# NPC cut-out 시나리오 파라미터
CUT_OUT_PROGRESS_PCT = 60.0
CUT_OUT_DIRECTION_RIGHT = True

# 인지 / 경로 회랑
PERCEPTION_RANGE = 100.0
TRACK_CORRIDOR_WIDTH = 3.5

# 센서 융합 유효 범위 (LiDAR + Camera 중첩 영역)
FUSION_RANGE = 100.0

# LiDAR 설정 (지붕 장착 Velodyne 64ch, 360° 회전)
LIDAR_RANGE = 100.0
LIDAR_CHANNELS = 64
LIDAR_POINTS_PER_SECOND = 2300000
LIDAR_ROTATION_FREQUENCY = 10.0
LIDAR_UPPER_FOV = 2.4
LIDAR_LOWER_FOV = -24.8
LIDAR_HORIZONTAL_FOV = 360.0
LIDAR_POS_X = 0.0
LIDAR_POS_Y = 0.0
LIDAR_POS_Z = 2.25
LIDAR_MIN_RANGE = 1.0
LIDAR_GT_FORWARD_ONLY = False

# 단일 카메라 설정
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1280
CAMERA_FOV_DEG = 90.0
CAMERA_POS_X = 1.8
CAMERA_POS_Y = 0.0
CAMERA_POS_Z = 1.5
CAMERA_PITCH_DEG = -2.0
CAMERA_MAX_RANGE = 75.0

# Tesla-style 8-카메라 레이아웃
CAMERA_LAYOUT = {
    "front_bumper": {
        "x": 2.45, "y": 0.00, "z": 0.65,
        "roll": 0.0, "pitch": -2.0, "yaw": 0.0,
        "fov": 100.0, "width": 1280, "height": 720,
    },
    "front_main": {
        "x": 1.80, "y": 0.00, "z": 1.55,
        "roll": 0.0, "pitch": -2.0, "yaw": 0.0,
        "fov": 60.0, "width": 1280, "height": 720,
    },
    "front_wide": {
        "x": 1.80, "y": 0.00, "z": 1.55,
        "roll": 0.0, "pitch": -2.0, "yaw": 0.0,
        "fov": 120.0, "width": 1280, "height": 720,
    },
    "left_pillar": {
        "x": 0.20, "y": -0.85, "z": 1.35,
        "roll": 0.0, "pitch": -2.0, "yaw": -75.0,
        "fov": 100.0, "width": 1280, "height": 720,
    },
    "right_pillar": {
        "x": 0.20, "y": 0.85, "z": 1.35,
        "roll": 0.0, "pitch": -2.0, "yaw": 75.0,
        "fov": 100.0, "width": 1280, "height": 720,
    },
    "left_fender": {
        "x": 1.20, "y": -0.95, "z": 0.95,
        "roll": 0.0, "pitch": -2.0, "yaw": -120.0,
        "fov": 100.0, "width": 1280, "height": 720,
    },
    "right_fender": {
        "x": 1.20, "y": 0.95, "z": 0.95,
        "roll": 0.0, "pitch": -2.0, "yaw": 120.0,
        "fov": 100.0, "width": 1280, "height": 720,
    },
    "rear": {
        "x": -2.10, "y": 0.00, "z": 1.10,
        "roll": 0.0, "pitch": -5.0, "yaw": 180.0,
        "fov": 120.0, "width": 1280, "height": 720,
    },
}

# ============================================================
# Lv2+ 차선변경 의사결정 파라미터
# ============================================================
LC_ENABLE = True
LC_ALLOW_DISCRETIONARY = True
LC_ALLOW_MANDATORY = True

# ODD / 도로 제약
# 비고: LC_MIN_SPEED_KMH=25 km/h는 OVERTAKE 시나리오에서 ego가 약 35 km/h
# 느린 lead에 추종 중일 때도 discretionary LC가 발동될 수 있도록 한 값이다.
# 양산 highway ODD에서는 통상 45 km/h 이상으로 유지한다.
LC_MIN_SPEED_KMH = 25.0
LC_MAX_SPEED_KMH = 90.0
LC_DISABLE_IN_JUNCTION = True
LC_MIN_DIST_TO_JUNCTION_M = 80.0
LC_MIN_DIST_AFTER_JUNCTION_M = 30.0
LC_REQUIRE_LANE_CHANGE_PERMISSION = True

# ============================================================
# MOBIL / IDM 차선변경 의사결정 파라미터
# ============================================================
# behavior_planner 는 IDM 기반 가속도와 MOBIL safety / incentive 기준만으로
# LC 를 판단한다. 두 모델의 파라미터를 한 곳에 모아 둔다.

# IDM 파라미터 (승용차 기본값)
IDM_T_S = 1.5            # 안전 time headway [s]
IDM_S0_M = 2.0           # 정지 시 최소 간격 [m]
IDM_A_MPS2 = 1.4         # 최대 가속도 [m/s²]
IDM_B_MPS2 = 2.0         # 편안한 감속도 [m/s²]
IDM_DELTA = 4.0          # 가속도 지수

# MOBIL 파라미터
MOBIL_P = 0.2            # politeness factor (0~1)
MOBIL_B_SAFE_MPS2 = 4.0  # new follower에 강요되는 최대 허용 감속도
MOBIL_A_TH_MPS2 = 0.2    # LC 전환 임계치 (히스테리시스)
MOBIL_A_BIAS_MPS2 = 0.2  # 우측 keep-right 비대칭 bias (US-style은 0)

# 결정층 debounce: 추천 결과가 이 시간 동안 유지되어야 commit한다.
# MOBIL의 a_th 히스테리시스와 별개로, perception jitter를 1차 차단한다.
LC_DECISION_HYSTERESIS_S = 2.0

# ============================================================
# Motion-prediction 기반 MOBIL 평가
# ============================================================
# ConstantVelocityPredictor 가 산출한 1차 예측을 MOBIL 평가에 반영해, 인접
# 차로 leader/follower 의 "현재" 와 "t_pred_s 후" 중 더 보수적인(가까운)
# 위치를 사용한다. 1.0 s 는 CV 예측 horizon(1.5 s) 의 중앙 부근으로, 직선
# 가정이 유효한 범위에서 0.5~1 s 사이의 cut-in / cut-out 위협을 한 박자
# 먼저 잡기 위함이다.
LC_PREDICTION_LOOKAHEAD_S = 1.0

# ============================================================
# PREPARE -> EXECUTE 전이 직전 후방 안전 재검증
# ============================================================
# PREPARE 1.0 s 동안 후방 차량이 급접근하는 케이스를 잡기 위해, EXECUTE 진입
# 직전에 한 번 더 후방 gap / TTC 게이트를 통과시킨다. 미달 시 REQUESTED 로
# 회귀해 hysteresis 부터 다시 쌓는다.
LC_PREEXEC_HORIZON_S = 1.0   # 재검증에 사용할 미래 시점 [s]
LC_PREEXEC_TTC_MIN_S = 3.0   # 후방 TTC 하한 [s]
# 후방 gap 의 절대 하한은 ``max(LC_PREEXEC_GAP_ABS_MIN_M, time_gap_factor *
# v_follower_future)`` 로 산출한다. ego 가 정지에 가까울 때도 일정 거리 이상은
# 유지하기 위함이다.
LC_PREEXEC_GAP_ABS_MIN_M = 8.0
LC_PREEXEC_GAP_TIME_FACTOR_S = 1.0

# 안정 추종(stable car-follow) 트리거
LC_STABLE_FOLLOW_TIME_S = 5.0
LC_STABLE_SPEED_STD_MAX_MPS = 0.35
LC_STABLE_GAP_STD_MAX_M = 1.5
LC_REL_SPEED_ABS_MAX_MPS = 0.6
LC_EGO_UNDER_SET_SPEED_MIN_MPS = 1.5
LC_LEAD_RANGE_MIN_M = 18.0
LC_LEAD_RANGE_MAX_M = 60.0

# LC 쿨다운 및 히스테리시스
LC_COOLDOWN_AFTER_SUCCESS_S = 8.0
LC_COOLDOWN_AFTER_ABORT_S = 5.0

# Target 차선 안전 간격
LC_TARGET_FRONT_MIN_GAP_M = 30.0
LC_TARGET_REAR_MIN_GAP_M = 35.0
LC_TARGET_FRONT_TTC_MIN_S = 4.0
LC_TARGET_REAR_TTC_MIN_S = 5.0
LC_TARGET_SIDE_GAP_MIN_M = 8.0

# 현재 차선 안전 조건
LC_CURRENT_LEAD_HARD_GAP_M = 10.0
LC_CURRENT_LEAD_TTC_MIN_S = 1.8

# 기대 이득
LC_MIN_SPEED_GAIN_MPS = 1.5
LC_TARGET_LANE_SPEED_SAMPLE_DIST_M = 70.0

# 경로 생성
LC_PREPARE_TIME_S = 1.0
LC_EXECUTE_TIME_CANDIDATES_S = [4.0, 4.5, 5.0]
LC_MIN_TOTAL_TIME_S = 4.5
LC_MAX_TOTAL_TIME_S = 6.5
LC_MIN_LANE_CHANGE_LENGTH_M = 45.0
LC_MAX_LANE_CHANGE_LENGTH_M = 85.0

# 승차감 제약
LC_MAX_LATERAL_ACCEL_MPS2 = 1.2
LC_MAX_LATERAL_JERK_MPS3 = 0.8
LC_MAX_LONG_ACCEL_MPS2 = 1.5
LC_COMFORT_DECEL_MPS2 = -2.0

# Abort / cancel
LC_CANCEL_ALLOWED_IN_PREPARE = True
LC_ABORT_ALLOWED_IN_EXECUTE = True
LC_ABORT_MAX_LATERAL_OFFSET_M = 1.2
LC_ABORT_MIN_RETURN_GAP_M = 20.0

# EXECUTE 진행 중 측·후방 위협 대응 (ABORT / COMMIT 분기).
# behavior_planner._update_execute 가 매 tick target 차선의 후방 차량을
# 평가하여 LC 진행도에 따라 ABORT(원 차선 복귀) 또는 COMMIT(가속 통과)을
# 선택한다. 본 임계값들은 EXECUTE 전용이며, LC 시작 전 게이팅에는 위의
# LC_TARGET_REAR_* gap/TTC 값이 계속 사용된다.
LC_EXECUTE_REAR_HORIZON_M = 60.0           # 후방 탐색 거리 상한 [m]
LC_EXECUTE_REAR_TTC_ABORT_S = 4.0          # EXECUTE 중 rear TTC 임계 [s]
LC_EXECUTE_REAR_GAP_HARD_M = 15.0          # 무조건 ABORT를 강제하는 hard gap [m]
LC_EXECUTE_ABORT_PROGRESS_THRESHOLD = 0.5  # 진행도가 이 미만이면 ABORT, 이상이면 COMMIT

# Turn signal 시뮬레이션
LC_TURN_SIGNAL_PRE_TIME_S = 1.0
LC_TURN_SIGNAL_POST_TIME_S = 0.5

# Local Planning 시야 거리
LOCAL_PATH_HORIZON_M = 90.0
LOCAL_OBJECT_HORIZON_FRONT_M = 100.0
LOCAL_OBJECT_HORIZON_REAR_M = 60.0
LOCAL_PLANNING_SAMPLE_STEP_M = 1.0
LOCAL_PLANNING_TIME_HORIZON_S = 6.0

# Local Plan cost 가중치
LC_COST_COLLISION = 1000.0
LC_COST_TTC = 100.0
LC_COST_LAT_JERK = 5.0
LC_COST_LAT_ACCEL = 3.0
LC_COST_ROUTE = 10.0
LC_COST_SPEED_LOSS = 2.0

# ============================================================
# 로깅
# ============================================================
# False(기본): 1초당 한 줄의 압축된 상태 라인(속도, SCC 상태, LC 상태,
# lead range, route 진행도)과 핵심 LC 이벤트(REQUESTED / CANCELLED /
# COMPLETE)만 콘솔에 출력한다. True로 두면 BP 전체 진단 덤프와
# SCC/BP의 중간 상태 전이 로그가 모두 출력된다.
# main_adas_loop의 --verbose 플래그로 전환한다.
LOG_VERBOSE = False
