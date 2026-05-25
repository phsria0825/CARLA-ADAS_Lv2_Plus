"""
ADAS Lv2+ 스택 전체에서 공유되는 데이터 타입 정의 모듈.

본 모듈에 정의된 타입들은 perception / localization / planning / control 계층
사이의 인터페이스 계약 역할을 한다. 한 모듈에 모아 두는 이유는 두 가지이다.
  - 순환 import 를 방지한다.
  - 컨트롤러가 자신이 요구하는 입력을 CARLA 에 의존하지 않은 채 선언할 수 있다.

기존 호출자 ``EgoState(timestamp=0.0, x=0.0, ...)`` 같은 키워드 인자 생성 패턴은
그대로 동작한다.
"""
from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# Ego 상태
# ---------------------------------------------------------------------------

class EgoState:
    """자차의 단일 tick 상태 스냅샷.

    위치 / yaw / 속도 (world & body frame) / 가속도 / 유효성을 한 번에 담아
    EgoStateProvider 가 매 tick 생성한다. 다른 모듈(차선 모델, perception,
    planner, controller)이 모두 이 객체만 보고 자차 정보를 얻는다.
    """

    def __init__(
        self,
        timestamp,
        x,                  # 월드 x [m]
        y,                  # 월드 y [m]
        z,                  # 월드 z [m]
        yaw_rad,            # 월드 yaw [rad], CCW 양수
        speed_mps,          # 전진 속도 크기 [m/s]
        speed_kmh,
        vx_world,
        vy_world,
        # 차체(body) 좌표계 속도. vx_body 는 부호 있는 전진속도(후진 시 음수),
        # vy_body 는 CARLA 의 left-handed frame 에서 우측이 양수이다. 본 스택
        # 전반에 사용되는 RIGHT-positive lateral-error 컨벤션과 일치한다.
        vx_body=0.0,
        vy_body=0.0,
        yaw_rate_rad_s=0.0,
        accel_mps2=0.0,     # 부호 있는 종방향 가속도 [m/s²]
        is_valid=True,
    ):
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.yaw_rad = yaw_rad
        self.speed_mps = speed_mps
        self.speed_kmh = speed_kmh
        self.vx_world = vx_world
        self.vy_world = vy_world
        self.vx_body = vx_body
        self.vy_body = vy_body
        self.yaw_rate_rad_s = yaw_rate_rad_s
        self.accel_mps2 = accel_mps2
        self.is_valid = is_valid


# ---------------------------------------------------------------------------
# Lead 차량 / SCC 입력
# ---------------------------------------------------------------------------

class LeadInfo:
    """단일 차로의 lead 차량 정보.

    LeadVehicleSelector 가 산출하며 SCC / GAP_CTRL / AEB 가 소비한다. lead 가
    없으면 ``detected = False`` 이고 나머지 필드는 디폴트(inf, 0) 로 채워진다.
    """

    def __init__(
        self,
        detected=False,
        track_id=None,                       # int 또는 None
        range=float('inf'),                  # lead까지 종방향 gap [m]
        range_rate=0.0,                      # d(range)/dt, 접근 중이면 음수
        lateral_offset=0.0,                  # 현재 차선 중심선 대비 부호 있는 횡거리 [m]
        lane_id=None,                        # int 또는 None
        ttc=float('inf'),
        confidence=0.0,
        lead_speed_mps=0.0,
    ):
        self.detected = detected
        self.track_id = track_id
        self.range = range
        self.range_rate = range_rate
        self.lateral_offset = lateral_offset
        self.lane_id = lane_id
        self.ttc = ttc
        self.confidence = confidence
        self.lead_speed_mps = lead_speed_mps


class LeadInfoSet:
    """현재 차로와 좌·우 인접 차로의 lead 정보를 묶은 컨테이너.

    SCC / 종방향 제어는 ``current`` 만 소비해 기존 인터페이스를 유지하며,
    MOBIL / BehaviorPlanner 는 ``left`` / ``right`` 도 참조해 인접 차로 lead
    진입을 의사결정에 반영한다. ``left`` / ``right`` 가 None 인 경우는 해당
    인접 차로 자체가 invalid (LaneModel.left/right_lane_available == False)
    한 경우이며, 차로는 있으나 lead 가 없을 뿐이면 ``detected = False`` 인
    LeadInfo 가 들어간다.
    """

    def __init__(self, current=None, left=None, right=None):
        # current 가 None 으로 호출되면 기본 LeadInfo() 가 자체 생성된다.
        self.current = current if current is not None else LeadInfo()
        self.left = left
        self.right = right


# ---------------------------------------------------------------------------
# 차선 모델
# ---------------------------------------------------------------------------

class LaneBoundary:
    """단일 차선 경계의 점열과 마킹 속성.

    LaneModel.left_boundary / right_boundary 에 들어가며, BEV 시각화와 LC
    가능 여부 판정에 사용된다.
    """

    def __init__(
        self,
        points_xy=None,            # [(x, y), ...] 월드 좌표
        marking_type="Unknown",
        marking_color="Unknown",
        lane_change_allowed=False,
    ):
        self.points_xy = points_xy if points_xy is not None else []
        self.marking_type = marking_type
        self.marking_color = marking_color
        self.lane_change_allowed = lane_change_allowed


class LaneModel:
    """운전자 좌표계(driver frame) 기준 차선 모델.

    본 구조체의 모든 "left/right" 필드는 운전자 시점을 따른다. 즉 "left" 는
    운전자의 물리적 왼쪽이며, CARLA OpenDRIVE 의 reference line 방향이 아니다.
    CARLA 의 ``get_left_lane()`` 은 ego 의 ``lane_id > 0`` (도로 reference 에
    역방향으로 주행) 일 때 부호가 뒤바뀌므로, lane provider 내부에서 운전자
    좌표계로 통일해 소비자에게 노출한다.

    centerline_kappa 는 centerline 과 동일 길이의 샘플별 부호 있는 곡률
    [1/m] (Menger curvature, 외적 부호 사용). 양 끝 두 점은 0 이다.
    """

    def __init__(
        self,
        is_valid=False,
        current_lane_id=0,
        road_id=0,
        section_id=0,
        lane_width=3.5,                  # 전방 10 m 구간의 median-smoothed 값
        is_junction=False,
        distance_to_junction=float('inf'),
        centerline=None,                  # [(x, y), ...] 월드 좌표
        centerline_ego=None,              # [(x, y), ...] ego 좌표
        centerline_kappa=None,
        left_boundary=None,               # LaneBoundary 또는 None
        right_boundary=None,
        left_centerline=None,             # [(x, y), ...] 월드 좌표
        right_centerline=None,
        left_lane_available=False,
        right_lane_available=False,
        left_lane_change_allowed=False,
        right_lane_change_allowed=False,
    ):
        self.is_valid = is_valid
        self.current_lane_id = current_lane_id
        self.road_id = road_id
        self.section_id = section_id
        self.lane_width = lane_width
        self.is_junction = is_junction
        self.distance_to_junction = distance_to_junction
        self.centerline = centerline if centerline is not None else []
        self.centerline_ego = centerline_ego if centerline_ego is not None else []
        self.centerline_kappa = centerline_kappa if centerline_kappa is not None else []
        self.left_boundary = left_boundary if left_boundary is not None else LaneBoundary()
        self.right_boundary = right_boundary if right_boundary is not None else LaneBoundary()
        self.left_centerline = left_centerline if left_centerline is not None else []
        self.right_centerline = right_centerline if right_centerline is not None else []
        self.left_lane_available = left_lane_available
        self.right_lane_available = right_lane_available
        self.left_lane_change_allowed = left_lane_change_allowed
        self.right_lane_change_allowed = right_lane_change_allowed


class ObjectLaneRelation:
    """단일 fused 객체가 어느 차로에 속하는지 분류한 결과.

    LaneProvider.classify_object_lane() 의 출력으로, FusedObject.lane_relation
    에 부착되어 lead 선택과 인접 차로 평가에 사용된다.
    """

    def __init__(
        self,
        track_id,
        lane_id,
        longitudinal_s,
        lateral_d,
        relation,           # "CURRENT_LANE" | "LEFT_LANE" | "RIGHT_LANE" | "ADJACENT_UNKNOWN" | "OUT_OF_ROAD"
    ):
        self.track_id = track_id
        self.lane_id = lane_id
        self.longitudinal_s = longitudinal_s
        self.lateral_d = lateral_d
        self.relation = relation


# ---------------------------------------------------------------------------
# 센서별 측정값
# ---------------------------------------------------------------------------

class ObjectMeasurement:
    """단일 센서, 단일 tick 의 detection 한 건. ego frame 으로 표현한다.

    각 pseudo sensor (카메라 마운트, LiDAR) 가 매 tick 본 객체의 리스트를
    출력하며, EKF tracker 가 이를 통합 입력으로 받아 FusedObject 를 생성한다.
    """

    def __init__(
        self,
        sensor_id,
        timestamp,
        object_type,       # "vehicle" | "walker" | "static"
        x,                 # ego 전방 [m]
        y,                 # ego 우측 [m]
        width,
        length,
        yaw,               # ego 전방축 기준 객체 heading [rad] (또는 None)
        range,             # |(x, y)|
        bearing,           # atan2(y, x) [rad]
        confidence,        # 0~1
        # ego (px, py) 좌표계의 2×2 측정 공분산.
        # 카메라: bearing 정확도 高, range 정확도 低.
        # LiDAR : range 정확도 高, bearing 정확도 低.
        covariance=((1.0, 0.0), (0.0, 1.0)),
        # 디버깅 / 데이터 association 비교용 ground-truth track ID.
        # EKF 자체는 이 값을 사용하지 않는다.
        truth_id=None,
    ):
        self.sensor_id = sensor_id
        self.timestamp = timestamp
        self.object_type = object_type
        self.x = x
        self.y = y
        self.width = width
        self.length = length
        self.yaw = yaw
        self.range = range
        self.bearing = bearing
        self.confidence = confidence
        self.covariance = covariance
        self.truth_id = truth_id


# ---------------------------------------------------------------------------
# 융합 객체 (EKF confirmed track 의 외부 표현)
# ---------------------------------------------------------------------------

class FusedObject:
    """SFOT EKF 가 confirmed 상태로 발행하는 단일 트랙의 외부 표현.

    BehaviorPlanner / LeadVehicleSelector / BEV 가 모두 이 객체를 소비한다.
    ego frame 과 world frame 위치를 둘 다 들고 있어 소비자 측의 좌표 변환을
    줄인다.

    본 구조체에는 의도적으로 가속도 필드를 두지 않는다. 하류 소비자인 MOBIL
    (traffic_models.mobil_evaluate) 이 모든 차량 가속도를 IDM 식을 통해
    (speed, gap) 으로부터 추정하기 때문이다. 실차 ADAS 인지 시스템은 별도의
    레이더/LiDAR 추적과 수치미분 없이 주변 차량의 가속도를 신뢰성 있게 측정할
    수 없으므로, CARLA 의 ground-truth actor.get_acceleration() 을 가져와 쓰면
    편리하더라도 본 설계는 이를 거부한다. 가속도 필드를 추가하면 MOBIL 의
    incentive term 비교 일관성이 깨진다.
    """

    def __init__(
        self,
        track_id,
        object_type,        # "vehicle" | "walker" | "static" | "unknown"
        x_ego,              # ego 전방 [m]
        y_ego,              # ego 우측 [m]
        vx_ego,             # ego 좌표계 [m/s]
        vy_ego,
        yaw_ego,            # ego 전방축 기준 객체 heading [rad]
        length,
        width,
        confidence,
        age_s,
        x_world=0.0,
        y_world=0.0,
        yaw_world=0.0,
        speed_mps=0.0,
        lane_relation=None,
    ):
        self.track_id = track_id
        self.object_type = object_type
        self.x_ego = x_ego
        self.y_ego = y_ego
        self.vx_ego = vx_ego
        self.vy_ego = vy_ego
        self.yaw_ego = yaw_ego
        self.length = length
        self.width = width
        self.confidence = confidence
        self.age_s = age_s
        self.x_world = x_world
        self.y_world = y_world
        self.yaw_world = yaw_world
        self.speed_mps = speed_mps
        self.lane_relation = lane_relation


# ---------------------------------------------------------------------------
# Motion Prediction (planning/predictors 출력)
# ---------------------------------------------------------------------------

class PredictedTrack:
    """단일 트랙의 일정 horizon 미래 trajectory 예측.

    ``states`` 는 shape ``(T+1, 4)`` 의 ``np.ndarray`` 로 k 번째 행이
    ``t_ref + k*dt`` 시각의 ``[x_world, y_world, vx_world, vy_world]`` 이다.
    등속(CV) 모델의 경우 vx / vy 는 모든 행에서 동일하다. 0 행은 항상 측정된
    현재 상태이다.
    """

    def __init__(self, track_id, confidence, states, dt, horizon_s):
        self.track_id = track_id
        self.confidence = confidence
        self.states = states
        self.dt = dt
        self.horizon_s = horizon_s

    def state_at(self, t_offset_s):
        """``t_ref`` 기준 ``t_offset_s`` 초 후 예측 상태를 선형 보간으로 반환.

        t_offset_s : 예측 시작 시점 이후 경과 시간 [s]
        반환       : (x_world, y_world, vx_world, vy_world) 튜플.
                     horizon 밖이거나 states 가 비어 있으면 None.
        """
        if t_offset_s < 0.0 or t_offset_s > self.horizon_s + 1e-6:
            return None
        if self.states.size == 0 or self.dt <= 0.0:
            return None
        idx_f = t_offset_s / self.dt
        i0 = int(idx_f)
        i1 = i0 + 1
        n = self.states.shape[0]
        if i1 >= n:
            s = self.states[n - 1]
            return float(s[0]), float(s[1]), float(s[2]), float(s[3])
        frac = idx_f - i0
        s = self.states[i0] + (self.states[i1] - self.states[i0]) * frac
        return float(s[0]), float(s[1]), float(s[2]), float(s[3])


class PredictionSet:
    """한 tick 단위의 prediction snapshot 컨테이너.

    ``t_ref`` 는 ego clock (시뮬레이션 시각) 기준 예측 시작 시각이며,
    ``by_id`` 의 모든 트랙은 동일한 ``dt`` / ``horizon_s`` 를 공유한다.
    BehaviorPlanner 가 MOBIL 평가에서 worst-case (현재 vs 미래) gap 을 만들기
    위해 ``state_at()`` 을 호출한다.
    """

    def __init__(self, by_id=None, t_ref=0.0, dt=0.1, horizon_s=1.5):
        self.by_id = by_id if by_id is not None else {}
        self.t_ref = t_ref
        self.dt = dt
        self.horizon_s = horizon_s

    def get(self, track_id):
        """track_id 에 해당하는 PredictedTrack 을 반환. 없으면 None."""
        return self.by_id.get(track_id)

    def state_at(self, track_id, t_offset_s):
        """track_id 의 ``t_offset_s`` 초 후 예측 상태를 반환.

        track_id    : 조회할 트랙 id
        t_offset_s  : 예측 시작 시점 이후 경과 시간 [s]
        반환        : (x_world, y_world, vx_world, vy_world) 튜플 또는 None.
        """
        pt = self.by_id.get(track_id)
        if pt is None:
            return None
        return pt.state_at(t_offset_s)


# ---------------------------------------------------------------------------
# Behavior / 경로 계획
# ---------------------------------------------------------------------------

class BehaviorState(Enum):
    """주행 행동 상태. LocalPlanner / HMI 가 표시 및 분기에 사용한다."""

    LANE_KEEP = "LANE_KEEP"
    CAR_FOLLOW = "CAR_FOLLOW"
    STOP_AND_GO = "STOP_AND_GO"
    AEB = "AEB"
    LANE_CHANGE_PREPARE = "LC_PREPARE"
    LANE_CHANGE_EXECUTE = "LC_EXECUTE"
    LANE_CHANGE_ABORT = "LC_ABORT"
    LANE_CHANGE_COMPLETE = "LC_COMPLETE"
    MIN_RISK = "MIN_RISK"


class LaneChangeState(Enum):
    """LC 상태머신의 8 단계 상태."""

    IDLE = "IDLE"
    REQUESTED = "REQUESTED"
    PREPARE = "PREPARE"
    EXECUTE = "EXECUTE"
    COMPLETE = "COMPLETE"
    CANCEL = "CANCEL"
    ABORT = "ABORT"
    COOLDOWN = "COOLDOWN"


class TrajectoryPoint:
    """LocalPlanner 가 발행하는 trajectory 의 단일 샘플 (월드 좌표).

    LateralController 가 이 점들을 polyline 으로 보고 LQR 입력을 계산한다.
    """

    def __init__(self, x, y, yaw, curvature, speed_mps, accel_mps2, t):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.curvature = curvature
        self.speed_mps = speed_mps
        self.accel_mps2 = accel_mps2
        self.t = t


class BehaviorDecision:
    """BehaviorPlanner 의 출력. 매 tick LocalPlanner 가 소비한다.

    LC 상태/방향/진행률과, EXECUTE 시작 시점에 캡처한 원본 차선 중심선의
    월드 좌표 스냅샷을 함께 싣는다. LocalPlanner 는 본 폴리라인을 횡으로
    시프트해 LC 궤적을 생성한다.
    """

    def __init__(
        self,
        behavior_state,                     # BehaviorState
        lc_state,                           # LaneChangeState
        lc_direction=None,                  # LC 진행 중 "LEFT" / "RIGHT", 아니면 None
        lc_target_lane_id=None,
        lc_total_length_m=50.0,             # 계획된 LC 진행 거리
        lc_distance_done_m=0.0,             # LC 진행률 누적 거리
        lc_progress_ratio=0.0,              # LC s-커브 상에서 0 → 1 비율
        lc_reference_polyline_world=None,
        lc_safety_reason="",
        # SCC lead 마스킹 플래그. 현재 구현은 항상 False 로 유지하며, 마스킹 없이
        # lead 가 자연스럽게 current-lane 필터에서 빠지도록 둔다.
        mask_lead_for_scc=False,
    ):
        self.behavior_state = behavior_state
        self.lc_state = lc_state
        self.lc_direction = lc_direction
        self.lc_target_lane_id = lc_target_lane_id
        self.lc_total_length_m = lc_total_length_m
        self.lc_distance_done_m = lc_distance_done_m
        self.lc_progress_ratio = lc_progress_ratio
        self.lc_reference_polyline_world = (
            lc_reference_polyline_world if lc_reference_polyline_world is not None else []
        )
        self.lc_safety_reason = lc_safety_reason
        self.mask_lead_for_scc = mask_lead_for_scc


class LocalPlan:
    """LocalPlanner 출력 — 궤적 + 목표속도 + 안전 판정.

    LateralController 가 trajectory 로 조향을 계산하고, SCC 가
    desired_speed_kmh 를 set-speed 로 받아 종방향 제어를 한다.
    """

    def __init__(
        self,
        behavior_state,
        lane_change_state,
        target_lane_id,
        trajectory,
        desired_speed_kmh,
        lead_info,
        is_safe,
        safety_reason="",
    ):
        self.behavior_state = behavior_state
        self.lane_change_state = lane_change_state
        self.target_lane_id = target_lane_id
        self.trajectory = trajectory
        self.desired_speed_kmh = desired_speed_kmh
        self.lead_info = lead_info
        self.is_safe = is_safe
        self.safety_reason = safety_reason


class RouteContext:
    """현재 시점 ego 기준 글로벌 경로 스냅샷.

    RouteManager 가 글로벌 플랜을 1 회 구축한 뒤 매 tick 본 스냅샷을 갱신
    한다. 소비자(lane_provider, local_planner, BEV) 는 본 스냅샷만 읽으며,
    내부 그래프나 CARLA waypoint 를 직접 참조하지 않는다.
    """

    def __init__(
        self,
        is_valid=False,
        is_finished=False,
        # 최대 LOCAL_OBJECT_HORIZON_FRONT_M 전방까지의 (x, y) 쌍. BEV / planner 소비용.
        preview_waypoints_world=None,
        # 현재 도로 구간에서 경로가 지시하는 ego 의 lane ID.
        # ego 의 current_lane_id 와 다를 경우, 본 값이 권장(discretionary) 또는
        # 강제(mandatory) 목표 차선이 된다.
        target_lane_id=None,
        # 다음 경로 waypoint 의 OpenDRIVE road id (분기 인지형 차선 선택용)
        next_road_id=None,
        distance_to_next_junction=float('inf'),
        distance_to_route_lane_change=None,
        # 전방에 경로상 차선 변경이 필요하면 "LEFT" / "RIGHT", 아니면 None.
        mandatory_lane_change_direction=None,
        # 계획 경로 상의 진행 거리와 남은 거리 (모두 미터 단위).
        distance_along_route=0.0,
        distance_remaining=0.0,
        total_route_length=0.0,
    ):
        self.is_valid = is_valid
        self.is_finished = is_finished
        self.preview_waypoints_world = (
            preview_waypoints_world if preview_waypoints_world is not None else []
        )
        self.target_lane_id = target_lane_id
        self.next_road_id = next_road_id
        self.distance_to_next_junction = distance_to_next_junction
        self.distance_to_route_lane_change = distance_to_route_lane_change
        self.mandatory_lane_change_direction = mandatory_lane_change_direction
        self.distance_along_route = distance_along_route
        self.distance_remaining = distance_remaining
        self.total_route_length = total_route_length


# ---------------------------------------------------------------------------
# 디버그 / HMI
# ---------------------------------------------------------------------------

class ControlDebug:
    """제어기 단일 tick 의 디버그 출력. BEV sidebar 와 콘솔 로그가 소비한다."""

    def __init__(
        self,
        steer=0.0,
        throttle=0.0,
        brake=0.0,
        e_y=0.0,
        e_psi=0.0,
        delta_pp=0.0,
        delta_lqr=0.0,
        lookahead_dist=0.0,
        desired_speed_kmh=0.0,
        scc_state="",
    ):
        self.steer = steer
        self.throttle = throttle
        self.brake = brake
        self.e_y = e_y
        self.e_psi = e_psi
        self.delta_pp = delta_pp
        self.delta_lqr = delta_lqr
        self.lookahead_dist = lookahead_dist
        self.desired_speed_kmh = desired_speed_kmh
        self.scc_state = scc_state
