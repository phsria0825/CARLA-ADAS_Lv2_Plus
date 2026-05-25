"""
Local planner — lane-keeping 궤적 + LC EXECUTE 횡 시프트 궤적을 생성.

역할:
    - 현재 차선 중심선으로부터 lane-keeping 궤적을 생성한다.
    - LC EXECUTE 진행 중에는 횡으로 시프트된 LC 기준 폴리라인을 생성한다.
    - LeadInfo 에 따라 LANE_KEEP / CAR_FOLLOW 를 결정한다.
    - SCCController 가 소비할 desired_speed_kmh 를 발행한다.

출력되는 LocalPlan.trajectory 는 LateralController 가 소비하므로 다음 조건을
만족해야 한다.
    * 월드 좌표계 (lateral_error_to_polyline 이 월드 좌표를 가정한다)
    * 컨트롤러가 타깃점을 해소할 수 있을 만큼 조밀하게 샘플링되어 있어야 한다
    * LQR feed-forward 를 위해 yaw 와 곡률을 함께 실어야 한다
"""
import math

import config
from core.adas_types import (
    BehaviorState,
    LaneChangeState,
    LocalPlan,
    TrajectoryPoint,
)
from core.adas_utils import project_onto_polyline


# =====================================================================
# Quintic 5차 다항식 횡 offset 프로파일
# =====================================================================
# y(τ)/Δy = 10τ³ - 15τ⁴ + 6τ⁵  (τ = t/T, τ ∈ [0,1])
# 경계조건: y(0)=0, y'(0)=0, y''(0)=0, y(T)=Δy, y'(T)=0, y''(T)=0
# 따라서 LC 시작/종료 시점에 횡속도와 횡가속도가 0 이 되어 jerk 트랜지언트가
# 없는 부드러운 차선 변경을 만든다.
#
# 시간 미분 peak (τ ∈ [0,1] 의 극값에서 정확히 닫힌 형식):
#   ay_peak   = _QUINTIC_AY_COEFF   · |Δy| / T²    (τ = 0.5 ± √3/6 에서)
#   jerk_peak = _QUINTIC_JERK_COEFF · |Δy| / T³    (τ = 0, 1 에서)
# _QUINTIC_AY_COEFF = (10·√3)/3 ≈ 5.7735 로 closed-form 으로 유도한 값.
_QUINTIC_AY_COEFF = 10.0 * math.sqrt(3.0) / 3.0
_QUINTIC_JERK_COEFF = 60.0


def _quintic_position_profile(tau):
    """정규화된 시간 τ ∈ [0,1] 에서 quintic 위치 비율을 반환한다.

    y(τ)/Δy = 10τ³ - 15τ⁴ + 6τ⁵.
    τ ≤ 0 이면 0, τ ≥ 1 이면 1 로 자른다.
    """
    if tau <= 0.0:
        return 0.0
    if tau >= 1.0:
        return 1.0
    return tau * tau * tau * (10.0 - 15.0 * tau + 6.0 * tau * tau)


def _smoothstep(x):
    """Deprecated 별칭 — _quintic_position_profile 과 동일한 결과를 반환한다."""
    return _quintic_position_profile(x)


def _quintic_min_time(delta_y_abs, ay_limit, jerk_limit, t_floor):
    """|ay| ≤ ay_limit 와 |jerk| ≤ jerk_limit 를 동시에 만족하는 최소 LC 시간 T.

    Closed-form:
        T_ay   = √(_QUINTIC_AY_COEFF   · |Δy| / ay_limit)
        T_jerk = ∛(_QUINTIC_JERK_COEFF · |Δy| / jerk_limit)
        T      = max(t_floor, T_ay, T_jerk)
    """
    dy = max(1e-6, abs(delta_y_abs))
    t_ay = math.sqrt(_QUINTIC_AY_COEFF * dy / max(ay_limit, 1e-6))
    t_jerk = (_QUINTIC_JERK_COEFF * dy / max(jerk_limit, 1e-6)) ** (1.0 / 3.0)
    return max(t_floor, t_ay, t_jerk)


def solve_lc_length(
    lane_width_m,
    v_ego_mps,
    ay_limit=None,
    jerk_limit=None,
    t_floor=None,
    t_ceiling=None,
    length_min=None,
    length_max=None,
):
    """Quintic ay / jerk 한계를 만족하는 (lc_length_m, lc_time_s) 산출.

    1) ay / jerk 한계로 T_min 을 구하고 t_floor / t_ceiling 으로 clamp.
    2) lc_length = T · v_ego (등속 가정) 를 length_min / length_max 로 clamp.

    None 인 인자는 config 의 LC_MAX_LATERAL_ACCEL_MPS2 /
    LC_MAX_LATERAL_JERK_MPS3 / LC_MIN_TOTAL_TIME_S / LC_MAX_TOTAL_TIME_S /
    LC_MIN_LANE_CHANGE_LENGTH_M / LC_MAX_LANE_CHANGE_LENGTH_M 기본값을 쓴다.

    lane_width_m : LC 횡변위 [m]
    v_ego_mps    : 현재 ego 속도 [m/s]
    반환         : (lc_length [m], lc_time [s])
    """
    ay_limit = ay_limit if ay_limit is not None else config.LC_MAX_LATERAL_ACCEL_MPS2
    jerk_limit = jerk_limit if jerk_limit is not None else config.LC_MAX_LATERAL_JERK_MPS3
    t_floor = t_floor if t_floor is not None else config.LC_MIN_TOTAL_TIME_S
    t_ceiling = t_ceiling if t_ceiling is not None else config.LC_MAX_TOTAL_TIME_S
    length_min = length_min if length_min is not None else config.LC_MIN_LANE_CHANGE_LENGTH_M
    length_max = length_max if length_max is not None else config.LC_MAX_LANE_CHANGE_LENGTH_M

    t_lim = _quintic_min_time(lane_width_m, ay_limit, jerk_limit, t_floor)
    t = min(t_ceiling, t_lim)
    v_safe = max(v_ego_mps, 0.5)
    length_unclamped = t * v_safe
    length = max(length_min, min(length_max, length_unclamped))
    return length, t


def _lane_change_trajectory(
    ref_polyline,
    ego_x,
    ego_y,
    direction,                 # "LEFT" or "RIGHT"
    lc_total_length_m,
    lane_width_m,
    speed_mps,
):
    """LC EXECUTE 동안 컨트롤러가 추종할, 횡으로 시프트된 폴리라인을 생성한다.

    기준 폴리라인은 LC EXECUTE 시작 시점에 캡처한 스냅샷이며, ref_polyline[0]
    은 그 순간의 ego.xy 이다. 호 길이 s 는 이 앵커로부터 측정되므로 maneuver
    는 s ∈ [0, lc_total_length_m] 구간에서 진행되며, 횡 offset 프로파일은
    다음과 같다.

        offset(s) = direction_sign · lane_width · quintic(s / lc_length)

    따라서 본 함수가 발행하는 궤적은 스냅샷 상의 EGO 투영점에서 시작하되
    s_ego 에 해당하는 offset (0 이 아니다!) 을 이미 적용한 채로 전방으로 확장
    되며, 절대 호 길이에 따라 offset 이 증가한다. 이 구성이 매 tick 궤적이
    안정적이게 하는 핵심 — 매 tick ego 가 s 를 따라 더 진행했더라도 동일한
    (s -> offset) 함수가 동일한 경로를 계속 만들어내기 때문이다.

    부호 컨벤션: CARLA 의 left-handed frame 에서 LEFT 란 각 접선 점에서의
    차체 LEFT 단위벡터를 의미하며, 접선 (dx, dy) 에 대해 (dy/seg, -dx/seg)
    이다.

    ref_polyline       : EXECUTE 시작 시점에 캡처한 lane.centerline 스냅샷
    ego_x, ego_y       : 현 ego 위치 (월드 좌표)
    direction          : "LEFT" 또는 "RIGHT"
    lc_total_length_m  : solve_lc_length 가 정한 LC 횡 시프트 총 거리
    lane_width_m       : 차선 폭 (= 횡 변위 |Δy|)
    speed_mps          : 현 ego 속도 (사용 안 함, 시그니처 호환 목적)
    반환               : 시프트된 polyline [(x, y), ...]
    """
    if len(ref_polyline) < 2:
        return list(ref_polyline)

    # 스냅샷 상 ego 투영점의 호 길이
    seg_i, t_seg, s_ego = project_onto_polyline(ref_polyline, ego_x, ego_y)

    # 스냅샷 시작점으로부터 각 점까지의 누적 호 길이
    cum_s = [0.0] * len(ref_polyline)
    for i in range(1, len(ref_polyline)):
        ax, ay = ref_polyline[i - 1]
        bx, by = ref_polyline[i]
        cum_s[i] = cum_s[i - 1] + math.hypot(bx - ax, by - ay)

    direction_sign = +1.0 if direction == "LEFT" else -1.0
    lc_len = max(lc_total_length_m, 1e-3)

    def _alpha(s_val):
        return _quintic_position_profile(min(1.0, max(0.0, s_val / lc_len)))

    def _left_unit_at_segment(i):
        """segment i 의 left 단위벡터 (CARLA left-handed 월드 좌표계 기준)."""
        if i + 1 < len(ref_polyline):
            dx = ref_polyline[i + 1][0] - ref_polyline[i][0]
            dy = ref_polyline[i + 1][1] - ref_polyline[i][1]
        else:
            dx = ref_polyline[i][0] - ref_polyline[i - 1][0]
            dy = ref_polyline[i][1] - ref_polyline[i - 1][1]
        seg = math.hypot(dx, dy)
        if seg < 1e-6:
            return 0.0, 0.0
        return dy / seg, -dx / seg

    out = []

    # 첫 점: 스냅샷 상 ego 투영점에 s_ego 기반 alpha 만큼 OFFSET 을 적용.
    # 현재까지 진행된 maneuver 진척도에 맞춰 ego 가 있어야 할 경로상의 위치
    # 이며, LQR 이 ego 를 이 점으로 끌어당기게 된다.
    ax, ay = ref_polyline[seg_i]
    bx, by = ref_polyline[min(seg_i + 1, len(ref_polyline) - 1)]
    sx_seg = bx - ax
    sy_seg = by - ay
    seg_len = math.hypot(sx_seg, sy_seg)
    if seg_len < 1e-6:
        return list(ref_polyline)
    proj_x = ax + sx_seg * t_seg
    proj_y = ay + sy_seg * t_seg
    left_x_ego, left_y_ego = _left_unit_at_segment(seg_i)
    offset_ego = direction_sign * lane_width_m * _alpha(s_ego)
    out.append((proj_x + offset_ego * left_x_ego,
                proj_y + offset_ego * left_y_ego))

    # 이후 ego 너머의 스냅샷 점들. 각 점의 alpha 는 해당 점의 절대 호 길이.
    for i in range(seg_i + 1, len(ref_polyline)):
        left_x, left_y = _left_unit_at_segment(i)
        offset = direction_sign * lane_width_m * _alpha(cum_s[i])
        px = ref_polyline[i][0] + offset * left_x
        py = ref_polyline[i][1] + offset * left_y
        out.append((px, py))

    return out


def _polyline_to_trajectory(polyline, speed_mps, accel_mps2):
    """(x, y) polyline → yaw / 곡률 / 시간이 포함된 TrajectoryPoint 리스트로 변환.

    yaw 는 forward difference (마지막 점은 직전 segment 의 yaw 재사용).
    곡률은 가능한 경우 central difference 로 dyaw/ds 계산.
    타임스탬프는 등속 가정으로 누적 (종방향은 SCC 가 별도 관리).

    polyline    : [(x, y), ...] 월드 좌표
    speed_mps   : 모든 점에 동일하게 부여할 속도
    accel_mps2  : 모든 점에 동일하게 부여할 가속도
    반환        : list[TrajectoryPoint]
    """
    out = []
    n = len(polyline)
    if n < 2:
        return out

    yaws = []
    for i in range(n):
        if i + 1 < n:
            dx = polyline[i + 1][0] - polyline[i][0]
            dy = polyline[i + 1][1] - polyline[i][1]
        else:
            dx = polyline[i][0] - polyline[i - 1][0]
            dy = polyline[i][1] - polyline[i - 1][1]
        yaws.append(math.atan2(dy, dx))

    curvatures = []
    for i in range(n):
        if 0 < i < n - 1:
            dy = yaws[i + 1] - yaws[i - 1]
            while dy > math.pi:
                dy -= 2 * math.pi
            while dy < -math.pi:
                dy += 2 * math.pi
            ds = math.hypot(
                polyline[i + 1][0] - polyline[i - 1][0],
                polyline[i + 1][1] - polyline[i - 1][1],
            )
            curvatures.append(dy / ds if ds > 1e-6 else 0.0)
        else:
            curvatures.append(0.0)

    safe_v = max(speed_mps, 0.5)
    t = 0.0
    prev_x, prev_y = polyline[0]
    for i in range(n):
        x, y = polyline[i]
        if i > 0:
            ds = math.hypot(x - prev_x, y - prev_y)
            t += ds / safe_v
        out.append(TrajectoryPoint(
            x=x, y=y,
            yaw=yaws[i],
            curvature=curvatures[i],
            speed_mps=speed_mps,
            accel_mps2=accel_mps2,
            t=t,
        ))
        prev_x, prev_y = x, y
    return out


class LocalPlanner:
    """매 tick 의 행동/궤적/목표속도를 묶어 LocalPlan 으로 발행한다.

    기본 동작은 현재 차선 중심선 추종이며, BehaviorDecision 이 LC EXECUTE
    상태이면 횡으로 시프트된 quintic 궤적으로 전환한다. 경로 종점 접근 시는
    desired_speed 를 ramp-down 해 SCC 가 부드럽게 감속하도록 유도한다.
    """

    def __init__(self):
        self.set_speed_kmh = config.SET_SPEED

    def plan(
        self,
        ego_state,
        lane,
        lead_info,
        route_context=None,
        behavior_decision=None,
    ):
        """한 tick 의 LocalPlan 을 산출한다.

        LC EXECUTE 진행 중에는 BehaviorDecision 이 들고 있는 reference
        polyline 스냅샷을 횡으로 시프트해 trajectory 를 만들고, 그 외 상태
        에서는 lane.centerline 을 그대로 추종한다. 경로가 종료/접근 중이면
        desired_speed_kmh 를 0 까지 선형으로 ramp down 해 SCC 의 soft arrival
        을 돕는다.

        ego_state         : 자차 상태
        lane              : 현재 LaneModel
        lead_info         : 현재 차선 lead (LANE_KEEP vs CAR_FOLLOW 결정)
        route_context     : 전역 경로 스냅샷 (선택)
        behavior_decision : BehaviorPlanner 출력 (LC 상태/방향/폴리라인)
        반환              : LocalPlan
        """
        if not lane.is_valid:
            return LocalPlan(
                behavior_state=BehaviorState.MIN_RISK,
                lane_change_state=LaneChangeState.IDLE,
                target_lane_id=0,
                trajectory=[],
                desired_speed_kmh=0.0,
                lead_info=lead_info,
                is_safe=False,
                safety_reason="LANE_INVALID",
            )

        # desired_speed_kmh 가 0 으로 떨어지면 SCC 가 제동하므로, 본 구간은
        # hard stop 이 아닌 "soft arrival" 을 구성한다.
        route_finished = route_context is not None and route_context.is_valid \
            and route_context.is_finished
        approach_arrival = (
            route_context is not None and route_context.is_valid
            and 0.0 < route_context.distance_remaining < 30.0
        )

        # 기본 behavior 는 lead 존재 여부로부터 도출. LC 진행 중이면
        # BehaviorPlanner 의 상태로 override.
        if lead_info.detected and lead_info.range < config.LEAD_DETECT_DIST:
            behavior = BehaviorState.CAR_FOLLOW
        else:
            behavior = BehaviorState.LANE_KEEP
        lc_state_out = LaneChangeState.IDLE

        # ----- 궤적 source 선택 -----
        traj_polyline = lane.centerline
        if (
            behavior_decision is not None
            and behavior_decision.lc_state == LaneChangeState.EXECUTE
            and behavior_decision.lc_reference_polyline_world
            and behavior_decision.lc_direction in ("LEFT", "RIGHT")
        ):
            traj_polyline = _lane_change_trajectory(
                ref_polyline=behavior_decision.lc_reference_polyline_world,
                ego_x=ego_state.x,
                ego_y=ego_state.y,
                direction=behavior_decision.lc_direction,
                lc_total_length_m=behavior_decision.lc_total_length_m,
                lane_width_m=lane.lane_width,
                speed_mps=ego_state.speed_mps,
            )
            behavior = behavior_decision.behavior_state
            lc_state_out = behavior_decision.lc_state
        elif behavior_decision is not None and behavior_decision.lc_state != LaneChangeState.IDLE:
            # PREPARE / COMPLETE / COOLDOWN: 현재 차선을 계속 추종하되 HMI 가
            # 실제 LC 상태를 반영하도록 노출만 한다.
            behavior = behavior_decision.behavior_state
            lc_state_out = behavior_decision.lc_state

        trajectory = _polyline_to_trajectory(
            traj_polyline,
            speed_mps=max(ego_state.speed_mps, 0.5),
            accel_mps2=0.0,
        )

        # ----- desired speed 산출 -----
        # SCC 상태 기계가 이미 lead 거리 / TTC 를 게이트하므로, 기본적으로
        # set speed 를 그대로 전달. 경로 종점 접근 시 마지막 30 m 에서 0 까지
        # 선형 ramp.
        if route_finished:
            desired_speed_kmh = 0.0
            behavior = BehaviorState.MIN_RISK
            safety_reason = "ROUTE_FINISHED"
        elif approach_arrival:
            ratio = max(0.0, route_context.distance_remaining / 30.0)
            desired_speed_kmh = self.set_speed_kmh * ratio
            safety_reason = "APPROACHING_DESTINATION"
        else:
            desired_speed_kmh = self.set_speed_kmh
            safety_reason = ""

        return LocalPlan(
            behavior_state=behavior,
            lane_change_state=lc_state_out,
            target_lane_id=lane.current_lane_id,
            trajectory=trajectory,
            desired_speed_kmh=desired_speed_kmh,
            lead_info=lead_info,
            is_safe=not route_finished,
            safety_reason=safety_reason,
        )
