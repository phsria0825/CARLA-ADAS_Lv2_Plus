"""
교통 흐름 모델 — IDM (car-following) + MOBIL (lane-change).

본 모듈은 BehaviorPlanner 가 호출하는 종방향 / LC 의사결정의 핵심 함수를
제공한다. 모든 가속도 수치는 (속도, gap) 입력으로부터 IDM 식으로 계산되며,
외부로부터 측정된 가속도는 사용하지 않는다.

설계 원칙 — ground-truth 가속도 거부.
  본 모듈의 모든 가속도 (a_c, a_c_tilde, a_n, a_n_tilde, a_o, a_o_tilde) 는
  (speed, gap) 입력으로부터 IDM 으로 계산한다. CARLA 의 actor.get_acceleration()
  같은 ground-truth 는 사용하지 않으며, FusedObject schema 도 이러한 이유로
  acceleration 필드를 의도적으로 배제한다. 이는 MOBIL 공식화에서 비롯된 두
  가지 제약 때문이다.

    1. "after-LC" 가속도 (a_c_tilde, a_n_tilde, a_o_tilde) 는 반사실적
       (counterfactual) 값으로, LC 가 아직 발생하지 않았으므로 본질적으로
       측정 불가능하다. 따라서 모델 예측치여야 한다.
    2. incentive criterion 은 (a_tilde - a) 차이를 비교한다. 측정된 "before"
       와 모델링된 "after" 를 섞으면 두 항이 동일한 noise model 을 공유하지
       못하게 되어 비교가 깨진다. before 와 after 양쪽 모두 동일한 car-
       following 모델 (= IDM) 로 산출해야 incentive 식의 의미가 유지된다.
"""
import math


# ---------------------------------------------------------------------------
# IDM 파라미터 (passenger car 기본값)
# ---------------------------------------------------------------------------

class IDMParams:
    """IDM 모델의 6 개 운전자 파라미터 묶음.

    behavior_planner 가 매 tick config 값으로 인스턴스를 만들어
    idm_acceleration / mobil_evaluate 에 주입한다.
    """

    def __init__(
        self,
        v0_mps=33.3,   # 희망 속도 [m/s] -- 기본 120 km/h
        T_s=1.5,       # safe time headway [s]
        s0_m=2.0,      # 최소 정지 gap [m]
        a_mps2=1.4,    # 최대 가속도 [m/s²]
        b_mps2=2.0,    # 편안한 제동 [m/s²]
        delta=4.0,     # 가속도 지수
    ):
        self.v0_mps = v0_mps
        self.T_s = T_s
        self.s0_m = s0_m
        self.a_mps2 = a_mps2
        self.b_mps2 = b_mps2
        self.delta = delta


def idm_acceleration(v, v_lead, gap, params):
    """IDM 자유흐름 + 상호작용 항으로 종방향 가속도를 산출한다.

        dv/dt = a · (1 - (v/v0)^delta - (s*(v, Δv) / s)²)
        s*    = s0 + max(0, v·T + v·Δv / (2·√(a·b)))

    여기서 Δv = v - v_lead (접근 시 양수), gap 은 bumper-to-bumper 거리.
    선행 차량이 없으면 (gap = inf) 자유흐름 항만 남아 v → v0 로 점근 가속한다.

    v       : 자차 현재 속도 [m/s]
    v_lead  : 선행 차량 속도 [m/s]
    gap     : 선행 차량과의 bumper-to-bumper 거리 [m]
    params  : IDMParams
    반환    : 추천 종방향 가속도 [m/s²]
    """
    free = params.a_mps2 * (1.0 - (max(0.0, v) / max(params.v0_mps, 1e-3)) ** params.delta)
    if not math.isfinite(gap) or gap <= 0.0:
        return free
    dv = v - v_lead
    s_star = params.s0_m + max(
        0.0,
        v * params.T_s + (v * dv) / (2.0 * math.sqrt(params.a_mps2 * params.b_mps2)),
    )
    interaction = -params.a_mps2 * (s_star / gap) ** 2
    return free + interaction


# ---------------------------------------------------------------------------
# MOBIL
# ---------------------------------------------------------------------------

class MOBILParams:
    """MOBIL 모델의 4 개 파라미터 묶음 (politeness, safety, threshold, bias)."""

    def __init__(
        self,
        p=0.2,            # politeness factor (0 = egoistic, 1 = altruistic)
        b_safe_mps2=4.0,  # new follower 에게 유발되는 최대 허용 제동량
        a_th_mps2=0.2,    # 전환을 위한 threshold gain (hysteresis)
        a_bias_mps2=0.2,  # 비대칭 keep-right bias (US-style 의 경우 0)
    ):
        self.p = p
        self.b_safe_mps2 = b_safe_mps2
        self.a_th_mps2 = a_th_mps2
        self.a_bias_mps2 = a_bias_mps2


class MOBILLaneSnapshot:
    """한 target lane 에 대해 MOBIL 평가에 필요한 입력 — 전·후 차량 (speed, gap).

    BehaviorPlanner 가 FusedObject 의 ego-frame 데이터로부터 추출해 만든다.
    선행 차량이 없으면 gap = +inf, 후행 차량이 없으면 follower_gap = +inf.
    """

    def __init__(
        self,
        # target lane 의 전방 차량
        new_lead_speed,
        new_lead_gap,
        # target lane 의 후방 차량
        new_follower_speed,
        new_follower_gap,
    ):
        self.new_lead_speed = new_lead_speed
        self.new_lead_gap = new_lead_gap
        self.new_follower_speed = new_follower_speed
        self.new_follower_gap = new_follower_gap


class MOBILCurrentSnapshot:
    """현재 lane 의 ego 와 전·후 차량 (speed, gap) 입력 — MOBIL 평가의 baseline."""

    def __init__(
        self,
        ego_speed,
        # 현재 lane 의 ego 전방 차량
        old_lead_speed,
        old_lead_gap,
        # 현재 lane 의 ego 후방 차량
        old_follower_speed,
        old_follower_gap,
    ):
        self.ego_speed = ego_speed
        self.old_lead_speed = old_lead_speed
        self.old_lead_gap = old_lead_gap
        self.old_follower_speed = old_follower_speed
        self.old_follower_gap = old_follower_gap


class MOBILEvaluation:
    """단일 방향 LC 에 대한 MOBIL 평가 결과 + 6 개 가속도 디버그 값.

    BehaviorPlanner 가 LEFT / RIGHT 각각에 대해 본 객체를 받아 incentive 여유
    가 더 큰 쪽을 권고로 선택한다.
    """

    def __init__(
        self,
        direction,         # "LEFT" 또는 "RIGHT"
        safe,
        safety_reason,
        incentive_mps2,    # incentive criterion 의 좌변
        threshold_mps2,    # 우변 (a_th + bias)
        passes,
        # HMI / 디버깅을 위한 상세 가속도
        a_c,
        a_c_tilde,
        a_n,
        a_n_tilde,
        a_o,
        a_o_tilde,
    ):
        self.direction = direction
        self.safe = safe
        self.safety_reason = safety_reason
        self.incentive_mps2 = incentive_mps2
        self.threshold_mps2 = threshold_mps2
        self.passes = passes
        self.a_c = a_c
        self.a_c_tilde = a_c_tilde
        self.a_n = a_n
        self.a_n_tilde = a_n_tilde
        self.a_o = a_o
        self.a_o_tilde = a_o_tilde


def _idm_or_zero(have_follower, *args):
    """follower 가 있으면 idm_acceleration(*args), 없으면 0 을 반환하는 헬퍼."""
    if not have_follower:
        return 0.0
    return idm_acceleration(*args)


def mobil_evaluate(
    direction,
    current,
    target,
    idm_params,
    mobil_params,
    ego_length_m=4.7,
):
    """한 방향(LEFT 또는 RIGHT) 차선 변경에 대해 MOBIL safety + incentive 평가.

    LC 전후의 6 개 가속도 (ego, old follower, new follower 각각의 "before"
    와 "after") 를 모두 IDM 으로 계산한 뒤 두 조건을 동시에 만족하는지 본다.

        safety   : a_n_tilde ≥ -b_safe          (new follower 가 안전한 감속)
        incentive: (a_c_tilde - a_c) + p · [(a_n_tilde - a_n) + (a_o_tilde - a_o)]
                   > a_th + bias

    여기서 bias 는 RIGHT (keep-right) 시 -a_bias, LEFT (추월) 시 +a_bias.
    a_bias = 0 으로 두면 대칭형 (US-style) MOBIL 이 복원된다.

    direction    : "LEFT" 또는 "RIGHT"
    current      : 현재 lane 의 ego / 전·후 차량 스냅샷
    target       : target lane 의 전·후 차량 스냅샷
    idm_params   : IDMParams
    mobil_params : MOBILParams
    ego_length_m : ego 차량 길이 [m]. LC 후 new-follower 와 new-leader 사이
                   gap 을 계산할 때 사용한다.
    반환         : MOBILEvaluation (6 개 가속도 + safe / incentive / passes)
    """
    has_new_lead = math.isfinite(target.new_lead_gap)
    has_new_follower = math.isfinite(target.new_follower_gap)
    has_old_lead = math.isfinite(current.old_lead_gap)
    has_old_follower = math.isfinite(current.old_follower_gap)

    # -------------- lane change 이전의 가속도 --------------
    # a_c: 현재 lane 의 ego 가 현재 leader 를 추종
    a_c = idm_acceleration(
        v=current.ego_speed,
        v_lead=current.old_lead_speed if has_old_lead else current.ego_speed,
        gap=current.old_lead_gap if has_old_lead else float('inf'),
        params=idm_params,
    )
    # a_o: ego 후방의 old follower 가 ego 를 추종
    a_o = _idm_or_zero(
        has_old_follower,
        current.old_follower_speed, current.ego_speed,
        current.old_follower_gap, idm_params,
    )
    # a_n: target lane 의 new follower 가 target lane 의 현재 leader 를 추종.
    # 두 차량 사이 gap 은 (new follower → ego 까지의 gap) + ego_length + (ego →
    # new leader 까지의 gap) 로 합성한다.
    if has_new_follower and has_new_lead:
        gap_n_to_new_lead = (
            target.new_follower_gap + ego_length_m + target.new_lead_gap
        )
        a_n = idm_acceleration(
            v=target.new_follower_speed,
            v_lead=target.new_lead_speed,
            gap=gap_n_to_new_lead,
            params=idm_params,
        )
    elif has_new_follower:
        # target-lane leader 없음 → free flow
        a_n = idm_acceleration(
            v=target.new_follower_speed,
            v_lead=target.new_follower_speed,
            gap=float('inf'),
            params=idm_params,
        )
    else:
        a_n = 0.0

    # -------------- lane change 이후의 가속도 --------------
    # a_c_tilde: ego 가 target lane 으로 이동해 new leader 를 추종
    a_c_tilde = idm_acceleration(
        v=current.ego_speed,
        v_lead=target.new_lead_speed if has_new_lead else current.ego_speed,
        gap=target.new_lead_gap if has_new_lead else float('inf'),
        params=idm_params,
    )
    # a_n_tilde: new follower 가 이제 ego 를 추종
    a_n_tilde = _idm_or_zero(
        has_new_follower,
        target.new_follower_speed, current.ego_speed,
        target.new_follower_gap, idm_params,
    )
    # a_o_tilde: old follower 전방에 ego 가 없어졌으므로 → old leader 를 추종
    if has_old_follower and has_old_lead:
        gap_o_to_old_lead = (
            current.old_follower_gap + ego_length_m + current.old_lead_gap
        )
        a_o_tilde = idm_acceleration(
            v=current.old_follower_speed,
            v_lead=current.old_lead_speed,
            gap=gap_o_to_old_lead,
            params=idm_params,
        )
    elif has_old_follower:
        a_o_tilde = idm_acceleration(
            v=current.old_follower_speed,
            v_lead=current.old_follower_speed,
            gap=float('inf'),
            params=idm_params,
        )
    else:
        a_o_tilde = 0.0

    # -------------- safety criterion --------------
    safe = a_n_tilde >= -mobil_params.b_safe_mps2
    safety_reason = "" if safe else f"a_n_tilde={a_n_tilde:+.2f} < -b_safe"

    # -------------- incentive criterion (비대칭 keep-right) --------------
    # 우측 통행 환경에서는 우측 차로가 자연 차선이므로, 우측 LC 는 threshold
    # 가 낮고 (쉽게 발동) 좌측 LC (추월) 는 threshold 가 높다.
    #   threshold(LC to RIGHT) = a_th - a_bias
    #   threshold(LC to LEFT)  = a_th + a_bias
    # a_bias = 0 으로 두면 대칭형 MOBIL 이 복원된다.
    if direction == "RIGHT":
        bias = -mobil_params.a_bias_mps2
    elif direction == "LEFT":
        bias = +mobil_params.a_bias_mps2
    else:
        bias = 0.0

    incentive = (a_c_tilde - a_c) + mobil_params.p * (
        (a_n_tilde - a_n) + (a_o_tilde - a_o)
    )
    threshold = mobil_params.a_th_mps2 + bias
    passes = safe and (incentive > threshold)

    return MOBILEvaluation(
        direction=direction,
        safe=safe,
        safety_reason=safety_reason,
        incentive_mps2=incentive,
        threshold_mps2=threshold,
        passes=passes,
        a_c=a_c, a_c_tilde=a_c_tilde,
        a_n=a_n, a_n_tilde=a_n_tilde,
        a_o=a_o, a_o_tilde=a_o_tilde,
    )
