"""
BehaviorPlanner — IDM 위에 MOBIL 기반의 자유 차선 변경을 수행하며 LC 상태
머신을 함께 구현한다.

매 tick 의 파이프라인:
    1) CURRENT 차로에 대해 FusedObjects 로부터 leader / follower 를 추출한다.
    2) 접근 가능한 인접 차로 (LEFT / RIGHT) 각각에 대해 다음을 수행한다.
       - new_lead 와 new_follower 를 추출한다
       - traffic_models.mobil_evaluate() 를 실행해 MOBILEvaluation 을 얻는다
    3) MOBIL 안전성과 incentive 를 동시에 만족하는 LEFT / RIGHT 평가 중에서
       incentive 여유가 더 큰 쪽을 선택한다 (= 권고 LC).
    4) 권고가 LC_DECISION_HYSTERESIS_S 동안 유지되어야 commit (IDLE →
       REQUESTED). MOBIL 의 a_th 가 이미 hysteresis 를 제공하지만, 본 추가
       디바운스는 인지 단계의 gap/range 진동을 1차 차단한다. IDM 의
       (s*/s)² 상호작용 항이 진동을 증폭시키기 때문이다.
    5) 상태 기계: IDLE → REQUESTED → PREPARE → EXECUTE → COMPLETE →
       COOLDOWN → IDLE.

본 모듈은 교과서적인 의사결정 로직이며, 시나리오별 특수 분기는 두지 않는다.
"""
import math

import config
from core.adas_types import (
    BehaviorDecision,
    BehaviorState,
    LaneChangeState,
)
from planning.traffic_models import (
    IDMParams,
    MOBILCurrentSnapshot,
    MOBILLaneSnapshot,
    MOBILParams,
    mobil_evaluate,
)


_INF = float('inf')


class LaneInfo:
    """단일 차로 (CURRENT / LEFT / RIGHT) 의 전·후방 차량 요약.

    front_gap_m 이 inf 이면 전방 차량 없음, has_rear 가 False 이면 후방
    차량 없음. _lane_neighbors() 가 fused 객체로부터 산출해 채운다.
    """

    def __init__(
        self,
        has_front,
        front_gap_m=_INF,
        front_speed_mps=0.0,
        has_rear=False,
        rear_gap_m=_INF,
        rear_speed_mps=0.0,
    ):
        self.has_front = has_front
        self.front_gap_m = front_gap_m
        self.front_speed_mps = front_speed_mps
        self.has_rear = has_rear
        self.rear_gap_m = rear_gap_m
        self.rear_speed_mps = rear_speed_mps


# ---------------------------------------------------------------------------
# 모듈 레벨 헬퍼 함수
# ---------------------------------------------------------------------------

def _predicted_x_ego(obj, ego, predictions, t_pred_s):
    """객체의 t_pred_s 후 ego-frame 종방향 위치를 worst-case 평가용으로 산출.

    CV 가정 하에서 ego 와 객체 모두 등속으로 움직인다고 보고, world frame
    에서 두 점의 변위를 ego yaw 기준으로 회전해 ego frame 종방향 좌표를
    구한다. predictions 가 None 이거나 해당 트랙이 없으면 None 을 반환해
    호출자가 현재 값만으로 fallback 하도록 한다.

    obj          : FusedObject
    ego          : 현재 EgoState
    predictions  : PredictionSet 또는 None
    t_pred_s     : 미래 시점 [s]
    반환         : 미래 ego-frame 종방향 위치 [m] 또는 None
    """
    if predictions is None or ego is None:
        return None
    fut = predictions.state_at(obj.track_id, t_pred_s)
    if fut is None:
        return None
    obj_x_w, obj_y_w, _, _ = fut
    # ego 도 등속 가정 — yaw 일정.
    ex_f = ego.x + ego.speed_mps * math.cos(ego.yaw_rad) * t_pred_s
    ey_f = ego.y + ego.speed_mps * math.sin(ego.yaw_rad) * t_pred_s
    dx = obj_x_w - ex_f
    dy = obj_y_w - ey_f
    c = math.cos(ego.yaw_rad)
    s = math.sin(ego.yaw_rad)
    # world → body 회전 (ego yaw 만큼 반대로 돌림)
    x_ego_future = c * dx + s * dy
    return x_ego_future


def _lane_neighbors(position, ego, lane, fused_objects, predictions=None, t_pred_s=1.0):
    """ego frame y 좌표로 객체를 차로별로 분류하고 전·후방 가장 가까운 차량을 뽑는다.

    각 객체에 대해 "지금" 과 "t_pred_s 후" 두 시점의 ego-frame 종방향 위치를
    모두 살펴 더 가까운 (worst-case) 값을 front_gap / rear_gap 산출에 쓴다.
    결과적으로 1 초 안에 발생할 cut-in / closing 위협을 MOBIL 평가가 한
    박자 먼저 보게 된다. 속도 자체는 CV 가정 하에서 변하지 않으므로
    front_speed / rear_speed 는 그대로 o.vx_ego 를 사용한다.

    FusedObject.vx_ego 는 객체의 절대 종방향 속도를 ego 전방축에 투영한
    값이다. MOBIL 이 가정하는 동방향 차로 (인접 차로) 에서는 이 값이 객체
    도로 좌표계 속도와 같다.

    position       : "CURRENT" | "LEFT" | "RIGHT"
    ego            : 현재 EgoState
    lane           : 현재 LaneModel (lane_width 사용)
    fused_objects  : confirmed FusedObject 리스트
    predictions    : PredictionSet 또는 None
    t_pred_s       : worst-case 평가에 쓸 미래 시점 [s]
    반환           : LaneInfo (전·후방 gap / speed)
    """
    lane_w = max(lane.lane_width, 2.5)
    if position == "CURRENT":
        lat_lo, lat_hi = -(lane_w * 0.5 + 0.5), +(lane_w * 0.5 + 0.5)
    elif position == "LEFT":
        lat_lo, lat_hi = -(lane_w * 1.5 + 0.5), -(lane_w * 0.5 - 0.5)
    elif position == "RIGHT":
        lat_lo, lat_hi = +(lane_w * 0.5 - 0.5), +(lane_w * 1.5 + 0.5)
    else:
        return LaneInfo(has_front=False, has_rear=False)

    front_gap = _INF
    front_speed = 0.0
    rear_gap = _INF
    rear_speed = 0.0
    for o in fused_objects:
        if not (lat_lo < o.y_ego < lat_hi):
            continue

        x_now = o.x_ego
        x_pred = _predicted_x_ego(o, ego, predictions, t_pred_s)

        # ---- 전방 영역 평가 ----
        # 현재가 전방이면 항상 후보. 예측 위치도 전방 (>1 m) 이면 더 작은
        # (= 더 가까운) 값으로 worst-case 갱신.
        if x_now > 1.0:
            x_front_eff = x_now
            if x_pred is not None and x_pred > 1.0:
                x_front_eff = min(x_front_eff, x_pred)
            if x_front_eff < front_gap:
                front_gap = x_front_eff
                front_speed = max(0.0, o.vx_ego)

        # ---- 후방 영역 평가 ----
        # ego 후방 탐색 범위를 REAR_HORIZON 으로 제한해 멀리 떨어진 후방
        # 차량이 LC 진행 중 abort 를 유발하지 않도록 한다. LC 진입 전 MOBIL
        # 게이팅은 영향을 안 받는데, 해당 범위에서는 IDM 의 (s*/s)² 항이
        # 이미 zero interaction 을 갖기 때문이다.
        if -config.LC_EXECUTE_REAR_HORIZON_M < x_now < -1.0:
            d_eff = -x_now
            if x_pred is not None and x_pred < -1.0:
                d_eff = min(d_eff, -x_pred)
            if d_eff < rear_gap:
                rear_gap = d_eff
                rear_speed = max(0.0, o.vx_ego)
    return LaneInfo(
        has_front=math.isfinite(front_gap),
        front_gap_m=front_gap,
        front_speed_mps=front_speed,
        has_rear=math.isfinite(rear_gap),
        rear_gap_m=rear_gap,
        rear_speed_mps=rear_speed,
    )


def _pick_recommendation(evaluations):
    """pass 한 LEFT / RIGHT MOBIL evaluation 중 incentive 여유가 가장 큰 쪽 선택.

    incentive 여유 = (incentive_mps2 - threshold_mps2). 통과한 후보가 없으면
    None 을 반환해 호출자가 LC 권고 없음으로 처리하도록 한다. 동률 발생 시
    dict 순회 순서에 의존하지만, 통과 후 incentive 차이가 크게 갈리는 경우
    가 대부분이라 실제 비결정성은 거의 없다.
    """
    passing = [(d, e) for d, e in evaluations.items() if e.passes]
    if not passing:
        return None
    passing.sort(key=lambda x: (x[1].incentive_mps2 - x[1].threshold_mps2),
                 reverse=True)
    return passing[0][0]


class BehaviorPlanner:
    """LC 의사결정과 LC 8-state 상태머신을 매 tick 진행한다.

    update() 가 유일한 진입점이며, 매 tick BehaviorDecision 한 개를 발행해
    LocalPlanner 에 전달한다. 내부적으로는 MOBIL 평가, hysteresis 추적,
    ODD 게이트, 후방 안전 재검증, ABORT / COMMIT 분기를 처리한다.
    """

    def __init__(self):
        # ---- LC 상태 기계 ----
        self.lc_state = LaneChangeState.IDLE
        self.lc_direction = None
        self.lc_origin_lane_id = None
        self.lc_target_lane_id = None
        self.lc_request_t = None
        self.lc_prepare_start_t = None
        self.lc_exec_start_t = None
        self.lc_exec_start_xy = None
        self.lc_reference_polyline = []
        self.lc_total_length_m = config.LC_MIN_LANE_CHANGE_LENGTH_M
        self.lc_safety_reason = ""
        self.cooldown_until_t = -1.0

        # ---- 의사결정 hysteresis ----
        self.current_recommendation = None
        self.first_recommendation_t = None

        # ---- Telemetry / HMI ----
        self.last_mobil_evaluations = {}
        self.last_lane_infos = {}

        # ---- 진단 출력 — 주기적 콘솔 dump 로 LC 비발화 사유를 노출 ----
        self._last_diag_t = -1e9
        self._diag_period_s = 2.0

        # ---- LC 진행 중 ABORT/COMMIT 관리용 ----
        # EXECUTE 도중 COMMIT 결정이 내려진 첫 tick 에만 True 로 set 해 콘솔
        # spam 을 방지. 새 EXECUTE 진입마다 False 로 reset.
        self._committed = False

    def update(
        self,
        t,
        ego,
        lane,
        lead,                  # 미사용 (lane 정보는 fused 에서 재구성)
        scc_state_value,       # 미사용 (API 호환용)
        set_speed_kmh,
        fused_objects,
        route_context=None,
        # 인접 차로 lead 와 동기화된 LeadInfoSet. 현재는 PREPARE 재검증
        # 흐름에서만 보조적으로 참조한다.
        lead_set=None,
        # ConstantVelocityPredictor 가 산출한 미래 trajectory 예측. None
        # 이면 기존 snapshot-only 동작 유지.
        predictions=None,
    ):
        """한 tick 의 LC 의사결정 + 상태머신 step 을 진행해 BehaviorDecision 반환.

        파이프라인:
            1) IDM/MOBIL 파라미터 인스턴스화
            2) 차로별 인접 차량 요약 (worst-case gap)
            3) 좌·우 차로에 대해 MOBIL 평가
            4) 현재 LC 상태에 따라 전이 (IDLE/REQUESTED/PREPARE/EXECUTE/
               ABORT/COMPLETE/COOLDOWN)
            5) BehaviorDecision 으로 패키징

        t                : 현 tick 시각 [s]
        ego              : EgoState
        lane             : LaneModel
        lead             : LeadInfo (사용 안 함, 시그니처 호환)
        scc_state_value  : 사용 안 함 (시그니처 호환)
        set_speed_kmh    : SCC 의 set speed (IDM v0 로 사용)
        fused_objects    : confirmed FusedObject 리스트
        route_context    : RouteContext (선택, 사용 안 함)
        lead_set         : LeadInfoSet (선택, 진단용)
        predictions      : PredictionSet (선택, worst-case gap 산정에 사용)
        반환             : BehaviorDecision
        """
        # 인접 차로 평가에 사용할 미래 시점 [s]. CV horizon 1.5 s 의 중앙 부근.
        t_pred_s = config.LC_PREDICTION_LOOKAHEAD_S
        # IDM 파라미터 — ego 의 desired speed 는 SCC set speed 와 동일.
        idm_params = IDMParams(
            v0_mps=set_speed_kmh / 3.6,
            T_s=config.IDM_T_S,
            s0_m=config.IDM_S0_M,
            a_mps2=config.IDM_A_MPS2,
            b_mps2=config.IDM_B_MPS2,
            delta=config.IDM_DELTA,
        )
        mobil_params = MOBILParams(
            p=config.MOBIL_P,
            b_safe_mps2=config.MOBIL_B_SAFE_MPS2,
            a_th_mps2=config.MOBIL_A_TH_MPS2,
            a_bias_mps2=config.MOBIL_A_BIAS_MPS2,
        )

        # --- 1) 차로별 인접 차량 요약 구축 ---
        # predictions 가 있으면 _lane_neighbors 가 worst-case (현재 vs
        # t_pred_s 후) gap 으로 평가한다.
        self.last_lane_infos = self._build_lane_infos(
            ego, lane, fused_objects,
            predictions=predictions, t_pred_s=t_pred_s,
        )

        # --- 2) 인접 차로 MOBIL 평가 ---
        cur_snap = MOBILCurrentSnapshot(
            ego_speed=ego.speed_mps,
            old_lead_speed=self.last_lane_infos["CURRENT"].front_speed_mps,
            old_lead_gap=self.last_lane_infos["CURRENT"].front_gap_m,
            old_follower_speed=self.last_lane_infos["CURRENT"].rear_speed_mps,
            old_follower_gap=self.last_lane_infos["CURRENT"].rear_gap_m,
        )
        evaluations = {}
        if lane.left_lane_available and lane.left_lane_change_allowed:
            evaluations["LEFT"] = mobil_evaluate(
                "LEFT", cur_snap,
                self._snapshot_target_lane("LEFT"),
                idm_params, mobil_params,
            )
        if lane.right_lane_available and lane.right_lane_change_allowed:
            evaluations["RIGHT"] = mobil_evaluate(
                "RIGHT", cur_snap,
                self._snapshot_target_lane("RIGHT"),
                idm_params, mobil_params,
            )
        self.last_mobil_evaluations = evaluations

        # --- 3) 상태 기계 ---
        if self.lc_state == LaneChangeState.COOLDOWN:
            if t >= self.cooldown_until_t:
                self.lc_state = LaneChangeState.IDLE
                self.lc_direction = None
                self.lc_target_lane_id = None
                self.lc_safety_reason = ""
                self._reset_recommendation()

        if self.lc_state == LaneChangeState.IDLE:
            recommended = _pick_recommendation(evaluations)
            self._update_recommendation_history(t, recommended)
            if config.LOG_VERBOSE:
                self._maybe_diag(t, ego, lane, evaluations, recommended)
            if (
                recommended is not None
                and self._recommendation_persisted(t)
                and self._odd_gates_pass(ego, lane)
            ):
                self._issue_request(t, ego, lane, recommended,
                                    evaluations[recommended])

        elif self.lc_state == LaneChangeState.REQUESTED:
            ev = evaluations.get(self.lc_direction)
            if ev is not None and ev.passes:
                self.lc_state = LaneChangeState.PREPARE
                self.lc_prepare_start_t = t
                self.lc_safety_reason = ""
                if config.LOG_VERBOSE:
                    print(f"[BP] LC PREPARE  dir={self.lc_direction}")
            else:
                self._cancel_to_cooldown(
                    t, reason=(ev.safety_reason if ev else "NO_EVAL"))

        elif self.lc_state == LaneChangeState.PREPARE:
            ev = evaluations.get(self.lc_direction)
            if ev is None or not ev.passes:
                self._cancel_to_cooldown(
                    t, reason=(ev.safety_reason if ev else "NO_EVAL"))
            elif (t - self.lc_prepare_start_t) >= config.LC_PREPARE_TIME_S:
                # EXECUTE 진입 직전 후방 안전 재검증. PREPARE 동안 후방
                # 차량이 급접근했을 가능성을 prediction 으로 한 번 더 본다.
                ok, reason = self._lc_rear_revalidate(
                    ego=ego,
                    fused_objects=fused_objects,
                    predictions=predictions,
                )
                if not ok:
                    # REQUESTED 로 회귀해 hysteresis 부터 다시 쌓는다.
                    self._revert_to_requested(t, reason=reason)
                else:
                    self._enter_execute(t, ego, lane)

        elif self.lc_state == LaneChangeState.EXECUTE:
            self._update_execute(t, ego, lane)

        elif self.lc_state == LaneChangeState.ABORT:
            self._update_abort(t, ego, lane)

        elif self.lc_state == LaneChangeState.COMPLETE:
            self.lc_state = LaneChangeState.COOLDOWN
            self.cooldown_until_t = t + config.LC_COOLDOWN_AFTER_SUCCESS_S
            self._reset_recommendation()
            if config.LOG_VERBOSE:
                print(f"[BP] LC COOLDOWN until t={self.cooldown_until_t:.2f}s")

        return self._build_decision(ego, lane)

    # ==================================================================
    # 인접 차량 추출
    # ==================================================================
    def _build_lane_infos(
        self,
        ego,
        lane,
        fused_objects,
        predictions=None,
        t_pred_s=1.0,
    ):
        """CURRENT / LEFT / RIGHT 세 차로의 LaneInfo 를 일괄 산출해 dict 로 반환."""
        infos = {}
        infos["CURRENT"] = _lane_neighbors(
            "CURRENT", ego, lane, fused_objects, predictions, t_pred_s,
        )
        if lane.left_lane_available:
            infos["LEFT"] = _lane_neighbors(
                "LEFT", ego, lane, fused_objects, predictions, t_pred_s,
            )
        if lane.right_lane_available:
            infos["RIGHT"] = _lane_neighbors(
                "RIGHT", ego, lane, fused_objects, predictions, t_pred_s,
            )
        return infos

    def _snapshot_target_lane(self, position):
        """target lane (LEFT/RIGHT) 의 LaneInfo 를 MOBILLaneSnapshot 형태로 변환."""
        info = self.last_lane_infos.get(position, LaneInfo(has_front=False))
        return MOBILLaneSnapshot(
            new_lead_speed=info.front_speed_mps,
            new_lead_gap=info.front_gap_m,
            new_follower_speed=info.rear_speed_mps,
            new_follower_gap=info.rear_gap_m,
        )

    # ==================================================================
    # 권고 및 hysteresis
    # ==================================================================
    def _update_recommendation_history(self, t, recommended):
        """매 tick 권고 결과를 히스토리에 기록 — 권고가 바뀌면 타임스탬프 reset."""
        if recommended is None:
            self.first_recommendation_t = None
            self.current_recommendation = None
            return
        if self.current_recommendation != recommended:
            self.first_recommendation_t = t
            self.current_recommendation = recommended

    def _recommendation_persisted(self, t):
        """현재 권고가 LC_DECISION_HYSTERESIS_S 이상 유지됐는지 검사."""
        if self.first_recommendation_t is None:
            return False
        return (t - self.first_recommendation_t) >= config.LC_DECISION_HYSTERESIS_S

    def _reset_recommendation(self):
        """권고 히스토리를 비운다 — LC 완료 / CANCEL / COOLDOWN 종료 시 호출."""
        self.first_recommendation_t = None
        self.current_recommendation = None

    # ==================================================================
    # ODD 게이트
    # ==================================================================
    def _odd_gates_pass(self, ego, lane):
        """LC 발동 가능한 ODD (속도 범위 / 차선 유효 / junction 거리) 통과 여부."""
        if not (config.LC_MIN_SPEED_KMH <= ego.speed_kmh <= config.LC_MAX_SPEED_KMH):
            return False
        if not lane.is_valid:
            return False
        if lane.is_junction:
            return False
        if lane.distance_to_junction < config.LC_MIN_DIST_TO_JUNCTION_M:
            return False
        return True

    # ==================================================================
    # 상태 전이
    # ==================================================================
    def _issue_request(self, t, ego, lane, direction, ev):
        """IDLE → REQUESTED 전이. 디버그 콘솔에 incentive / a_n_tilde 요약 출력."""
        self.lc_state = LaneChangeState.REQUESTED
        self.lc_direction = direction
        self.lc_request_t = t
        self.lc_origin_lane_id = lane.current_lane_id
        self.lc_target_lane_id = None
        self.lc_safety_reason = ""
        print(
            f"[BP] LC REQUESTED dir={direction}  "
            f"incentive={ev.incentive_mps2:+.2f} > th={ev.threshold_mps2:+.2f}  "
            f"a_c={ev.a_c:+.2f} -> a_c'={ev.a_c_tilde:+.2f}  "
            f"a_n'={ev.a_n_tilde:+.2f} (b_safe=-{config.MOBIL_B_SAFE_MPS2:.1f})"
        )

    def _cancel_to_cooldown(self, t, reason):
        """REQUESTED / PREPARE 에서 안전성 불통과 시 COOLDOWN 으로 강제 종료."""
        prev = self.lc_state.value
        self.lc_state = LaneChangeState.COOLDOWN
        self.cooldown_until_t = t + config.LC_COOLDOWN_AFTER_ABORT_S
        self.lc_safety_reason = reason
        self._reset_recommendation()
        print(f"[BP] LC CANCELLED ({prev}): {reason}")

    # ------------------------------------------------------------------
    # PREPARE → EXECUTE 직전 후방 안전 재검증
    # ------------------------------------------------------------------
    def _lc_rear_revalidate(self, ego, fused_objects, predictions):
        """타깃 차로 후방 차량의 LC_PREEXEC_HORIZON_S 후 worst-case gap / TTC 검사.

        반환 (ok, reason):
          * ok=True  : 통과 — EXECUTE 진입 가능
          * ok=False : 차단 — 호출자가 REQUESTED 로 회귀시킴

        검증 게이트:
          1) gap_future ≥ max(LC_PREEXEC_GAP_ABS_MIN_M,
                              LC_PREEXEC_GAP_TIME_FACTOR_S · v_follower_future)
          2) rel_speed > 0.1 m/s 면 TTC_future ≥ LC_PREEXEC_TTC_MIN_S

        prediction 이 None 이면 현재 snapshot 만으로 게이트한다 (안전 측 평가
        가 조금 덜 보수적일 뿐 잘못된 결과는 아님).
        """
        if self.lc_direction is None or self.lc_target_lane_id is None and self.lc_direction not in ("LEFT", "RIGHT"):
            return True, ""

        t_pred = config.LC_PREEXEC_HORIZON_S

        info = self.last_lane_infos.get(self.lc_direction)
        if info is None or not info.has_rear:
            # 후방 차량 자체가 없으면 통과
            return True, ""

        rear_speed_future = info.rear_speed_mps  # CV 가정 — 속도 불변
        rear_gap_future = info.rear_gap_m

        # prediction 으로 worst-case gap 갱신: 가장 가까운 후방 차량의 미래
        # 위치가 현재 rear_gap 보다 더 가까운지 (= 더 작은 gap) 확인한다.
        if predictions is not None:
            for o in fused_objects:
                if o.x_ego >= -1.0 or o.x_ego <= -config.LC_EXECUTE_REAR_HORIZON_M:
                    continue
                x_pred = _predicted_x_ego(o, ego, predictions, t_pred)
                if x_pred is None or x_pred >= -1.0:
                    continue
                d_pred = -x_pred
                # 타깃 차로 lateral 분류 (LEFT / RIGHT). lane 정보가 없는
                # 자리이므로 객체의 현재 y_ego 부호로만 좁게 판단한다.
                if self.lc_direction == "LEFT" and o.y_ego >= 0.0:
                    continue
                if self.lc_direction == "RIGHT" and o.y_ego <= 0.0:
                    continue
                if d_pred < rear_gap_future:
                    rear_gap_future = d_pred
                    rear_speed_future = max(0.0, o.vx_ego)

        # ---- 게이트 ----
        gap_min = max(
            config.LC_PREEXEC_GAP_ABS_MIN_M,
            config.LC_PREEXEC_GAP_TIME_FACTOR_S * rear_speed_future,
        )
        if rear_gap_future < gap_min:
            return False, f"REAR_GAP_FUTURE_{rear_gap_future:.1f}m<{gap_min:.1f}"

        rel_speed = rear_speed_future - ego.speed_mps
        if rel_speed > 0.1:
            ttc_future = rear_gap_future / rel_speed
            if ttc_future < config.LC_PREEXEC_TTC_MIN_S:
                return False, f"REAR_TTC_FUTURE_{ttc_future:.1f}s"

        return True, ""

    def _revert_to_requested(self, t, reason):
        """PREPARE 단계 후방 안전 재검증 실패 시 REQUESTED 로 회귀.

        CANCELLED (COOLDOWN) 와 달리 MOBIL evaluation 이 다시 통과하면 새
        PREPARE 로 즉시 진입 가능하지만, hysteresis 는 다시 쌓아야 하므로
        _reset_recommendation() 을 호출한다.
        """
        prev = self.lc_state.value
        self.lc_state = LaneChangeState.REQUESTED
        self.lc_prepare_start_t = None
        self.lc_safety_reason = reason
        self._reset_recommendation()
        print(f"[BP] LC PREPARE->REQUESTED  reason={reason}  ({prev})")

    def _enter_execute(self, t, ego, lane):
        """PREPARE → EXECUTE 전이. lane.centerline 스냅샷 + dynamic lc_length 결정.

        lane_width 만큼의 횡 변위를 LC_MAX_LATERAL_ACCEL_MPS2 /
        LC_MAX_LATERAL_JERK_MPS3 한계 내에서 마칠 수 있는 최소 시간 T 를
        구하고, lc_length = T · v_ego 로 환산해 LC_MIN/MAX_LANE_CHANGE_LENGTH
        로 clamp 한다.
        """
        self.lc_state = LaneChangeState.EXECUTE
        self.lc_exec_start_t = t
        self.lc_exec_start_xy = (ego.x, ego.y)
        self.lc_reference_polyline = list(lane.centerline)
        from planning.local_planner import solve_lc_length   # 순환 import 방지를 위한 지역 import
        length, t_lc = solve_lc_length(
            lane_width_m=lane.lane_width,
            v_ego_mps=ego.speed_mps,
        )
        self.lc_total_length_m = length
        self._committed = False
        if config.LOG_VERBOSE:
            print(f"[BP] LC EXECUTE   dir={self.lc_direction} "
                  f"len={self.lc_total_length_m:.0f}m  T={t_lc:.2f}s "
                  f"(quintic ay/jerk solved)")

    # ------------------------------------------------------------------
    # LC 진행 중 후방 위협 평가 (ABORT/COMMIT 분기)
    # ------------------------------------------------------------------
    def _assess_target_rear_threat(self, ego):
        """타깃 차로 후방이 위험할 경우 사유 문자열을, 안전하면 None 을 반환.

        두 가지 트리거:
          (1) 상대 속도와 무관한 hard gap 컷오프
              (LC_EXECUTE_REAR_GAP_HARD_M, 기본 15 m)
          (2) closing TTC 가 임계 이하로 감소
              (LC_EXECUTE_REAR_TTC_ABORT_S, 기본 4 s)
        매 tick 갱신되는 self.last_lane_infos 를 사용하므로 값이 항상 최신.
        """
        if self.lc_direction is None:
            return None
        info = self.last_lane_infos.get(self.lc_direction)
        if info is None or not info.has_rear:
            return None
        if info.rear_gap_m < config.LC_EXECUTE_REAR_GAP_HARD_M:
            return f"REAR_HARD_GAP_{info.rear_gap_m:.1f}m"
        rel_speed = info.rear_speed_mps - ego.speed_mps
        if rel_speed > 0.1:
            ttc = info.rear_gap_m / rel_speed
            if ttc < config.LC_EXECUTE_REAR_TTC_ABORT_S:
                return f"REAR_TTC_{ttc:.1f}s"
        return None

    def _update_execute(self, t, ego, lane):
        """EXECUTE 상태의 한 tick — 진행도 갱신 + 후방 위협 평가 + COMPLETE 판정."""
        if self.lc_exec_start_xy is None:
            return
        dx = ego.x - self.lc_exec_start_xy[0]
        dy = ego.y - self.lc_exec_start_xy[1]
        dist_done = math.hypot(dx, dy)
        progress = min(1.0, dist_done / max(self.lc_total_length_m, 1e-3))

        # ---- LC 진행 중 후방 위협 재검사 (ABORT/COMMIT 분기) ----
        threat = self._assess_target_rear_threat(ego)
        if threat is not None:
            if progress < config.LC_EXECUTE_ABORT_PROGRESS_THRESHOLD:
                # 아직 중간 지점을 넘지 않아 ABORT 복귀가 안전하다.
                self._enter_abort(t, reason=threat, progress=progress)
                return
            elif not self._committed:
                # 중간 지점을 넘었으므로 되돌리는 것보다 진행을 강행하는 편이
                # 안전. 매 tick spam 을 막기 위해 일회성 로그만 출력.
                print(f"[BP] LC COMMIT   progress={progress:.2f} "
                      f"threat={threat}  (past half-way, accelerating through)")
                self._committed = True
            # 어느 경우든 EXECUTE 정상 진행

        # 완료 기준: 거리 progress 만 사용. LC 궤적이 ego 를 약 95 % progress
        # (사실상 타깃 차로 중심) 까지 끌고 가도록 해 lane.centerline 으로의
        # handover 가 연속적으로 이루어진다. 컨트롤러가 멈출 경우를 대비해
        # 방어적 타임아웃을 둔다.
        timed_out = (
            self.lc_exec_start_t is not None
            and (t - self.lc_exec_start_t) > 12.0
        )
        if progress >= 0.95 or timed_out:
            self.lc_state = LaneChangeState.COMPLETE
            print(f"[BP] LC COMPLETE  progress={progress:.2f}"
                  + ("  (TIMEOUT)" if timed_out else ""))

    # ------------------------------------------------------------------
    # ABORT 상태 — LC 진행 중 안전 위반 발생 시 origin 차로로 복귀
    # ------------------------------------------------------------------
    def _enter_abort(self, t, reason, progress):
        """EXECUTE → ABORT 전이. 안전 이벤트이므로 항상 로그를 남긴다."""
        self.lc_state = LaneChangeState.ABORT
        self.lc_safety_reason = reason
        print(f"[BP] LC ABORT    progress={progress:.2f} reason={reason}  "
              f"-> returning to origin lane")

    def _update_abort(self, t, ego, lane):
        """ABORT 를 완료까지 진행. local planner 가 origin 차로 polyline 으로 회귀.

        local planner 는 lc_state != EXECUTE 를 확인하고 lane.centerline 으로
        돌아간다 (ABORT 가 허용되는 초기 progress 구간에서는 이것이 origin
        차로). 그 결과 LQR 이 ego 를 origin 차로 중심으로 끌어당기며, 저장된
        origin 중심선으로부터 40 cm 이내가 되면 종료한다.
        """
        if not self.lc_reference_polyline or self.lc_exec_start_xy is None:
            # 스냅샷이 소실되었으므로 정체를 막기 위해 COMPLETE 로 fallback.
            self.lc_state = LaneChangeState.COMPLETE
            return

        from core.adas_utils import project_onto_polyline   # 순환 import 방지를 위한 지역 import
        seg_i, _, _ = project_onto_polyline(
            self.lc_reference_polyline, ego.x, ego.y,
        )
        last_idx = len(self.lc_reference_polyline) - 1
        ax, ay = self.lc_reference_polyline[seg_i]
        bx, by = self.lc_reference_polyline[min(seg_i + 1, last_idx)]
        sx = bx - ax
        sy = by - ay
        seg = math.hypot(sx, sy)
        if seg < 1e-6:
            lat = 0.0
        else:
            # 세그먼트 직선에 대한 ego 의 부호 있는 수직 거리.
            # "차로 복귀" 판정에는 크기만 필요.
            lat = abs((sx * (ego.y - ay) - sy * (ego.x - ax)) / seg)

        # 방어적 처리: ABORT 총 시간을 12 s 로 제한
        abort_timed_out = (
            self.lc_exec_start_t is not None
            and (t - self.lc_exec_start_t) > 12.0
        )
        if lat < 0.4 or abort_timed_out:
            # ABORT 는 abort 타이밍 (LC_COOLDOWN_AFTER_ABORT_S = 5 s) 으로
            # 바로 COOLDOWN 에 진입하며, COMPLETE 가 사용하는 8 s SUCCESS
            # cooldown 을 건너뛴다.
            self.lc_state = LaneChangeState.COOLDOWN
            self.cooldown_until_t = t + config.LC_COOLDOWN_AFTER_ABORT_S
            self._reset_recommendation()
            print(f"[BP] LC ABORT COMPLETE  ego back on origin lane "
                  f"(lat={lat:.2f}m)"
                  + ("  (TIMEOUT)" if abort_timed_out else ""))

    # ==================================================================
    # 진단 출력 — IDLE 상태에서 LC 가 발화되지 않는 이유를 주기적으로 dump
    # ==================================================================
    def _maybe_diag(self, t, ego, lane, evaluations, recommended):
        """LC_VERBOSE 시 _diag_period_s 주기로 LC 비발화 사유를 콘솔에 dump."""
        if (t - self._last_diag_t) < self._diag_period_s:
            return
        self._last_diag_t = t

        odd_ok = self._odd_gates_pass(ego, lane)
        odd_reasons = []
        if not (config.LC_MIN_SPEED_KMH <= ego.speed_kmh <= config.LC_MAX_SPEED_KMH):
            odd_reasons.append(f"speed={ego.speed_kmh:.0f}kmh outside "
                               f"[{config.LC_MIN_SPEED_KMH:.0f},{config.LC_MAX_SPEED_KMH:.0f}]")
        if not lane.is_valid:
            odd_reasons.append("lane invalid")
        if lane.is_junction:
            odd_reasons.append("in junction")
        if lane.distance_to_junction < config.LC_MIN_DIST_TO_JUNCTION_M:
            odd_reasons.append(f"junction in {lane.distance_to_junction:.0f}m "
                               f"(<{config.LC_MIN_DIST_TO_JUNCTION_M:.0f})")

        adj_state = []
        if lane.left_lane_available:
            adj_state.append("LEFT(avail"
                             + (",LC-ok" if lane.left_lane_change_allowed else ",no-LC")
                             + ")")
        else:
            adj_state.append("LEFT(none)")
        if lane.right_lane_available:
            adj_state.append("RIGHT(avail"
                             + (",LC-ok" if lane.right_lane_change_allowed else ",no-LC")
                             + ")")
        else:
            adj_state.append("RIGHT(none)")

        eval_lines = []
        for direction in ("LEFT", "RIGHT"):
            ev = evaluations.get(direction)
            if ev is None:
                continue
            verdict = "PASS" if ev.passes else "BLOCK"
            block_why = ""
            if not ev.passes:
                if not ev.safe:
                    block_why = f" safety: {ev.safety_reason}"
                else:
                    block_why = f" incentive {ev.incentive_mps2:+.2f} <= th {ev.threshold_mps2:+.2f}"
            eval_lines.append(
                f"  {direction:5s} {verdict}  "
                f"a_c={ev.a_c:+5.2f} a_c'={ev.a_c_tilde:+5.2f}  "
                f"a_n'={ev.a_n_tilde:+5.2f}  "
                f"incent={ev.incentive_mps2:+5.2f} (th {ev.threshold_mps2:+5.2f}){block_why}"
            )

        if recommended is None:
            hyst = "no recommendation"
        else:
            elapsed = (t - self.first_recommendation_t) if self.first_recommendation_t else 0.0
            hyst = (f"recommending {recommended}, held {elapsed:.1f}s"
                    f"/{config.LC_DECISION_HYSTERESIS_S:.1f}s")

        print(f"[BP-DIAG t={t:6.1f}s] ego={ego.speed_kmh:4.1f}km/h  "
              f"ODD={'ok' if odd_ok else 'BLOCK ' + '|'.join(odd_reasons)}  "
              f"adj={'/'.join(adj_state)}  hyst={hyst}")
        for line in eval_lines:
            print(line)

    # ==================================================================
    # 의사결정 패키징
    # ==================================================================
    def _build_decision(self, ego, lane):
        """현재 내부 LC 상태로부터 BehaviorDecision 한 개를 만들어 반환."""
        if self.lc_state in (LaneChangeState.REQUESTED, LaneChangeState.PREPARE):
            bs = BehaviorState.LANE_CHANGE_PREPARE
        elif self.lc_state == LaneChangeState.EXECUTE:
            bs = BehaviorState.LANE_CHANGE_EXECUTE
        elif self.lc_state == LaneChangeState.COMPLETE:
            bs = BehaviorState.LANE_CHANGE_COMPLETE
        elif self.lc_state == LaneChangeState.ABORT:
            bs = BehaviorState.LANE_CHANGE_ABORT
        else:
            bs = BehaviorState.LANE_KEEP

        dist_done = 0.0
        progress = 0.0
        if self.lc_state == LaneChangeState.EXECUTE and self.lc_exec_start_xy is not None:
            dx = ego.x - self.lc_exec_start_xy[0]
            dy = ego.y - self.lc_exec_start_xy[1]
            dist_done = math.hypot(dx, dy)
            progress = min(1.0, dist_done / max(self.lc_total_length_m, 1e-3))

        return BehaviorDecision(
            behavior_state=bs,
            lc_state=self.lc_state,
            lc_direction=(self.lc_direction
                          if self.lc_state != LaneChangeState.IDLE else None),
            lc_target_lane_id=self.lc_target_lane_id,
            lc_total_length_m=self.lc_total_length_m,
            lc_distance_done_m=dist_done,
            lc_progress_ratio=progress,
            lc_reference_polyline_world=(
                self.lc_reference_polyline
                if self.lc_state == LaneChangeState.EXECUTE else []
            ),
            lc_safety_reason=self.lc_safety_reason,
            # mask_lead_for_scc 는 의도적으로 FALSE 유지. LC 가 약 50 m 에
            # 걸친 점진적 횡 이동이므로 maneuver 전반부 동안 ego 는 여전히
            # 종방향으로 lead 뒤에 있다. SCC 는 lane_provider 가 ego 를 타깃
            # 차로로 재투영할 때까지 해당 lead 에 대해 GAP_CTRL 을 유지해야
            # 하며, 그 시점에 in-lane 필터에서 lead 가 자연스럽게 제거되어
            # SCC 가 스스로 SPEED_CTRL 로 전환된다.
            mask_lead_for_scc=False,
        )
