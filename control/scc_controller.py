"""
AEB 와 Stop-and-Go 를 함께 지원하는 Smart Cruise Control 4-mode 상태 기계.

상태 전이:
    SPEED_CONTROL ↔ GAP_CONTROL ↔ STOP_AND_GO
    위의 어느 상태에서나 즉시 AEB 로 진입 가능 (긴급 제동 우선).

매 tick update() 한 번 호출로 상태 갱신 + 해당 모드의 throttle / brake 산출
까지 한꺼번에 수행한다.
"""
from enum import Enum

import config
from control.longitudinal_controller import LongitudinalController


class SCCState(Enum):
    """SCC 4-mode 상태머신의 enum."""

    SPEED_CONTROL = 'SPEED_CTRL'
    GAP_CONTROL = 'GAP_CTRL'
    STOP_AND_GO = 'STOP_GO'
    AEB = 'AEB'


class SCCController:
    """SCC + Stop-and-Go + AEB 통합 종방향 제어기.

    내부에 LongitudinalController 한 개를 두고 상태별로 그것을 호출한다.
    update() 가 유일한 진입점이며, lead 존재 여부 / 속도 / TTC 에 따라 매
    tick 자동으로 상태 전이한다.
    """

    # Stop-and-Go hold 해제 직후 "재latch 금지" 강제 구간 [s].
    # 한 tick 짜리 gap 변화로 latch bit 가 즉시 다시 켜지는 진동 방지용.
    SNG_RELATCH_COOLDOWN_S = 0.5

    def __init__(self):
        self.state = SCCState.SPEED_CONTROL
        self.long_ctrl = LongitudinalController()
        self.set_speed = config.SET_SPEED

        self._lead_lost_count = 0
        self._sng_hold_latched = False
        self._sng_relatch_cooldown_s = 0.0

        self.desired_gap = 0.0
        self.gap_error = 0.0
        self.last_ttc = float('inf')

    def update(self, speed_kmh, speed_ms, actual_accel, lead_info, dt, desired_speed_kmh=None):
        """한 tick 의 상태 전이 + 해당 모드의 throttle / brake 산출.

        speed_kmh        : 자차 속도 [km/h]
        speed_ms         : 자차 속도 [m/s]
        actual_accel     : 측정 종방향 가속도 [m/s²]
        lead_info        : 현재 차선 LeadInfo
        dt               : 시간 간격 [s]
        desired_speed_kmh: LocalPlanner 의 목표 속도 (선택). 주어지면 set_speed 갱신.
        반환             : (throttle ∈ [0,1], brake ∈ [0,1])
        """
        if desired_speed_kmh is not None:
            self.set_cruise_speed(desired_speed_kmh, announce=False)
        self.last_ttc = self._compute_ttc(lead_info)
        self._update_state(lead_info, speed_kmh)

        if self.state == SCCState.AEB:
            return self._aeb_control(speed_kmh, speed_ms, lead_info)
        if self.state == SCCState.STOP_AND_GO:
            return self._stop_and_go_control(speed_kmh, speed_ms, actual_accel, lead_info, dt)
        if self.state == SCCState.GAP_CONTROL:
            return self._gap_control(speed_kmh, speed_ms, actual_accel, lead_info, dt)

        return self._speed_control(speed_kmh, actual_accel, dt)

    def _set_state(self, new_state):
        """상태 전이 + 관련 PID / latch 초기화 + 진입 로그.

        AEB 진입은 안전 임계 이벤트이므로 항상 출력. 그 외 전이는 verbose
        플래그 뒤로 숨겨 콘솔 spam 을 막는다.
        """
        if new_state == self.state:
            return

        prev_state = self.state
        self.state = new_state
        self._lead_lost_count = 0
        if new_state != SCCState.STOP_AND_GO:
            self._sng_hold_latched = False
            self._sng_relatch_cooldown_s = 0.0

        if new_state == SCCState.SPEED_CONTROL:
            self.long_ctrl.reset_speed_pid()
        elif new_state in (SCCState.GAP_CONTROL, SCCState.STOP_AND_GO):
            self.long_ctrl.reset_gap_pid()
        else:
            self.long_ctrl.reset_all()

        if new_state == SCCState.AEB or getattr(config, 'LOG_VERBOSE', False):
            print(f'[SCC] State transition: {prev_state.value} -> {self.state.value}')

    def _update_state(self, lead_info, speed_kmh):
        """다음 상태를 결정한다 — AEB 진입 우선, 그 다음 lead 존재 / 거리로 분기."""
        # 1) AEB 진입 조건 우선 검사
        if self._should_enter_aeb(lead_info, speed_kmh):
            self._set_state(SCCState.AEB)
            return

        # 2) AEB 에서 빠져나오기
        if self.state == SCCState.AEB:
            if self._should_hold_aeb(lead_info, speed_kmh):
                return
            self._set_state(self._select_follow_state(lead_info, speed_kmh, allow_speed_fallback=True))
            return

        # 3) SPEED → GAP 전환
        if self.state == SCCState.SPEED_CONTROL:
            if lead_info.detected and lead_info.range < config.LEAD_DETECT_DIST:
                self._set_state(self._select_follow_state(lead_info, speed_kmh))
            return

        # 4) lead 가 가까이 있으면 GAP / SnG 유지
        if lead_info.detected and lead_info.range < config.LEAD_LOST_DIST:
            self._lead_lost_count = 0
            self._set_state(self._select_follow_state(lead_info, speed_kmh))
            return

        # 5) lead 분실 카운트 후 SPEED 복귀
        self._lead_lost_count += 1
        if self._lead_lost_count >= config.LEAD_LOST_COUNT:
            self._set_state(SCCState.SPEED_CONTROL)

    def _select_follow_state(self, lead_info, speed_kmh, allow_speed_fallback=False):
        """저속·짧은 gap 이면 STOP_AND_GO, 아니면 GAP_CONTROL 을 선택해 반환.

        현재 STOP_AND_GO 상태에서 빠져나올 때는 hysteresis (EXIT > ENGAGE)
        를 적용해 진동을 막는다.
        """
        if not lead_info.detected:
            return SCCState.SPEED_CONTROL if allow_speed_fallback else self.state

        if self.state == SCCState.STOP_AND_GO:
            # 두 조건이 모두 충족될 때만 SnG 유지 (AND).
            if (
                speed_kmh < config.SNG_EXIT_SPEED_KMH and
                lead_info.range < config.SNG_EXIT_DIST
            ):
                return SCCState.STOP_AND_GO

        if (
            speed_kmh <= config.SNG_ENGAGE_SPEED_KMH and
            lead_info.range <= config.SNG_ENGAGE_DIST
        ):
            return SCCState.STOP_AND_GO

        return SCCState.GAP_CONTROL

    def _compute_closing_speed(self, lead_info):
        """lead 에 접근하는 속도 (≥ 0). range_rate 가 음수일 때만 양수, 아니면 0."""
        if not lead_info.detected:
            return 0.0
        return max(0.0, -float(lead_info.range_rate))

    def _compute_ttc(self, lead_info):
        """TTC = range / closing_speed. 접근 속도가 임계 미만이면 inf."""
        closing_speed = self._compute_closing_speed(lead_info)
        if closing_speed < config.AEB_MIN_CLOSING_SPEED:
            return float('inf')
        return float(lead_info.range) / max(closing_speed, 1e-3)

    def _should_enter_aeb(self, lead_info, speed_kmh):
        """AEB 진입 조건 — TTC ≤ AEB_TTC_ENTER 또는 접근 중 + close-range."""
        if not lead_info.detected:
            return False

        closing_speed = self._compute_closing_speed(lead_info)
        too_close = (
            lead_info.range <= config.AEB_MIN_RANGE and
            speed_kmh > config.AEB_HOLD_SPEED_KMH and
            closing_speed > 0.3
        )
        return too_close or self.last_ttc <= config.AEB_TTC_ENTER

    def _should_hold_aeb(self, lead_info, speed_kmh):
        """AEB 유지 조건 — 정지에 가까우면서 접근 정지하면 해제, 아니면 유지."""
        if not lead_info.detected:
            return False

        closing_speed = self._compute_closing_speed(lead_info)
        if speed_kmh <= config.AEB_HOLD_SPEED_KMH and closing_speed < 0.3:
            return False

        if lead_info.range <= config.AEB_RELEASE_RANGE and closing_speed > 0.3:
            return True

        return self.last_ttc <= config.AEB_TTC_EXIT

    def _update_gap_metrics(self, speed_ms, lead_info):
        """desired_gap = TIME_GAP·v + STANDSTILL_DIST 와 gap_error 를 갱신."""
        self.desired_gap = config.TIME_GAP * speed_ms + config.STANDSTILL_DIST
        if lead_info.detected:
            self.gap_error = lead_info.range - self.desired_gap
        else:
            self.gap_error = 0.0

    def _apply_follow_speed_governor(self, speed_kmh, throttle, brake, max_throttle):
        """추종 중 set_speed 를 넘지 않도록 throttle taper + overspeed brake 부여.

        speed_headroom 이 0 에 가까워질수록 throttle 을 점진 축소하고,
        set_speed 를 넘으면 비례 brake 를 추가한다.
        """
        throttle = min(throttle, max_throttle)

        speed_headroom = self.set_speed - speed_kmh
        taper_window = max(config.FOLLOW_SPEED_TAPER_KMH, 1e-3)
        throttle_scale = max(0.0, min(1.0, speed_headroom / taper_window))
        throttle *= throttle_scale

        overspeed = max(0.0, speed_kmh - self.set_speed)
        if overspeed > 0.0:
            overspeed_ratio = overspeed / max(config.FOLLOW_OVERSPEED_BRAKE_KMH, 1e-3)
            governor_brake = min(
                config.FOLLOW_OVERSPEED_BRAKE_MAX,
                overspeed_ratio * config.FOLLOW_OVERSPEED_BRAKE_MAX,
            )
            brake = max(brake, governor_brake)

        return throttle, brake

    def _should_latch_sng_hold(self, speed_kmh, lead_info):
        """SnG hold latch 진입 조건 — 매우 느리면서 lead 가 desired_gap 안에 있을 때."""
        if not lead_info.detected:
            return False

        hold_limit = self.desired_gap + config.SNG_HOLD_MARGIN_M
        return (
            speed_kmh <= config.SNG_HOLD_ENTRY_SPEED_KMH and
            lead_info.range <= hold_limit
        )

    def _should_release_sng_hold(self, lead_info):
        """SnG hold latch 해제 조건 — lead 가 충분히 멀어졌거나 빠르게 떠나면."""
        if not lead_info.detected:
            return False

        gap_opened = lead_info.range >= self.desired_gap + config.SNG_HOLD_RELEASE_GAP_M
        lead_pulling_away = (
            lead_info.range_rate >= config.SNG_HOLD_RELEASE_RATE_MPS and
            lead_info.range >= self.desired_gap + 0.5 * config.SNG_HOLD_RELEASE_GAP_M
        )
        return gap_opened or lead_pulling_away

    def _speed_control(self, speed_kmh, actual_accel, dt):
        """SPEED_CTRL 모드 — set_speed 추종으로 throttle / brake 산출."""
        self.desired_gap = 0.0
        self.gap_error = 0.0
        speed_error = self.set_speed - speed_kmh
        return self.long_ctrl.compute_speed_control(speed_error, actual_accel, dt)

    def _gap_control(self, speed_kmh, speed_ms, actual_accel, lead_info, dt):
        """GAP_CTRL 모드 — desired_gap 추종 + follow speed governor 적용."""
        if not lead_info.detected:
            return self._speed_control(speed_kmh, actual_accel, dt)

        self._update_gap_metrics(speed_ms, lead_info)
        gap_throttle, gap_brake = self.long_ctrl.compute_gap_control(
            self.gap_error,
            lead_info.range_rate,
            actual_accel,
            dt,
        )
        gap_throttle, gap_brake = self._apply_follow_speed_governor(
            speed_kmh,
            gap_throttle,
            gap_brake,
            config.GAP_MAX_THROTTLE,
        )

        return gap_throttle, gap_brake

    def _stop_and_go_control(self, speed_kmh, speed_ms, actual_accel, lead_info, dt):
        """STOP_AND_GO 모드 — 저속 추종 + 정지 유지 latch + 해제 cooldown."""
        # SnG 에 머무는 동안 매 tick 해제 cooldown 을 감소시킨다.
        if self._sng_relatch_cooldown_s > 0.0:
            self._sng_relatch_cooldown_s = max(0.0, self._sng_relatch_cooldown_s - dt)

        if not lead_info.detected:
            if self._sng_hold_latched:
                return 0.0, config.SNG_HOLD_BRAKE
            return self._speed_control(speed_kmh, actual_accel, dt)

        self._update_gap_metrics(speed_ms, lead_info)

        if self._sng_hold_latched:
            if self._should_release_sng_hold(lead_info):
                # 해제 시 PID 상태 초기화 + 재latch cooldown 활성화 (한 tick
                # 노이즈 gap 으로 즉시 재latch 되는 것 방지).
                self._sng_hold_latched = False
                self.long_ctrl.reset_gap_pid()
                self._sng_relatch_cooldown_s = self.SNG_RELATCH_COOLDOWN_S
            else:
                return 0.0, config.SNG_HOLD_BRAKE

        throttle, brake = self.long_ctrl.compute_gap_control(
            self.gap_error,
            lead_info.range_rate,
            actual_accel,
            dt,
        )

        throttle = min(throttle, config.SNG_MAX_THROTTLE)
        brake = min(brake, config.SNG_MAX_BRAKE)
        throttle, brake = self._apply_follow_speed_governor(
            speed_kmh,
            throttle,
            brake,
            config.SNG_MAX_THROTTLE,
        )

        # cooldown 활성 중에는 재latch 차단.
        if (
            self._sng_relatch_cooldown_s <= 0.0
            and self._should_latch_sng_hold(speed_kmh, lead_info)
        ):
            self._sng_hold_latched = True
            throttle = 0.0
            brake = max(brake, config.SNG_HOLD_BRAKE)

        return throttle, brake

    def _aeb_control(self, speed_kmh, speed_ms, lead_info):
        """AEB 모드 — TTC / range severity 비례 brake 산출 + 정지 시 hold brake."""
        self._update_gap_metrics(speed_ms, lead_info)

        if not lead_info.detected:
            return 0.0, config.AEB_HOLD_BRAKE

        severity = 0.0
        if self.last_ttc < float('inf'):
            ttc_span = max(config.AEB_TTC_ENTER - config.AEB_FULL_BRAKE_TTC, 1e-3)
            severity = max(severity, (config.AEB_TTC_ENTER - self.last_ttc) / ttc_span)

        gap_span = max(config.AEB_MIN_RANGE, 1e-3)
        severity = max(severity, (config.AEB_MIN_RANGE - lead_info.range) / gap_span)
        severity = max(0.0, min(1.0, severity))

        brake = config.AEB_BRAKE_MIN + severity * (config.AEB_BRAKE_MAX - config.AEB_BRAKE_MIN)

        if (
            speed_kmh <= config.AEB_HOLD_SPEED_KMH and
            lead_info.range <= config.AEB_RELEASE_RANGE
        ):
            brake = max(brake, config.AEB_HOLD_BRAKE)

        brake = max(config.AEB_BRAKE_MIN, min(config.AEB_BRAKE_MAX, brake))
        return 0.0, brake

    def set_cruise_speed(self, speed_kmh, announce=True):
        """set_speed 를 갱신한다. announce=True 면 30~200 km/h 로 clamp + 로그 출력.

        announce=False (LocalPlanner 가 매 tick 자동 호출하는 경우) 는
        0~200 km/h 범위 + 로그 없음.
        """
        min_speed = 30.0 if announce else 0.0
        self.set_speed = max(min_speed, min(200.0, speed_kmh))
        if announce:
            print(f'[SCC] Set speed changed: {self.set_speed:.0f} km/h')

    def get_status_string(self):
        """현재 상태 + desired_gap / gap_error / TTC / HOLD 플래그를 한 줄 문자열로.

        BEV sidebar 와 콘솔 status 라인이 소비한다.
        """
        status = f'[{self.state.value:10s}]'
        if self.state in (SCCState.GAP_CONTROL, SCCState.STOP_AND_GO, SCCState.AEB):
            status += f' d*={self.desired_gap:.1f}m | err={self.gap_error:+.1f}m'
        if self.state == SCCState.AEB and self.last_ttc < float('inf'):
            status += f' | TTC={self.last_ttc:.2f}s'
        if self.state == SCCState.STOP_AND_GO and self._sng_hold_latched:
            status += ' | HOLD'
        return status
