"""
SCC / Stop-and-Go / AEB 가 사용하는 종방향 제어 유틸리티.

세 개의 PID (속도 / 간격 / 가속도 추종) + jerk-limited accel limiter 를
조합해 throttle / brake 명령을 만든다. SCCController 가 상태에 따라
compute_speed_control / compute_gap_control / reset_* 를 호출한다.
"""
import config
from core.adas_utils import clamp


class PIDController:
    """integral anti-windup 을 적용한 단순 PID 제어기.

    compute() 한 번 호출마다 (P + I + D) 합을 반환하며, 적분 항은
    integral_limit 으로 양/음 양쪽 clamp 한다.
    """

    def __init__(self, kp, ki, kd, integral_limit=20.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit

        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
        """현 error 와 dt 로부터 PID 출력을 산출한다.

        error : 비례 입력
        dt    : 시간 간격 [s] (0 또는 음수면 미분 항 0)
        반환  : kp·e + ki·∫e + kd·de/dt
        """
        self.integral += error * dt
        self.integral = clamp(self.integral, -self.integral_limit, self.integral_limit)

        derivative = (error - self.prev_error) / dt if dt > 0.0 else 0.0
        self.prev_error = error

        return self.kp * error + self.ki * self.integral + self.kd * derivative

    def reset(self):
        """적분 / 이전 error 를 0 으로 초기화한다 (상태 전이 시 호출)."""
        self.integral = 0.0
        self.prev_error = 0.0


class AccelLimiter:
    """원하는 가속도 명령을 MAX_ACCEL/DECEL 과 jerk 한계로 포화시킨다.

    매 tick limit() 호출이 직전 명령과의 차이를 MAX_JERK·dt 이내로 잘라 줘서
    승차감을 보호한다.
    """

    def __init__(self):
        self.prev_accel_cmd = 0.0

    def limit(self, desired_accel, dt):
        """desired_accel 을 가속도 / jerk 한계로 잘라 반환하고 내부 상태 갱신.

        desired_accel : 원하는 가속도 [m/s²]
        dt            : 시간 간격 [s]
        반환          : 포화된 명령 가속도 [m/s²]
        """
        accel = clamp(desired_accel, config.MAX_DECEL, config.MAX_ACCEL)

        max_delta = config.MAX_JERK * dt
        delta = clamp(accel - self.prev_accel_cmd, -max_delta, max_delta)

        accel = self.prev_accel_cmd + delta
        self.prev_accel_cmd = accel
        return accel

    def reset(self):
        """이전 명령 가속도를 0 으로 초기화 (상태 전이 시 호출)."""
        self.prev_accel_cmd = 0.0


class LongitudinalController:
    """속도 / 간격 → 목표 가속도 → throttle / brake 의 2-loop 종방향 제어기.

    상위 루프는 속도 또는 간격 오차로부터 desired_accel 을 만들고, 하위
    루프 (accel_pid + accel_limiter) 가 실제 가속도와의 오차를 보정해 jerk
    포화된 throttle / brake 명령으로 변환한다. SCCController 가 상태별로
    compute_speed_control / compute_gap_control 를 호출한다.
    """

    def __init__(self):
        self.speed_pid = PIDController(
            config.SPEED_KP, config.SPEED_KI, config.SPEED_KD
        )
        self.gap_pid = PIDController(
            config.GAP_KP, config.GAP_KI, config.GAP_KD
        )
        self.accel_pid = PIDController(
            config.ACCEL_KP, config.ACCEL_KI, config.ACCEL_KD
        )
        self.accel_limiter = AccelLimiter()

        self.last_desired_accel = 0.0
        self.last_limited_accel = 0.0
        self.last_actual_accel = 0.0
        self.last_accel_error = 0.0

    def compute_speed_control(self, speed_error_kmh, actual_accel, dt):
        """SPEED_CTRL 모드 — 속도 오차로부터 throttle / brake 산출.

        speed_error_kmh : set_speed - 현재 속도 [km/h]
        actual_accel    : 현재 측정 가속도 [m/s²]
        dt              : 시간 간격 [s]
        반환            : (throttle ∈ [0,1], brake ∈ [0,1])
        """
        desired_accel = self.speed_pid.compute(speed_error_kmh, dt)
        return self._accel_to_command(desired_accel, actual_accel, dt)

    def compute_gap_control(self, distance_error, relative_speed, actual_accel, dt):
        """GAP_CTRL 모드 — gap 오차 + 상대 속도로부터 throttle / brake 산출.

        distance_error : 원하는 gap - 현재 gap [m]. 양수면 너무 멀어서 가속 필요.
        relative_speed : 자차 대비 lead 상대 속도 [m/s]. relative_speed * 0.5 가
                         derivative 같은 효과로 한 단계 더해진다.
        actual_accel   : 현재 측정 가속도 [m/s²]
        dt             : 시간 간격 [s]
        반환           : (throttle ∈ [0,1], brake ∈ [0,1])
        """
        combined_error = distance_error + relative_speed * 0.5
        desired_accel = self.gap_pid.compute(combined_error, dt)
        return self._accel_to_command(desired_accel, actual_accel, dt)

    def _accel_to_command(self, desired_accel, actual_accel, dt):
        """desired_accel 을 jerk 포화 → accel_pid 보정 → throttle/brake 변환."""
        limited_accel = self.accel_limiter.limit(desired_accel, dt)

        accel_error = limited_accel - actual_accel
        accel_correction = self.accel_pid.compute(accel_error, dt)
        total_accel = limited_accel + accel_correction

        if total_accel >= 0.0:
            throttle = clamp(total_accel / config.MAX_ACCEL, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = clamp(abs(total_accel) / abs(config.MAX_DECEL), 0.0, 1.0)

        self.last_desired_accel = desired_accel
        self.last_limited_accel = limited_accel
        self.last_actual_accel = actual_accel
        self.last_accel_error = accel_error

        return throttle, brake

    def reset_gap_pid(self):
        """GAP_CTRL 진입 시 호출 — gap PID + accel PID + limiter 모두 초기화."""
        self.gap_pid.reset()
        self.accel_pid.reset()
        self.accel_limiter.reset()

    def reset_speed_pid(self):
        """SPEED_CTRL 진입 시 호출 — speed PID + accel PID + limiter 초기화."""
        self.speed_pid.reset()
        self.accel_pid.reset()
        self.accel_limiter.reset()

    def reset_all(self):
        """AEB 진입 등 전체 reset 이 필요한 경우 호출 — 모든 PID + limiter + 디버그 값 초기화."""
        self.speed_pid.reset()
        self.gap_pid.reset()
        self.accel_pid.reset()
        self.accel_limiter.reset()
        self.last_desired_accel = 0.0
        self.last_limited_accel = 0.0
        self.last_actual_accel = 0.0
        self.last_accel_error = 0.0
