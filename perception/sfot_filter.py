"""
교체 가능한 MotionModel (CV 또는 CTRV) 을 감싸는 EKF 예측 / 갱신 본체.

상태 천이, Jacobian, 프로세스 노이즈는 모션 모델이 소유하고, 본 필터는
표준 EKF 수식을 수행한다.

    predict:  x  ← f(x, dt)                       (CTRV 의 경우 비선형)
              P  ← F_jac · P · F_jac^T + Q(x, dt)
    update :  K  ← P H^T (H P H^T + R)^-1
              x  ← x + K · (z - h(x))
              P  ← (I - K H) P
"""
import numpy as np

import perception.sfot_params as P


class Filter:
    """단일 EKF 인스턴스 — 예측 / 갱신 / innovation 계산을 모두 제공한다.

    한 인스턴스가 모든 트랙을 처리하며, motion_model 만 다르게 주면 CV ↔
    CTRV 가 그대로 교체된다.
    """

    def __init__(self, motion_model):
        self.motion = motion_model
        self.dim_state = motion_model.dim_state
        self.dt = P.DT

    def predict(self, track):
        """단일 track 의 상태를 dt 만큼 시간 전파한다.

        x ← f(x, dt), P ← F P F^T + Q(x, dt). 측정과 무관하게 매 tick
        시작에 모든 트랙에 대해 호출된다.
        """
        x = track.x
        F = self.motion.jacobian(x, self.dt)
        Q = self.motion.process_noise(x, self.dt)
        track.set_x(self.motion.predict_state(x, self.dt))
        track.set_P(F @ track.P @ F.T + Q)

    def update(self, track, meas):
        """단일 (track, meas) 쌍의 EKF 갱신을 수행한다.

        Innovation γ = z - h(x), gain K = P H^T S^-1 를 계산해 상태와 공분산
        을 보정한 뒤, 트랙의 형상 / actor_id / 마지막 센서 등 부가 속성도
        update_attributes 로 갱신한다.
        """
        H = meas.sensor.get_H(track.x)
        gamma = self.gamma(track, meas)
        S = self.S(track, meas, H)
        K = track.P @ H.T @ np.linalg.inv(S)
        I = np.eye(self.dim_state)
        track.set_x(track.x + K @ gamma)
        track.set_P((I - K @ H) @ track.P)
        track.update_attributes(meas)

    def gamma(self, track, meas):
        """Innovation γ = z - h(x) 를 반환한다."""
        return meas.z - meas.sensor.get_hx(track.x)

    def S(self, track, meas, H):
        """Innovation 공분산 S = H P H^T + R 을 반환한다."""
        return H @ track.P @ H.T + meas.R
