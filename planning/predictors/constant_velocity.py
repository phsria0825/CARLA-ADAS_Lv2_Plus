"""
다른 차량의 미래 trajectory 를 등속 가정으로 예측하는 모듈.

ego 와 거의 같은 1.0~1.5 s 시점까지의 외삽이라면 등속 가정의 직선 외삽 오차
가 동력학 비선형성보다 작아, 양산 ADAS Lv2+ 의 1 차 motion predictor 로
충분히 활용된다. 본 예측 결과는 BehaviorPlanner 의 MOBIL 평가가
worst-case (현재 vs 1 초 후) gap 을 만드는 데 주로 쓰인다.

세부 특성:
  * world frame 으로 저장한다 (lane_provider 와 BehaviorPlanner 의 lateral
    offset 계산이 모두 world frame 이라 일관성 확보).
  * Lane 곡률이 임계 (1/m 단위) 이상이면 horizon 을 1.5 s → 0.8 s 로 자동
    단축해 직선 가정 오차를 억제한다.
  * heading 은 별도 저장하지 않고 ``atan2(vy_world, vx_world)`` 로 회수
    가능하다.
"""
import numpy as np

from core.adas_types import PredictedTrack, PredictionSet


_DEFAULT_DT = 0.1                    # 예측 step [s] — 20 Hz 메인 루프 대비 2x
_DEFAULT_HORIZON_S = 1.5             # 등속 가정 신뢰 한도 [s]
_SHORT_HORIZON_S = 0.8               # 곡선 구간 자동 단축 값 [s]
_CURVATURE_SHORTEN_THRESHOLD = 0.02  # |κ| 임계 [1/m] (반경 50 m)


def _cv_rollout(obj, t_axis):
    """단일 FusedObject 의 world-frame 등속 trajectory 를 (T+1, 4) 배열로 만든다.

    FusedObject 는 ``vx_ego`` / ``vy_ego`` 만 직접 보유하므로, world frame 의
    속도는 ``speed_mps`` 와 ``yaw_world`` 로부터 합성한다.

    obj    : 등속 외삽할 FusedObject
    t_axis : shape (T+1,) 의 시간축 (0 부터 horizon 까지)
    반환   : shape (T+1, 4) 배열, 각 행이 [x_world, y_world, vx_world, vy_world]
    """
    speed = float(obj.speed_mps)
    yaw = float(obj.yaw_world)
    vx_w = speed * np.cos(yaw)
    vy_w = speed * np.sin(yaw)

    x0 = float(obj.x_world)
    y0 = float(obj.y_world)

    n = t_axis.shape[0]
    states = np.empty((n, 4), dtype=float)
    states[:, 0] = x0 + vx_w * t_axis
    states[:, 1] = y0 + vy_w * t_axis
    states[:, 2] = vx_w
    states[:, 3] = vy_w
    return states


class ConstantVelocityPredictor:
    """모든 confirmed 트랙의 미래 trajectory 를 등속 가정으로 산출하는 1 차 예측기.

    매 tick predict() 한 번 호출로 PredictionSet 을 만들며, BehaviorPlanner
    가 MOBIL 평가에서 worst-case gap 산정에 사용한다.
    """

    def __init__(self, dt=_DEFAULT_DT, horizon_s=_DEFAULT_HORIZON_S):
        if dt <= 0.0:
            raise ValueError("ConstantVelocityPredictor: dt must be > 0")
        if horizon_s < dt:
            raise ValueError(
                "ConstantVelocityPredictor: horizon_s must be >= dt")
        self.dt = float(dt)
        self.horizon_s = float(horizon_s)

    def predict(self, fused, t_now, lane=None):
        """모든 fused 객체의 등속 trajectory 를 PredictionSet 으로 반환한다.

        fused  : 현재 tick 의 confirmed FusedObject 리스트
        t_now  : ego clock 시각 [s]
        lane   : 전방 LaneModel (선택). 주어지면 곡률에 따라 horizon 을 자동
                 단축한다.
        반환   : PredictionSet (track_id → PredictedTrack 매핑)
        """
        eff_horizon = self._effective_horizon(lane)
        n_steps = max(2, int(round(eff_horizon / self.dt)) + 1)
        t_axis = np.arange(n_steps, dtype=float) * self.dt   # (T+1,)

        pset = PredictionSet(t_ref=float(t_now),
                             dt=self.dt,
                             horizon_s=eff_horizon)

        if not fused:
            return pset

        for obj in fused:
            states = _cv_rollout(obj, t_axis)
            pset.by_id[obj.track_id] = PredictedTrack(
                track_id=obj.track_id,
                confidence=float(obj.confidence),
                states=states,
                dt=self.dt,
                horizon_s=eff_horizon,
            )
        return pset

    def _effective_horizon(self, lane):
        """LaneModel 의 전방 곡률을 보고 등속 가정의 신뢰 horizon 을 결정한다.

        centerline_kappa 의 전방 약 20 m 구간 절댓값 최댓값이 임계
        (_CURVATURE_SHORTEN_THRESHOLD) 이상이면 horizon 을 _SHORT_HORIZON_S
        로 떨어뜨린다. lane 이 None / invalid / kappa 가 비어 있으면 기본
        horizon 그대로 사용한다.
        """
        if lane is None or not lane.is_valid:
            return self.horizon_s
        kappas = lane.centerline_kappa
        if not kappas:
            return self.horizon_s
        head = kappas[: min(20, len(kappas))]
        kappa_abs_max = max((abs(float(k)) for k in head), default=0.0)
        if kappa_abs_max >= _CURVATURE_SHORTEN_THRESHOLD:
            return min(self.horizon_s, _SHORT_HORIZON_S)
        return self.horizon_s
