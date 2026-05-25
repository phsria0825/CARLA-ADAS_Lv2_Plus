"""
매 tick 마다 CARLA 차량 actor 로부터 EgoState 를 구성한다.

CARLA 의 transform.rotation.yaw 는 도(degree) 단위이며 CARLA 고유의 왼손
좌표계를 사용한다. 본 모듈에서 일괄적으로 라디안으로 변환해 모든 하류
소비자가 좌표계 일관성을 유지할 수 있도록 한다.

종방향 가속도는 CARLA 가 get_acceleration() 으로 world frame 으로 보고하는
값을 차량 전방축에 투영해 스칼라 a_lon 으로 만든다. world frame 가속도가
유한하지 않을 때를 대비해 속도 기반 유한차분 fallback 을 함께 유지해 두
값을 가중 평균한다.
"""
import math

from core.adas_types import EgoState
from core.adas_utils import deg2rad


class EgoStateProvider:
    """CARLA vehicle actor → EgoState 변환을 매 tick 수행하는 어댑터.

    이전 tick 의 speed / timestamp 를 들고 있다가 유한차분 가속도 fallback
    을 계산하므로, instance 를 매 tick 새로 만들지 말고 한 번 만들어 재사용
    해야 한다.
    """

    def __init__(self, vehicle):
        self.vehicle = vehicle
        self._prev_speed_mps = None
        self._prev_timestamp = None
        self._last_state = None

    def update(self, timestamp):
        """CARLA actor 의 transform / velocity / acceleration / yaw rate 를 읽어 EgoState 생성.

        도→라디안 변환, world → body frame 속도 투영, world 가속도 + 유한
        차분 fallback 의 70:30 가중 평균을 한 번에 수행해 일관된 단위계의
        EgoState 한 개를 발행한다.

        timestamp : 현 tick 시각 [s] (메인 루프의 sim_time)
        반환      : EgoState
        """
        tf = self.vehicle.get_transform()
        vel = self.vehicle.get_velocity()
        ang = self.vehicle.get_angular_velocity()
        accel_world = self.vehicle.get_acceleration()

        yaw_rad = deg2rad(tf.rotation.yaw)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)

        # world frame 속도를 차량 body 축으로 투영. body x 는 전방 (후진 시
        # 음수), body y 는 CARLA 의 왼손 좌표계에서 우측이 양수.
        vx_body = vel.x * cos_y + vel.y * sin_y
        vy_body = -vel.x * sin_y + vel.y * cos_y
        speed_mps = abs(vx_body)

        # world frame 가속도를 전방축에 투영해 종방향 가속도 산출.
        a_lon_world = accel_world.x * cos_y + accel_world.y * sin_y

        # 유한차분 fallback (이전 tick 의 vx_body 와 비교).
        a_lon_fd = 0.0
        if self._prev_speed_mps is not None and self._prev_timestamp is not None:
            dt = timestamp - self._prev_timestamp
            if dt > 1e-4:
                a_lon_fd = (vx_body - self._prev_speed_mps) / dt

        # world frame 가속도가 유한하면 우선 사용, 아니면 유한차분 사용.
        if math.isfinite(a_lon_world):
            a_lon = 0.7 * a_lon_world + 0.3 * a_lon_fd
        else:
            a_lon = a_lon_fd

        self._prev_speed_mps = vx_body
        self._prev_timestamp = timestamp

        state = EgoState(
            timestamp=timestamp,
            x=tf.location.x,
            y=tf.location.y,
            z=tf.location.z,
            yaw_rad=yaw_rad,
            speed_mps=speed_mps,
            speed_kmh=speed_mps * 3.6,
            vx_world=vel.x,
            vy_world=vel.y,
            vx_body=vx_body,
            vy_body=vy_body,
            yaw_rate_rad_s=deg2rad(ang.z),
            accel_mps2=a_lon,
            is_valid=True,
        )
        self._last_state = state
        return state

    @property
    def last(self):
        """가장 최근 update() 에서 발행한 EgoState. update 호출 전이면 None."""
        return self._last_state
