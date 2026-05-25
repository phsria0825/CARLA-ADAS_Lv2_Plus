"""
SFOT 의 센서 + 측정 클래스 — 모션 모델에 비의존적.

두 개의 센서가 존재한다.
  LiDAR  : 센서 좌표계의 3D 위치 [x, y, z]^T, ±90° 전방 반구 FOV.
  Camera : 핀홀 투영을 통해 산출된 2D 픽셀 [u, v]^T.

측정 Jacobian 은 motion_model.position_indices 로 지정된 상태 컬럼에 행을
기록하므로, CV (위치 인덱스 [0, 1, 2]) 와 CTRV (위치 인덱스 [0, 1, 5]) 간
전환 시 본 파일에서의 변경은 필요하지 않다.
"""
import math

import numpy as np

import perception.sfot_params as P


class Sensor:
    """단일 센서 (LiDAR 또는 카메라) 의 측정 모델 + pose.

    EKF Filter 가 update() 에서 호출하는 in_fov / get_hx / get_H 를 제공
    한다. sens_to_world 는 매 tick 갱신되어 ego 가 움직여도 정확한 측정
    좌표 변환을 보장한다.
    """

    def __init__(
        self,
        name,
        motion_model,
        sens_to_world=None,
        intrinsic=None,
    ):
        self.name = name
        self.motion = motion_model
        self.intrinsic = intrinsic or {}
        if name == 'lidar':
            self.dim_meas = 3
            self.fov = (-math.pi / 2.0, math.pi / 2.0)
        elif name == 'camera':
            self.dim_meas = 2
            self.fov = None
        else:
            raise ValueError(f'Unsupported sensor: {name}')
        if sens_to_world is None:
            sens_to_world = np.eye(4, dtype=float)
        self.update_pose(sens_to_world)

    def update_pose(self, sens_to_world):
        """센서 → 월드 변환 행렬을 갱신하고 그 역행렬도 캐시한다.

        매 tick FusionManager 가 ego transform 의 변화를 반영해 호출한다.
        """
        self.sens_to_world = np.asarray(sens_to_world, dtype=float)
        self.world_to_sens = np.linalg.inv(self.sens_to_world)

    def _world_to_sensor_xyz(self, x_state):
        """상태 x 에서 (px, py, pz) 를 뽑아 센서 좌표계 4-벡터로 변환."""
        px, py, pz = self.motion.extract_position(x_state)
        pos_world = np.array([[px], [py], [pz], [1.0]], dtype=float)
        return self.world_to_sens @ pos_world

    def in_fov(self, x_state):
        """상태 x 의 위치가 본 센서의 FOV 안에 있는지 판정한다.

        LiDAR : sensor x 가 0.1 m 초과 + bearing 이 ±90° 안.
        Camera: sensor x 가 0.1 m 초과 + bearing 이 ±(intrinsic FOV / 2) 안.

        x_state : 트랙의 현재 상태 벡터
        반환    : True / False
        """
        pos_s = self._world_to_sensor_xyz(x_state)
        sx = float(pos_s[0, 0]); sy = float(pos_s[1, 0])
        if sx <= 0.1:
            return False
        if self.name == 'lidar':
            alpha = math.atan2(sy, sx)
            return self.fov[0] <= alpha <= self.fov[1]
        half_fov = math.radians(self.intrinsic['fov_deg']) / 2.0
        alpha = math.atan2(sy, sx)
        return -half_fov <= alpha <= half_fov

    def get_hx(self, x_state):
        """예측 측정값 h(x) 를 산출한다.

        LiDAR : 센서 좌표계의 3D 위치 [sx, sy, sz]^T.
        Camera: 핀홀 모델로 투영한 픽셀 [u, v]^T.

        x_state : 트랙의 현재 상태 벡터
        반환    : LiDAR 면 (3, 1), Camera 면 (2, 1) 측정 벡터
        """
        pos_s = self._world_to_sensor_xyz(x_state)
        if self.name == 'lidar':
            return pos_s[0:3]
        sx = float(pos_s[0, 0]); sy = float(pos_s[1, 0]); sz = float(pos_s[2, 0])
        if sx <= 1e-6:
            raise ValueError('Camera projection undefined for sx <= 0')
        fx = self.intrinsic['fx']; fy = self.intrinsic['fy']
        cx = self.intrinsic['cx']; cy = self.intrinsic['cy']
        u = cx + fx * sy / sx
        v = cy - fy * sz / sx
        return np.array([[u], [v]], dtype=float)

    def get_H(self, x_state):
        """측정 Jacobian H = ∂h/∂x 를 산출한다.

        LiDAR : 회전 행렬 R 의 3 행을 motion_model.position_indices 가
                가리키는 컬럼에 그대로 채운다.
        Camera: 핀홀 모델의 해석적 Jacobian (이미지 du, dv 두 행).

        x_state : 트랙의 현재 상태 벡터
        반환    : LiDAR 면 (3, dim_state), Camera 면 (2, dim_state) Jacobian
        """
        dim_state = self.motion.dim_state
        pos_idx = self.motion.position_indices
        H = np.zeros((self.dim_meas, dim_state), dtype=float)
        R = self.world_to_sens[0:3, 0:3]
        t = self.world_to_sens[0:3, 3:4]

        if self.name == 'lidar':
            # dh/d[px, py, pz] = R 의 3×3 블록
            for r in range(3):
                for c, col in enumerate(pos_idx):
                    H[r, col] = R[r, c]
            return H

        # 카메라 핀홀 모델의 해석적 Jacobian (이미지 평면 du, dv 두 행)
        px, py, pz = self.motion.extract_position(x_state)
        p = np.array([[px], [py], [pz]], dtype=float)
        row_x = R[0:1, :]; row_y = R[1:2, :]; row_z = R[2:3, :]
        sx = float((row_x @ p)[0, 0] + t[0, 0])
        sy = float((row_y @ p)[0, 0] + t[1, 0])
        sz = float((row_z @ p)[0, 0] + t[2, 0])
        if sx <= 1e-6:
            raise ValueError('Camera Jacobian undefined for sx <= 0')
        fx = self.intrinsic['fx']; fy = self.intrinsic['fy']
        du_dp = fx * ((sx * row_y) - (sy * row_x)) / (sx ** 2)
        dv_dp = -fy * ((sx * row_z) - (sz * row_x)) / (sx ** 2)
        for c, col in enumerate(pos_idx):
            H[0, col] = du_dp[0, c]
            H[1, col] = dv_dp[0, c]
        return H

    def generate_measurement(self, num_frame, z, meas_list, **kwargs):
        """원시 검출 z 를 Measurement 객체로 wrap 해 meas_list 에 append.

        FusionManager 의 _build_lidar_measurements / _build_camera_measurements
        가 호출한다.

        num_frame : tick 인덱스 (Measurement.t = num_frame · DT 로 변환)
        z         : 원시 측정 ([x, y, z] 또는 [u, v])
        meas_list : append 할 리스트
        kwargs    : actor_id / width / length / height / yaw / point_count 등
        반환      : append 가 끝난 meas_list (in-place 갱신과 동일)
        """
        meas = Measurement(num_frame, z, self, **kwargs)
        meas_list.append(meas)
        return meas_list


class Measurement:
    """단일 tick × 단일 센서의 측정 한 건.

    z (측정 벡터), R (측정 노이즈 공분산), 그리고 LiDAR 의 경우 형상 정보
    (width / length / height / yaw / point_count) 를 함께 들고 있다. EKF 의
    update() 는 z 와 R 만 사용하지만, Trackmanagement 는 형상 정보를 트랙에
    blending 한다.
    """

    def __init__(self, num_frame, z, sensor, **kwargs):
        self.t = num_frame * P.DT
        self.sensor = sensor
        self.actor_id = kwargs.get('actor_id')
        self.point_count = int(kwargs.get('point_count', 0))

        if sensor.name == 'lidar':
            self.z = np.array([[z[0]], [z[1]], [z[2]]], dtype=float)
            self.R = np.diag([
                P.SIGMA_LIDAR_X ** 2,
                P.SIGMA_LIDAR_Y ** 2,
                P.SIGMA_LIDAR_Z ** 2,
            ])
        elif sensor.name == 'camera':
            self.z = np.array([[z[0]], [z[1]]], dtype=float)
            self.R = np.diag([
                P.SIGMA_CAM_PIXEL_U ** 2,
                P.SIGMA_CAM_PIXEL_V ** 2,
            ])
        else:
            raise ValueError(f'Unsupported sensor: {sensor.name}')

        self.height = float(kwargs.get('height', 1.5))
        self.width = float(kwargs.get('width', 1.8))
        self.length = float(kwargs.get('length', 4.5))
        self.yaw = float(kwargs.get('yaw', 0.0))
