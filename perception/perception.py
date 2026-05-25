"""
LiDAR + Camera pseudo perception — CARLA actor truth 를 두 센서 모델로 변환.

아키텍처: ``world.get_actors().filter('vehicle.*')`` 로 actor truth 를 읽어와
각 센서에 대해 다음을 수행한다.

  - LiDAR : actor centroid 를 LiDAR local frame 으로 투영하고, FOV / range
            로 필터링한 뒤 바운딩 박스 형상을 포함한 3D 측정 [sx, sy, sz]^T
            를 방출한다. 실제 LiDAR point cloud 는 검출에 사용되지 않으며
            (선택적으로 BEV viz 용으로만 사용된다).
  - Camera: actor centroid 를 핀홀 모델로 이미지 평면에 투영하고, 이미지
            경계 + max range 로 필터링한 뒤 2D 측정 [u, v]^T 를 방출한다.

Sensor-to-world pose 는 ``ego.transform @ offset_matrix`` 로 직접 합성한다.
CARLA 의 attached-actor get_transform() 이 부모 변환의 1-tick 지연 / 부분
누락 같은 결함을 갖기 때문으로, ego transform 을 신뢰 가능한 단일 소스로
사용해 측정 정확도를 보장한다. 실제 CARLA 센서는 (a) blueprint 유효성 검증
과 (b) BEV viz 용 선택적 point cloud 두 가지 이유로 여전히 spawn 된다.
"""
import math
import threading

import numpy as np

import carla
import config


def transform_to_matrix(transform):
    """carla.Transform 을 4×4 world-from-local 동차 변환 행렬로 변환한다.

    CARLA 의 회전 순서는 R = Rz(yaw) · Ry(pitch) · Rx(roll) 이며 (Euler ZYX),
    local point 에 적용된 뒤 translation 이 수행된다. 본 합성식은 SFOT 측정
    모델의 기대값과 일치한다.

    transform : carla.Transform (location + rotation)
    반환      : np.ndarray, shape (4, 4)
    """
    loc = transform.location
    rot = transform.rotation
    cy = math.cos(math.radians(rot.yaw))
    sy = math.sin(math.radians(rot.yaw))
    cp = math.cos(math.radians(rot.pitch))
    sp = math.sin(math.radians(rot.pitch))
    cr = math.cos(math.radians(rot.roll))
    sr = math.sin(math.radians(rot.roll))
    m = np.eye(4, dtype=float)
    m[0, 0] = cp * cy
    m[0, 1] = cy * sp * sr - sy * cr
    m[0, 2] = -cy * sp * cr - sy * sr
    m[1, 0] = cp * sy
    m[1, 1] = sy * sp * sr + cy * cr
    m[1, 2] = -sy * sp * cr + cy * sr
    m[2, 0] = sp
    m[2, 1] = -cp * sr
    m[2, 2] = cp * cr
    m[0, 3] = loc.x
    m[1, 3] = loc.y
    m[2, 3] = loc.z
    return m


# ---------------------------------------------------------------------------
# LiDAR manager
# ---------------------------------------------------------------------------

class LidarManager:
    """ego 에 부착된 LiDAR 의 sensor pose / 원시 point cloud / actor-truth 3D 검출 관리.

    검출은 ``get_detections()`` 가 actor-truth 기반으로 3D 측정을 만들고,
    raw point cloud 는 BEV 시각화에서만 사용한다.
    """

    def __init__(self, world, ego_vehicle, spawn_real_sensor=True):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.sensor = None
        self._lock = threading.Lock()
        self._raw_points = []
        if spawn_real_sensor:
            self._spawn_sensor()

    def _spawn_sensor(self):
        """CARLA LiDAR blueprint 로 sensor actor 를 spawn 하고 콜백을 연결한다."""
        bp_lib = self.world.get_blueprint_library()
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', str(config.LIDAR_RANGE))
        lidar_bp.set_attribute('channels', str(config.LIDAR_CHANNELS))
        lidar_bp.set_attribute('points_per_second', str(config.LIDAR_POINTS_PER_SECOND))
        lidar_bp.set_attribute('rotation_frequency', str(config.LIDAR_ROTATION_FREQUENCY))
        lidar_bp.set_attribute('upper_fov', str(config.LIDAR_UPPER_FOV))
        lidar_bp.set_attribute('lower_fov', str(config.LIDAR_LOWER_FOV))
        tf = carla.Transform(carla.Location(
            x=config.LIDAR_POS_X, y=config.LIDAR_POS_Y, z=config.LIDAR_POS_Z,
        ))
        self.sensor = self.world.spawn_actor(lidar_bp, tf,
                                             attach_to=self.ego_vehicle)
        self.sensor.listen(self._on_lidar)

    def _on_lidar(self, data):
        """CARLA LiDAR 콜백 — 원시 데이터를 sub-sample 해 _raw_points 에 저장."""
        raw = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        with self._lock:
            # BEV viz 를 가볍게 유지하기 위해 8 점마다 1 개만 보관.
            self._raw_points = [
                (float(r[0]), float(r[1]), float(r[2]))
                for r in raw[::8] if r[0] > 0
            ]

    def get_raw_points(self):
        """가장 최근 tick 의 sub-sample LiDAR raw point 리스트를 복사 반환."""
        with self._lock:
            return list(self._raw_points)

    def get_sensor_to_world(self):
        """ego transform + 마운트 offset 으로부터 LiDAR pose 4×4 행렬을 합성한다."""
        ego_matrix = transform_to_matrix(self.ego_vehicle.get_transform())
        offset = np.eye(4, dtype=float)
        offset[0, 3] = config.LIDAR_POS_X
        offset[1, 3] = config.LIDAR_POS_Y
        offset[2, 3] = config.LIDAR_POS_Z
        return ego_matrix @ offset

    def get_detections(self):
        """현재 ego 주변 차량 actor 들을 sensor-local 3D 측정 리스트로 변환.

        actor truth (위치 + bounding box) 를 sensor frame 으로 변환한 뒤
        센서 후방 / range 밖을 거른다. 각 검출은 dict {x, y, z, length,
        width, height, yaw, point_count, actor_id} 형식.
        """
        world_to_sensor = np.linalg.inv(self.get_sensor_to_world())
        detections = []
        for actor in self.world.get_actors().filter('vehicle.*'):
            if actor.id == self.ego_vehicle.id:
                continue
            if not actor.is_alive:
                continue
            tf = actor.get_transform()
            loc = tf.location
            bbox = actor.bounding_box
            pt_world = np.array([[loc.x], [loc.y], [loc.z], [1.0]],
                                dtype=float)
            pt_sensor = world_to_sensor @ pt_world
            sx = float(pt_sensor[0, 0])
            sy = float(pt_sensor[1, 0])
            sz = float(pt_sensor[2, 0])
            if sx <= 0.5:                          # 센서 뒤편
                continue
            dist = math.sqrt(sx * sx + sy * sy + sz * sz)
            if dist < config.LIDAR_MIN_RANGE or dist > config.LIDAR_RANGE:
                continue
            ego_yaw = math.radians(self.ego_vehicle.get_transform().rotation.yaw)
            actor_yaw = math.radians(tf.rotation.yaw)
            detections.append({
                'x': sx, 'y': sy, 'z': sz,
                'length': float(bbox.extent.x * 2),
                'width': float(bbox.extent.y * 2),
                'height': float(bbox.extent.z * 2),
                'yaw': actor_yaw - ego_yaw,
                'point_count': 20,
                'actor_id': int(actor.id),
            })
        return detections

    def destroy(self):
        """CARLA 센서 actor 를 정리한다 (종료 시 호출)."""
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None


# ---------------------------------------------------------------------------
# Camera manager
# ---------------------------------------------------------------------------

class CameraManager:
    """ego 에 부착된 전방 카메라의 sensor pose / 이미지 / actor-truth 2D 검출 관리.

    핀홀 모델 (fx, fy, cx, cy) 를 직접 계산해 SFOT 측정 / BEV 오버레이가
    동일한 intrinsic 으로 동작하도록 일관성을 유지한다.
    """

    def __init__(self, world, ego_vehicle, spawn_real_sensor=True):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.sensor = None
        self.width = config.CAMERA_WIDTH
        self.height = config.CAMERA_HEIGHT
        self.fov_deg = config.CAMERA_FOV_DEG
        self.fx = self.width / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))
        self.fy = self.fx
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        self._latest_image = None
        self._image_lock = threading.Lock()
        if spawn_real_sensor:
            self._spawn_sensor()

    def _spawn_sensor(self):
        """CARLA RGB 카메라를 spawn 하고 이미지 콜백을 연결한다."""
        bp_lib = self.world.get_blueprint_library()
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.width))
        camera_bp.set_attribute('image_size_y', str(self.height))
        camera_bp.set_attribute('fov', str(self.fov_deg))
        tf = carla.Transform(
            carla.Location(x=config.CAMERA_POS_X, y=config.CAMERA_POS_Y,
                           z=config.CAMERA_POS_Z),
            carla.Rotation(pitch=config.CAMERA_PITCH_DEG),
        )
        self.sensor = self.world.spawn_actor(camera_bp, tf,
                                             attach_to=self.ego_vehicle)
        self.sensor.listen(self._on_image)

    def _on_image(self, image):
        """CARLA 카메라 콜백 — BGRA → BGR 변환 + 복사본 저장."""
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((self.height, self.width, 4))[:, :, :3]
        with self._image_lock:
            self._latest_image = arr.copy()

    def get_image(self):
        """가장 최근 tick 의 BGR 이미지 복사본을 반환. 없으면 None."""
        with self._image_lock:
            return None if self._latest_image is None else self._latest_image.copy()

    def get_intrinsic(self):
        """fx, fy, cx, cy + fov_deg + width / height 를 dict 로 반환."""
        return {
            'fx': self.fx, 'fy': self.fy,
            'cx': self.cx, 'cy': self.cy,
            'fov_deg': self.fov_deg,
            'width': self.width, 'height': self.height,
        }

    def get_sensor_to_world(self):
        """ego transform + 마운트 offset + pitch 보정으로 카메라 pose 4×4 합성.

        attached-actor get_transform() 결함을 피하기 위해 ego transform 을
        신뢰 가능한 단일 소스로 삼고 마운트 offset 을 곱해 정확도를 보장한다.
        """
        ego_matrix = transform_to_matrix(self.ego_vehicle.get_transform())
        offset = np.eye(4, dtype=float)
        offset[0, 3] = config.CAMERA_POS_X
        offset[1, 3] = config.CAMERA_POS_Y
        offset[2, 3] = config.CAMERA_POS_Z
        # 카메라의 작은 pitch 보정
        pitch_rad = math.radians(config.CAMERA_PITCH_DEG)
        cp = math.cos(pitch_rad)
        sp = math.sin(pitch_rad)
        offset[0, 0] = cp
        offset[0, 2] = sp
        offset[1, 1] = 1.0
        offset[2, 0] = -sp
        offset[2, 2] = cp
        return ego_matrix @ offset

    def _project_world_point(self, world_point):
        """world 점을 카메라 픽셀로 투영. 센서 후방이거나 화면 밖이면 None."""
        world_to_sensor = np.linalg.inv(self.get_sensor_to_world())
        pt = np.array([[world_point[0]], [world_point[1]], [world_point[2]],
                       [1.0]], dtype=float)
        ps = world_to_sensor @ pt
        sx = float(ps[0, 0])
        sy = float(ps[1, 0])
        sz = float(ps[2, 0])
        if sx <= 0.1:
            return None
        u = self.cx + self.fx * sy / sx
        v = self.cy - self.fy * sz / sx
        if not (0.0 <= u < self.width and 0.0 <= v < self.height):
            return None
        return u, v, sx

    def get_detections(self):
        """현재 ego 주변 차량 actor 의 카메라 픽셀 검출 리스트를 반환.

        actor bbox 중심을 핀홀 모델로 투영하고, 이미지 경계 + max range
        밖을 거른다. 각 검출은 {u, v, actor_id} 형식.
        """
        detections = []
        for actor in self.world.get_actors().filter('vehicle.*'):
            if actor.id == self.ego_vehicle.id:
                continue
            if not actor.is_alive:
                continue
            tf = actor.get_transform()
            actor_matrix = transform_to_matrix(tf)
            bbox_center = actor.bounding_box.location
            center_world = actor_matrix @ np.array(
                [[bbox_center.x], [bbox_center.y], [bbox_center.z], [1.0]],
                dtype=float,
            )
            proj = self._project_world_point(center_world[0:3, 0])
            if proj is None:
                continue
            u, v, fwd = proj
            if fwd > config.CAMERA_MAX_RANGE:
                continue
            detections.append({
                'u': float(u), 'v': float(v),
                'actor_id': int(actor.id),
            })
        return detections

    def destroy(self):
        """CARLA 카메라 actor 를 정리한다 (종료 시 호출)."""
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None
