"""
FusionManager — SFOT 파이프라인을 조율해 매 tick 의 FusedObject 리스트를 발행.

확정된 SFOT 트랙을 BehaviorPlanner / BEV / lead_vehicle_selector 가 사용하는
프로젝트 전역 FusedObject 계약으로 변환한다.

Tick 단위 흐름:
    1) 모든 기존 트랙을 predict
    2) LiDAR 측정 생성 (3D, sensor frame)
    3) LiDAR 에 대해 Association.associate_and_update 수행
       (할당되지 않은 LiDAR 측정으로부터 Trackmanagement 가 새 트랙 init)
    4) Camera 측정 생성 (2D pixel)
    5) Camera 에 대해 Association.associate_and_update 수행
       (카메라는 트랙 init 불가 — Trackmanagement 가 LiDAR-only init 강제)
    6) confirmed 트랙을 ego frame FusedObject 로 방출

두 센서를 결합 배치가 아니라 순차로 처리하면 센서별로 다른 측정 노이즈 R 과
gating 임계를 자연스럽게 분리할 수 있다.
"""
import math

import numpy as np

import config
from core.adas_types import FusedObject
from core.adas_utils import world_to_ego
from perception.perception import (
    CameraManager,
    LidarManager,
)
import perception.sfot_params as SFOT_P
from perception.sfot_association import Association
from perception.sfot_filter import Filter
from perception.sfot_measurements import Sensor
from perception.sfot_motion import build_motion_model
from perception.sfot_trackmanagement import Trackmanagement


class FusionManager:
    """main_adas_loop 이 사용하는 perception + fusion 의 단일 진입점.

    update() 한 번 호출로 한 tick 전체의 SFOT 파이프라인을 돌리고
    FusedObject 리스트를 발행한다. 모션 모델은 config.SFOT_MOTION_MODEL 로
    선택한다 (기본 CTRV).
    """

    def __init__(self, world, ego_vehicle):
        # 센서 매니저 (blueprint 유효성과 BEV raw-points 사용을 위해 실제
        # CARLA 센서를 spawn 하지만, 검출은 여전히 actor-truth 기반).
        self.lidar_mgr = LidarManager(world, ego_vehicle, spawn_real_sensor=True)
        self.camera_mgr = CameraManager(world, ego_vehicle, spawn_real_sensor=True)
        # SFOT 컴포넌트 — 모션 모델은 config.SFOT_MOTION_MODEL (CV / CTRV) 로 선택.
        self.motion = build_motion_model(SFOT_P.SFOT_MOTION_MODEL)
        print(f"[FUSION] motion model = {self.motion.name} "
              f"(dim_state={self.motion.dim_state})")
        self.kf = Filter(self.motion)
        self.association = Association()
        self.manager = Trackmanagement(self.motion)
        self.lidar_sensor = Sensor('lidar', motion_model=self.motion)
        self.camera_sensor = Sensor(
            'camera', motion_model=self.motion,
            intrinsic=self.camera_mgr.get_intrinsic(),
        )
        self._frame_index = 0
        self.ego_vehicle = ego_vehicle

    def update(self, ego_state, timestamp):
        """한 tick 전체의 perception + fusion 파이프라인 실행.

        pose 갱신 → 트랙 predict → LiDAR pass associate+update → Camera pass
        associate+update → confirmed 트랙을 FusedObject 로 방출.

        ego_state  : 자차 상태 (ego frame 변환에 사용)
        timestamp  : 현 tick 시각 [s]
        반환       : list[FusedObject] (confirmed 트랙만)
        """
        self._frame_index += 1

        # 현재 ego transform 에 대해 센서 pose 를 갱신한다.
        self.lidar_sensor.update_pose(self.lidar_mgr.get_sensor_to_world())
        self.camera_sensor.update_pose(self.camera_mgr.get_sensor_to_world())

        # 모든 기존 트랙을 dt 만큼 predict.
        for track in self.manager.track_list:
            self.kf.predict(track)

        # --- LiDAR pass: 3D 측정, 새 트랙을 init 가능 ---
        lidar_dets = self.lidar_mgr.get_detections()
        lidar_meas = self._build_lidar_measurements(lidar_dets)
        try:
            self.association.associate_and_update(
                self.manager, lidar_meas, self.kf, sensor=self.lidar_sensor,
            )
        except Exception as e:
            print(f"[FUSION] LiDAR pass failed: {e}")
            # Safety net: 할당되지 않은 LiDAR 측정은 여전히 트랙을 초기화하도록.
            self.manager.manage_tracks(
                list(range(len(self.manager.track_list))),
                list(range(len(lidar_meas))),
                lidar_meas,
                sensor=self.lidar_sensor,
            )

        # --- Camera pass: 2D 픽셀 측정, 기존 트랙만 보정 (init 불가) ---
        cam_dets = self.camera_mgr.get_detections()
        cam_meas = self._build_camera_measurements(cam_dets)
        try:
            self.association.associate_and_update(
                self.manager, cam_meas, self.kf, sensor=self.camera_sensor,
            )
        except Exception as e:
            print(f"[FUSION] Camera pass failed: {e}")

        return self._emit_fused_objects(ego_state, timestamp)

    def _build_lidar_measurements(self, lidar_dets):
        """LiDAR 검출 dict 리스트를 Measurement 리스트로 wrap (FUSION_RANGE 필터)."""
        out = []
        for det in lidar_dets:
            dist = math.sqrt(det['x'] ** 2 + det['y'] ** 2 + det['z'] ** 2)
            if dist > config.FUSION_RANGE:
                continue
            self.lidar_sensor.generate_measurement(
                self._frame_index,
                [det['x'], det['y'], det['z']],
                out,
                width=det['width'], length=det['length'], height=det['height'],
                yaw=det['yaw'], point_count=det['point_count'],
                actor_id=det['actor_id'],
            )
        return out

    def _build_camera_measurements(self, cam_dets):
        """Camera 검출 dict 리스트를 Measurement 리스트로 wrap."""
        out = []
        for det in cam_dets:
            self.camera_sensor.generate_measurement(
                self._frame_index,
                [det['u'], det['v']],
                out,
                actor_id=det['actor_id'],
            )
        return out

    def _emit_fused_objects(self, ego_state, timestamp):
        """confirmed 트랙을 프로젝트 전역 FusedObject 로 변환해 리스트로 반환.

        모션 모델의 extract_position / extract_velocity_world 헬퍼를 쓰므로
        CV / CTRV 에 관계없이 변환 절차가 동일하다 (CV 는 상태에서 속도 직접
        제공, CTRV 는 v·cos(ψ), v·sin(ψ) 로 분해).
        """
        out = []
        for track in self.manager.track_list:
            if track.state != 'confirmed':
                continue
            px_w, py_w, _ = self.motion.extract_position(track.x)
            vx_w, vy_w, _ = self.motion.extract_velocity_world(track.x)
            x_ego, y_ego = world_to_ego(
                px_w, py_w, ego_state.x, ego_state.y, ego_state.yaw_rad,
            )
            cos_e = math.cos(-ego_state.yaw_rad)
            sin_e = math.sin(-ego_state.yaw_rad)
            vx_ego = vx_w * cos_e - vy_w * sin_e
            vy_ego = vx_w * sin_e + vy_w * cos_e
            speed = math.hypot(vx_w, vy_w)
            # 모션 모델이 yaw 를 추적하면 (CTRV) 그 값을 우선 사용,
            # 그렇지 않으면 LiDAR-blended shape yaw 로 fallback (CV).
            yaw_world = self.motion.extract_yaw_world(track.x)
            if yaw_world is None:
                yaw_world = track.yaw
            yaw_ego = yaw_world - ego_state.yaw_rad
            out.append(FusedObject(
                track_id=track.id,
                object_type='vehicle',
                x_ego=x_ego, y_ego=y_ego,
                vx_ego=vx_ego, vy_ego=vy_ego,
                yaw_ego=yaw_ego,
                length=track.length, width=track.width,
                confidence=track.score,
                age_s=0.0,
                x_world=px_w, y_world=py_w,
                yaw_world=yaw_world,
                speed_mps=speed,
            ))
        return out

    def destroy(self):
        """CARLA 센서 actor 들을 모두 정리한다 (종료 시 호출)."""
        try:
            self.lidar_mgr.destroy()
        except Exception:
            pass
        try:
            self.camera_mgr.destroy()
        except Exception:
            pass
