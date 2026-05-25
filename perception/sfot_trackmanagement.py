"""
Track + Trackmanagement — score 기반 lifecycle, 모션 모델에 비의존.

Track 은 차원이 모션 모델로부터 결정되는 x, P 를 저장한다. 초기화에는
motion_model.initial_state / initial_covariance 가 사용된다.

Lifecycle:
    score ∈ [0, 1], window = 8.
    hit  → score += 1/8 (상한 1)
    miss → score -= 1/8 (하한 0)
    score ≥ 0.7 이면 confirmed.
    삭제 조건:
      - confirmed 상태이면서 score < 0.4
      - tentative 상태이면서 score == 0 그리고 misses ≥ 2
      - 위치 분산이 폭주 (P[i,i] > MAX_P_POS for i in position_indices)
"""
import numpy as np

import perception.sfot_params as P


class Track:
    """단일 객체의 EKF 상태 + lifecycle + 형상 속성을 묶은 인스턴스.

    Trackmanagement 가 LiDAR 측정으로부터 init_track() 으로 생성한다.
    Filter.update() 가 매 측정마다 set_x / set_P / update_attributes 를
    호출해 갱신한다.
    """

    def __init__(self, meas, track_id, motion_model):
        self.motion = motion_model

        # LiDAR 측정으로부터 산출한 world 좌표계 위치 (sensor frame → world).
        # 카메라 측정은 depth 가 없어 트랙을 초기화할 수 없으며, 이는
        # Trackmanagement.manage_tracks 에서 강제된다.
        sens_to_world = meas.sensor.sens_to_world
        pos_sens = np.ones((4, 1), dtype=float)
        pos_sens[0:3, 0] = meas.z[0:3, 0]
        pos_world = sens_to_world @ pos_sens
        rot = sens_to_world[0:3, 0:3]
        R_pos_world = rot @ meas.R @ rot.T

        # 초기 yaw 는 LiDAR bounding box yaw 로부터. CTRV 의 경우 초기
        # heading 을 고정하는 역할을 한다.
        yaw_world = meas.yaw
        pos_world_tuple = (float(pos_world[0, 0]),
                           float(pos_world[1, 0]),
                           float(pos_world[2, 0]))
        self.x = motion_model.initial_state(pos_world_tuple, yaw_world)
        self.P = motion_model.initial_covariance(R_pos_world)

        # Lifecycle 상태
        self.state = 'initialized'
        self.score = 1.0 / P.SCORE_WINDOW
        self.hits = 1
        self.misses = 0
        self.id = track_id
        self.t = meas.t

        # 형상 속성
        self.width = meas.width
        self.length = meas.length
        self.height = meas.height
        self.yaw = meas.yaw
        self.point_count = meas.point_count
        self.last_sensor = meas.sensor.name
        self.last_actor_id = meas.actor_id

    def set_x(self, x):
        """상태 벡터 x 를 갱신 (Filter.predict / update 에서 호출)."""
        self.x = x

    def set_P(self, p):
        """상태 공분산 P 를 갱신."""
        self.P = p

    def update_attributes(self, meas):
        """측정 부가 속성을 트랙에 반영 — LiDAR 면 형상을 blending.

        LiDAR 의 width / length / height / yaw 는 SHAPE_BLEND_WEIGHT 로 가중
        평균해 점진적으로 안정화. 마지막 센서 이름, actor_id, 시각은 매번
        교체된다.
        """
        if meas.sensor.name == 'lidar':
            c = P.SHAPE_BLEND_WEIGHT
            self.width = c * meas.width + (1.0 - c) * self.width
            self.length = c * meas.length + (1.0 - c) * self.length
            self.height = c * meas.height + (1.0 - c) * self.height
            self.yaw = meas.yaw
            self.point_count = meas.point_count
        self.last_sensor = meas.sensor.name
        self.last_actor_id = meas.actor_id
        self.t = meas.t


class Trackmanagement:
    """확정/잠정 트랙 리스트를 보유하고 score-based lifecycle 을 진행한다.

    Association.associate_and_update() 와 FusionManager 가 호출한다. 트랙
    init 은 LiDAR 측정에서만 허용되어 depth 정보가 없는 카메라가 새 트랙을
    만들지 못하도록 한다.
    """

    def __init__(self, motion_model):
        self.motion = motion_model
        self.track_list = []
        self.N = 0
        self.last_id = -1

    def addTrackToList(self, track):
        """track_list 에 추가하고 N 과 last_id 를 갱신."""
        self.track_list.append(track)
        self.N += 1
        self.last_id = track.id

    def init_track(self, meas):
        """LiDAR 측정으로부터 새 Track 인스턴스를 만들어 등록."""
        track = Track(meas, self.last_id + 1, self.motion)
        self.addTrackToList(track)

    def delete_track(self, track):
        """track_list 에서 제거하고 N 을 재계산."""
        self.track_list.remove(track)
        self.N = len(self.track_list)

    def manage_tracks(self, unassigned_tracks, unassigned_meas, meas_list, sensor=None):
        """미할당 트랙에는 페널티, 폭주/저점수 트랙은 삭제, LiDAR 미할당 측정은 init.

        세 단계로 수행:
            1) 센서 FOV 안에 있어야 할 트랙이 매칭 안 됐으면 score -1/8,
               misses += 1, 비 confirmed 면 tentative 로 down.
            2) 위치 공분산이 MAX_P_POS 초과로 폭주했거나 score 가 낮으면 삭제.
            3) LiDAR 미할당 측정으로부터 새 트랙 init.

        unassigned_tracks : Association 이 남긴 미할당 트랙 인덱스 리스트
        unassigned_meas   : Association 이 남긴 미할당 측정 인덱스 리스트
        meas_list         : 이번 pass 의 전체 측정 리스트
        sensor            : FOV 평가용 sensor (선택)
        """
        # 1) 센서가 봤어야 하는 트랙에 페널티
        for idx in sorted(unassigned_tracks, reverse=True):
            if idx >= len(self.track_list):
                continue
            track = self.track_list[idx]
            visible = sensor.in_fov(track.x) if sensor is not None else True
            if visible:
                track.score = max(0.0, track.score - 1.0 / P.SCORE_WINDOW)
                track.misses += 1
                if track.state != 'confirmed':
                    track.state = 'tentative'

        # 2) 불확실성 폭주 또는 저점수 트랙 삭제
        pos_idx = self.motion.position_indices
        for track in list(self.track_list):
            pos_var_too_large = any(
                track.P[i, i] > P.MAX_P_POS for i in pos_idx
            )
            low_score = (
                (track.state == 'confirmed' and track.score < P.DELETE_SCORE)
                or (track.state != 'confirmed'
                    and track.score <= 0.0 and track.misses >= 2)
            )
            if pos_var_too_large or low_score:
                self.delete_track(track)

        # 3) LiDAR 측정만으로 새 트랙 init
        for mi in unassigned_meas:
            meas = meas_list[mi]
            if meas.sensor.name == 'lidar':
                self.init_track(meas)

    def handle_updated_track(self, track):
        """매칭 + EKF 갱신 직후 호출 — score +1/8, hits +1, 임계 넘으면 confirmed."""
        track.score = min(1.0, track.score + 1.0 / P.SCORE_WINDOW)
        track.hits += 1
        track.misses = 0
        if track.score >= P.CONFIRMED_SCORE:
            track.state = 'confirmed'
        else:
            track.state = 'tentative'
