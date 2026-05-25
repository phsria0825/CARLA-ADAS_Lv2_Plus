"""
Mahalanobis (chi² 99.5 %) gating 을 적용한 순차 nearest-neighbor 매칭.

패턴:
  associate(tracks, meas, KF) 가 MHD 거리로 association_matrix 를 구축하며,
  FOV 또는 gating 검사에 실패한 쌍은 +inf 로 채워 둔다. 이후
  get_closest_track_and_meas() 가 greedy 방식으로 최소 거리를 선택하고 해당
  행/열을 제거하며, 더 이상 남은 쌍이 없을 때까지 반복한다. 연결되지 않은
  측정/트랙은 init / scoring 을 위해 track-management 로 넘긴다.
"""
import numpy as np

import perception.sfot_params as P


def _mhd(track, meas, KF):
    """Mahalanobis 거리 d² = γ^T S⁻¹ γ 를 산출한다.

    track : Track 인스턴스
    meas  : Measurement 인스턴스
    KF    : Filter 인스턴스 (S 와 γ 를 같은 모델로 계산)
    반환  : float (제곱 거리)
    """
    H = meas.sensor.get_H(track.x)
    S = KF.S(track, meas, H)
    gamma = KF.gamma(track, meas)
    return float((gamma.T @ np.linalg.inv(S) @ gamma).item())


class Association:
    """단일 센서 pass 의 측정-트랙 매칭 + 연결되지 않은 항목 추적.

    associate() 가 association_matrix 를 채운 뒤, 호출자가
    get_closest_track_and_meas() 를 반복 호출하거나 associate_and_update()
    한 번으로 매칭+갱신 전체를 한꺼번에 처리한다.
    """

    def __init__(self):
        self.association_matrix = np.empty((0, 0), dtype=float)
        self.unassigned_tracks = []
        self.unassigned_meas = []

    def associate(self, track_list, meas_list, KF):
        """모든 (트랙, 측정) 쌍의 MHD 를 계산해 association_matrix 를 채운다.

        FOV 밖이거나 gating 임계 (chi² 99.5 %) 를 통과하지 못한 쌍은 +inf
        로 남는다. unassigned_tracks / unassigned_meas 도 전체 인덱스로
        초기화된다.

        track_list : 현재 확정/잠정 트랙 리스트
        meas_list  : 이번 pass 의 측정 리스트
        KF         : Filter 인스턴스 (gamma / S 계산용)
        """
        num_tracks = len(track_list)
        num_meas = len(meas_list)
        self.unassigned_tracks = list(range(num_tracks))
        self.unassigned_meas = list(range(num_meas))
        self.association_matrix = np.full(
            (num_tracks, num_meas), np.inf, dtype=float
        )
        for ti, track in enumerate(track_list):
            for mi, meas in enumerate(meas_list):
                if not meas.sensor.in_fov(track.x):
                    continue
                dist = _mhd(track, meas, KF)
                if dist <= P.GATING_THRESHOLDS[meas.sensor.dim_meas]:
                    self.association_matrix[ti, mi] = dist

    def get_closest_track_and_meas(self):
        """association_matrix 에서 최소 MHD 쌍을 한 번 뽑아 반환한다.

        뽑힌 행과 열을 행렬에서 즉시 제거하고, unassigned 리스트에서도
        해당 인덱스를 빼낸다. 더 이상 유한한 값이 없으면 (None, None) 반환.

        반환 : (track_index, meas_index) — 원래 track_list / meas_list 의 인덱스
        """
        if (self.association_matrix.size == 0
                or not np.isfinite(self.association_matrix).any()):
            return None, None
        row, col = np.unravel_index(
            np.argmin(self.association_matrix, axis=None),
            self.association_matrix.shape,
        )
        update_track = self.unassigned_tracks[row]
        update_meas = self.unassigned_meas[col]
        self.association_matrix = np.delete(self.association_matrix, row, axis=0)
        self.association_matrix = np.delete(self.association_matrix, col, axis=1)
        del self.unassigned_tracks[row]
        del self.unassigned_meas[col]
        return update_track, update_meas

    def associate_and_update(self, manager, meas_list, KF, sensor=None):
        """한 센서의 측정 배치에 대해 predict-associate-update-manage 흐름을 묶어 실행.

        FusionManager 가 LiDAR pass 와 Camera pass 각각에 대해 한 번씩 호출
        한다. 매칭된 쌍에 대해 KF.update() + manager.handle_updated_track()
        을 수행하고, 남은 미할당 트랙/측정은 manage_tracks() 로 넘긴다.

        manager   : Trackmanagement 인스턴스
        meas_list : 이번 pass 의 측정 리스트
        KF        : Filter 인스턴스
        sensor    : 측정이 비어 있는 경우의 sensor fallback (선택)
        """
        self.associate(manager.track_list, meas_list, KF)
        while (self.association_matrix.shape[0] > 0
               and self.association_matrix.shape[1] > 0):
            ti, mi = self.get_closest_track_and_meas()
            if ti is None:
                break
            track = manager.track_list[ti]
            meas = meas_list[mi]
            if not meas.sensor.in_fov(track.x):
                continue
            KF.update(track, meas)
            manager.handle_updated_track(track)
            manager.track_list[ti] = track
        if sensor is None and meas_list:
            sensor = meas_list[0].sensor
        manager.manage_tracks(
            self.unassigned_tracks, self.unassigned_meas, meas_list,
            sensor=sensor,
        )
