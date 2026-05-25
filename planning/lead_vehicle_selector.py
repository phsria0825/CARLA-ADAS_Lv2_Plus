"""
FusedObject 리스트로부터 현재 / 좌 / 우 인접 차선의 lead vehicle 을 동시 선정.

선정 규칙 (모든 차선에 동일):
    1. ego 전방 거리 > _MIN_FORWARD_M 이다
    2. 해당 차선 centerline 에 대한 횡방향 offset ≤ lane_width/2 + margin 이다
    3. config.PERCEPTION_RANGE 전방 이내이다
    4. 조건을 만족하는 가장 가까운 트랙을 선택한다
    5. 트랙 confidence ≥ _MIN_LEAD_CONFIDENCE 를 요구한다

호출자 계약:
  * SCC / 종방향 제어는 ``LeadInfoSet.current`` 만 소비한다 (기존 호환).
  * BehaviorPlanner / MOBIL 은 ``left`` / ``right`` 도 활용해 인접 차선 lead
    진입을 의사결정에 반영한다.
  * 좌/우 차선이 invalid 한 경우 (lane.{left,right}_lane_available == False)
    는 ``LeadInfoSet.left/right`` 가 None 이며, 차선은 있으나 lead 가 없을
    뿐이면 ``detected = False`` 인 LeadInfo 가 들어간다.
"""
import math

import config
from core.adas_types import LeadInfo, LeadInfoSet


_MARGIN_M = 0.3
_MIN_FORWARD_M = 1.0
_MIN_LEAD_CONFIDENCE = 0.35

# Range-rate smoothing 메모리 키
_KEY_CURRENT = "current"
_KEY_LEFT = "left"
_KEY_RIGHT = "right"


def _project_lateral_offset(polyline, px_world, py_world):
    """polyline 까지의 부호 없는 횡거리와 polyline 을 따라 잰 종방향 호 길이.

    LaneModel.centerline 만이 아니라 left_centerline / right_centerline 등
    임의 polyline 에 대해 동일 계산을 수행한다. polyline 이 2 점 미만이면
    (inf, 0) 을 반환한다.

    polyline : [(x, y), ...] 월드 좌표 점열
    px_world : 검사할 점의 월드 x
    py_world : 검사할 점의 월드 y
    반환     : (lateral_distance, arc_length_to_projection)
    """
    if len(polyline) < 2:
        return float('inf'), 0.0
    best_d = float('inf')
    best_s = 0.0
    s_accum = 0.0
    for i in range(len(polyline) - 1):
        ax, ay = polyline[i]
        bx, by = polyline[i + 1]
        sx = bx - ax
        sy = by - ay
        seg_len2 = sx * sx + sy * sy
        if seg_len2 < 1e-9:
            continue
        t = ((px_world - ax) * sx + (py_world - ay) * sy) / seg_len2
        t_cl = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        proj_x = ax + sx * t_cl
        proj_y = ay + sy * t_cl
        d = math.hypot(px_world - proj_x, py_world - proj_y)
        seg_len = math.sqrt(seg_len2)
        if d < best_d:
            best_d = d
            best_s = s_accum + seg_len * t_cl
        s_accum += seg_len
    return best_d, best_s


class LeadVehicleSelector:
    """현재 + 좌·우 인접 차선의 lead 차량을 매 tick 동시 선정한다.

    메모이제이션: lane key 별로 (track_id, range, timestamp) 스냅샷을 유지
    한다. 동일 track id 가 유지되는 동안엔 dRange/dt 로 range-rate 를 산출
    하고, 트랙이 바뀌면 EKF 의 vx_ego 로 fallback 한다. dRange/dt 가 속도
    성분의 EKF noise 에 덜 민감하므로 같은 트랙이 유지되는 동안엔 이쪽을
    우선한다.
    """

    def __init__(self):
        # key -> (track_id, range, timestamp) 또는 None
        self._last = {
            _KEY_CURRENT: None,
            _KEY_LEFT: None,
            _KEY_RIGHT: None,
        }

    def update(self, ego_state, lane, fused_objects, timestamp):
        """현재 / 좌 / 우 lead 차량을 한 묶음의 LeadInfoSet 으로 반환한다.

        lane.is_valid == False 인 경우 모든 lead 가 None / detected=False
        로 초기화된 빈 LeadInfoSet 을 반환하고 내부 메모리도 reset 한다.

        ego_state      : 자차 상태
        lane           : 현재 LaneModel (centerline + 좌·우 centerline 포함)
        fused_objects  : confirmed FusedObject 리스트
        timestamp      : 현 tick 시각 [s]
        반환           : LeadInfoSet (current / left / right)
        """
        if not lane.is_valid:
            self._reset_all()
            return LeadInfoSet()

        half_w = lane.lane_width * 0.5

        # 1) 현재 차선 — 기존 호환 경로
        current_lead = self._select_for_polyline(
            polyline=lane.centerline,
            lane_id=lane.current_lane_id,
            half_w=half_w,
            fused=fused_objects,
            ego_state=ego_state,
            timestamp=timestamp,
            key=_KEY_CURRENT,
        )

        # 2) 좌 차선 (운전자 좌표계; lane_provider 가 정규화)
        if lane.left_lane_available and len(lane.left_centerline) >= 2:
            left_lead = self._select_for_polyline(
                polyline=lane.left_centerline,
                lane_id=None,
                half_w=half_w,
                fused=fused_objects,
                ego_state=ego_state,
                timestamp=timestamp,
                key=_KEY_LEFT,
            )
        else:
            left_lead = None
            self._last[_KEY_LEFT] = None

        # 3) 우 차선
        if lane.right_lane_available and len(lane.right_centerline) >= 2:
            right_lead = self._select_for_polyline(
                polyline=lane.right_centerline,
                lane_id=None,
                half_w=half_w,
                fused=fused_objects,
                ego_state=ego_state,
                timestamp=timestamp,
                key=_KEY_RIGHT,
            )
        else:
            right_lead = None
            self._last[_KEY_RIGHT] = None

        return LeadInfoSet(
            current=current_lead,
            left=left_lead,
            right=right_lead,
        )

    def _select_for_polyline(
        self,
        polyline,
        lane_id,
        half_w,
        fused,
        ego_state,
        timestamp,
        key,
    ):
        """주어진 차선 polyline 에 대해 단일 lead LeadInfo 를 선정한다.

        polyline 안의 가장 가까운 confidence 통과 트랙을 lead 로 잡고,
        range_rate 를 dRange/dt 또는 EKF vx_ego fallback 으로 산출하며,
        AEB_MIN_CLOSING_SPEED 이상의 접근 속도일 때만 TTC 를 계산한다.
        """
        best = None
        best_range = float('inf')
        best_lateral = 0.0

        for obj in fused:
            if obj.confidence < _MIN_LEAD_CONFIDENCE:
                continue
            if obj.x_ego < _MIN_FORWARD_M:
                continue
            lat, lon = _project_lateral_offset(polyline, obj.x_world, obj.y_world)
            if lat > (half_w + _MARGIN_M):
                continue
            range_m = max(lon, obj.x_ego)
            if range_m > config.PERCEPTION_RANGE:
                continue
            if range_m < best_range:
                best_range = range_m
                best = obj
                best_lateral = lat

        if best is None:
            self._last[key] = None
            return LeadInfo(detected=False, lane_id=lane_id)

        # Range-rate 산출
        last = self._last[key]
        if (last is not None
                and last[0] == best.track_id
                and (timestamp - last[2]) > 1e-3):
            dr = best_range - last[1]
            dt = timestamp - last[2]
            range_rate = dr / dt
        else:
            # Lead 의 ego-frame 종방향 속도 = 절대 전방 속도.
            # range_rate 는 ego forward axis 를 따라 잰 (lead - ego) 위치의
            # 시간 미분이며 = (lead_vx_ego - ego.speed_mps) 와 동치.
            range_rate = best.vx_ego - ego_state.speed_mps

        self._last[key] = (best.track_id, best_range, timestamp)

        ttc = float('inf')
        if range_rate < -config.AEB_MIN_CLOSING_SPEED:
            ttc = best_range / max(-range_rate, 1e-3)

        return LeadInfo(
            detected=True,
            track_id=best.track_id,
            range=best_range,
            range_rate=range_rate,
            lateral_offset=best_lateral,
            lane_id=lane_id,
            ttc=ttc,
            confidence=best.confidence,
            lead_speed_mps=best.speed_mps,
        )

    def _reset_all(self):
        """세 차선 모두의 range-rate smoothing 메모리를 초기화한다."""
        self._last[_KEY_CURRENT] = None
        self._last[_KEY_LEFT] = None
        self._last[_KEY_RIGHT] = None
