"""
LaneProvider — CARLA OpenDRIVE 맵으로부터 운전자 좌표계 차선 모델을 산출.

본 모듈은 HD map (CARLA 의 map.get_waypoint() / waypoint.next()) 만 사용한다.
카메라 기반 차선 검출은 수행하지 않는다.

본 모듈이 다루는 OpenDRIVE / CARLA 특유의 사례:

  - get_left_lane() / get_right_lane() 은 차량 진행 방향이 아닌 OpenDRIVE
    도로 기준선 (reference line) 기준의 "좌/우" 를 반환한다. lane_id > 0
    인 차선 (기준선과 반대 방향 주행) 의 경우 OpenDRIVE 의 "좌" 는 기하학
    적으로 운전자의 우측에 해당하므로, _driver_left_right() 와
    _driver_lane_change_allows() 가 중앙에서 일관되게 좌우를 교체해
    운전자 좌표계로 통일한다.

  - waypoint.next(d) 가 junction 에서 반대 방향 분기로 점프하는 경우가
    있어, _polyline_from_waypoints 내부에서 `nxt.lane_id * cur.lane_id < 0`
    조건으로 부호 반전 후보를 거부한다.

  - junction 내부에서 get_waypoint() 가 yaw 와 무관하게 가장 가까운 차선
    중심으로 snap 되어 반대편 차선이 선택될 수 있다. next() walk 휴리스틱
    (yaw 일치 보너스 + route hint 보너스) 으로 yaw 가 정렬된 후보를
    우선시한다.

  - OpenDRIVE lane section 경계에서 lane_id 가 재번호화될 수 있어 "동일
    차선" 판정에 lane_id 만 쓰지 않고 (road_id, section_id, lane_id) 3-튜플
    을 사용한다.

  - lane_width 가 OpenDRIVE poly3 width 함수의 <width sOffset=...> 엔트리
    경계에서 계단형으로 점프할 수 있어, planner 를 위해 향후 약 10 m 구간의
    median 으로 평활화한다.

OpenDRIVE lane-id 의미:
  - 중앙 차선 (폭 없음) 은 lane_id == 0
  - lane_id > 0 은 기준선의 좌측 (도로 s축과 반대 방향 주행)
  - lane_id < 0 은 기준선의 우측 (도로 s축과 같은 방향 주행)

본 모듈이 하류로 제공하는 "운전자 좌표계" 차선 모델의 규약:
  - LaneModel.left_*  는 항상 운전자의 좌측
  - LaneModel.right_* 는 항상 운전자의 우측
  - LaneModel.lane_width 는 전방으로 평활화된 폭
  - LaneModel.centerline_kappa[i] 는 샘플별 부호 있는 Menger 곡률
"""
import math
from statistics import median

import carla

import config
from core.adas_types import LaneBoundary, LaneModel
from core.adas_utils import deg2rad


_LANE_CHANGE_LEFT = {carla.LaneChange.Left, carla.LaneChange.Both}
_LANE_CHANGE_RIGHT = {carla.LaneChange.Right, carla.LaneChange.Both}


# ---------------------------------------------------------------------------
# CARLA frame → 운전자 좌표계 변환 보조 함수
# ---------------------------------------------------------------------------

def _driver_left_right(wp):
    """waypoint 의 좌·우 인접 waypoint 를 운전자 좌표계로 정규화해 반환.

    lane_id > 0 (도로 기준선과 반대 방향 주행) 이면 CARLA 의 "좌" 가 운전자
    의 우측에 해당하므로 결과를 교체한다.

    wp   : 현재 차선의 waypoint
    반환 : (driver_left_wp, driver_right_wp) — 둘 중 하나는 None 가능
    """
    if wp.lane_id > 0:
        # 차량이 도로 기준선과 반대 방향으로 주행 → CARLA "좌" = 운전자 우측.
        return wp.get_right_lane(), wp.get_left_lane()
    return wp.get_left_lane(), wp.get_right_lane()


def _driver_lane_change_allows(wp, driver_direction):
    """OpenDRIVE 방향 기준의 wp.lane_change enum 을 운전자 시점 검사로 변환.

    CARLA 의 lane_change enum 도 get_left_lane / get_right_lane 과 동일한
    OpenDRIVE 방향 규약을 따르므로, lane_id > 0 이면 좌우를 교체한다.

    wp                : 현재 waypoint
    driver_direction  : "LEFT" 또는 "RIGHT" (운전자 좌표계 기준)
    반환              : 해당 방향 LC 가 허용되면 True
    """
    if wp.lane_id > 0:
        od_direction = "RIGHT" if driver_direction == "LEFT" else "LEFT"
    else:
        od_direction = driver_direction
    if od_direction == "LEFT":
        return wp.lane_change in _LANE_CHANGE_LEFT
    if od_direction == "RIGHT":
        return wp.lane_change in _LANE_CHANGE_RIGHT
    return False


# ---------------------------------------------------------------------------
# Polyline walking
# ---------------------------------------------------------------------------

def _polyline_from_waypoints(start_wp, horizon_m, step_m, route_hint_xy=None):
    """start_wp 부터 waypoint.next() 로 horizon_m 까지 전방 walking 해 polyline 산출.

    다음 세 가지 가드를 적용한다.
      - 부호 필터: 반대 방향 lane_id 후보는 거부.
      - "동일 차선" 튜플 보너스: (road_id, section_id, lane_id) 일치 시 강한
        보너스 부여 → section 경계를 넘어도 동일 차선 연속성 우선.
      - Route hint: 분기점에서 계획 경로의 다음 지점에 가까운 후보에 큰 보너스
        → junction 에서는 yaw 일치보다 경로가 선택한 방향을 우선.

    start_wp      : walking 시작점
    horizon_m     : 전방 합산 거리 [m]
    step_m        : segment 호출 step [m]
    route_hint_xy : 전역 경로 preview 점열 (선택). 분기점 안내에 사용.
    반환          : [(x, y), ...] 월드 좌표 polyline
    """
    pts = []
    if start_wp is None:
        return pts
    loc = start_wp.transform.location
    pts.append((loc.x, loc.y))
    current = start_wp
    accumulated = 0.0
    safety_count = 0

    # 임의 s 값에서 보간 가능하도록 hint 의 누적 호 길이 계산.
    hint_cum = []
    if route_hint_xy and len(route_hint_xy) >= 2:
        hint_cum.append(0.0)
        for i in range(1, len(route_hint_xy)):
            ax, ay = route_hint_xy[i - 1]
            bx, by = route_hint_xy[i]
            hint_cum.append(hint_cum[-1] + math.hypot(bx - ax, by - ay))

    def _hint_point_at(arc_s):
        """target arc_s 위치의 hint 점을 선형 보간으로 반환."""
        if not hint_cum:
            return None
        if arc_s <= hint_cum[0]:
            return route_hint_xy[0]
        if arc_s >= hint_cum[-1]:
            return route_hint_xy[-1]
        for i in range(1, len(hint_cum)):
            if hint_cum[i] >= arc_s:
                t = (arc_s - hint_cum[i - 1]) / max(hint_cum[i] - hint_cum[i - 1], 1e-6)
                ax, ay = route_hint_xy[i - 1]
                bx, by = route_hint_xy[i]
                return (ax + t * (bx - ax), ay + t * (by - ay))
        return route_hint_xy[-1]

    while accumulated < horizon_m and safety_count < 2048:
        safety_count += 1
        # junction 근처에서는 후보 판별 robustness 를 위해 더 작은 step 사용.
        # CARLA next() 가 긴 step 한 번에 junction 전체를 건너뛸 수 있다.
        local_step = step_m * 0.5 if current.is_junction else step_m
        nexts = current.next(local_step)
        if not nexts:
            break

        cur_yaw_rad = deg2rad(current.transform.rotation.yaw)
        target_arc = accumulated + local_step
        hint_pt = _hint_point_at(target_arc) if hint_cum else None

        best = None
        best_score = -1e9
        for nxt in nexts:
            # 반대 방향 차선 거부
            if nxt.lane_id * current.lane_id < 0:
                continue
            ny_rad = deg2rad(nxt.transform.rotation.yaw)
            score = math.cos(ny_rad - cur_yaw_rad)
            # (road, section, lane) 동일 → 강한 연속성 보너스
            if (nxt.road_id == current.road_id
                    and nxt.section_id == current.section_id
                    and nxt.lane_id == current.lane_id):
                score += 2.0
            # 동일 road 이나 section 다름 → 더 작은 보너스 (lane ID 가 재번호화
            # 됐을 가능성)
            elif nxt.road_id == current.road_id:
                score += 1.0
            # Route hint: junction 에서 지배적 요소
            if hint_pt is not None:
                nx_loc = nxt.transform.location
                d_hint = math.hypot(nx_loc.x - hint_pt[0], nx_loc.y - hint_pt[1])
                score += 5.0 / (1.0 + d_hint * 0.5)
            if score > best_score:
                best_score = score
                best = nxt
        if best is None:
            break
        current = best
        loc = current.transform.location
        pts.append((loc.x, loc.y))
        accumulated += local_step
    return pts


# ---------------------------------------------------------------------------
# Boundary / curvature 보조 함수
# ---------------------------------------------------------------------------

def _right_unit_at(p_prev, p, p_next):
    """polyline 의 점 p 에서 운전자 우측 (RIGHT) 단위 벡터를 반환.

    CARLA 의 왼손 world frame (world +y 가 차량 우측) 에서, 수학적 +90° CCW
    회전 결과가 시각적으로 우측에 대응한다. 따라서 tangent (dx, dy) 에 대해
    right_unit = (-dy/|t|, dx/|t|) 가 된다. (p_prev, p_next) 가 모두 주어지면
    중앙 차분 tangent 를 써서 급격한 샘플 변화에도 법선이 꺾이지 않는다.

    p_prev / p / p_next : 연속 3 점. 양 끝점은 None 가능.
    반환                : (rx, ry) 단위 벡터
    """
    if p_prev is not None and p_next is not None:
        dx = p_next[0] - p_prev[0]
        dy = p_next[1] - p_prev[1]
    elif p_next is not None:
        dx = p_next[0] - p[0]
        dy = p_next[1] - p[1]
    elif p_prev is not None:
        dx = p[0] - p_prev[0]
        dy = p[1] - p_prev[1]
    else:
        return 0.0, 0.0
    seg = math.hypot(dx, dy)
    if seg < 1e-6:
        return 0.0, 0.0
    return -dy / seg, dx / seg


def _offset_polyline_signed(centerline, offset_m):
    """centerline 을 횡방향으로 offset_m 만큼 오프셋한 polyline 반환.

    양수 offset 은 운전자의 우측 (CARLA world +y) 방향이다. 양 차선 경계
    polyline 산출에 사용된다.
    """
    if len(centerline) < 2:
        return list(centerline)
    out = []
    n = len(centerline)
    for i in range(n):
        prev_p = centerline[i - 1] if i > 0 else None
        next_p = centerline[i + 1] if i + 1 < n else None
        rx, ry = _right_unit_at(prev_p, centerline[i], next_p)
        px = centerline[i][0] + offset_m * rx
        py = centerline[i][1] + offset_m * ry
        out.append((px, py))
    return out


def _menger_curvature(polyline):
    """polyline 의 샘플별 부호 있는 Menger 곡률 산출.

        kappa_i = 4 · Area(P_{i-1}, P_i, P_{i+1}) /
                  (|P_{i-1}P_i| · |P_iP_{i+1}| · |P_{i-1}P_{i+1}|)

    부호는 외적으로 결정 (운전자 우측이 양의 법선 방향인 컨벤션에 맞춰
    우회전이 양수). 양 끝점은 0 이다. 3-tap moving average 로 한 번 평활화
    해 샘플 노이즈를 줄인다.
    """
    n = len(polyline)
    kappa = [0.0] * n
    for i in range(1, n - 1):
        ax, ay = polyline[i - 1]
        bx, by = polyline[i]
        cx, cy = polyline[i + 1]
        # 부호 있는 2배 면적
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        d_ab = math.hypot(bx - ax, by - ay)
        d_bc = math.hypot(cx - bx, cy - by)
        d_ac = math.hypot(cx - ax, cy - ay)
        denom = d_ab * d_bc * d_ac
        if denom < 1e-9:
            kappa[i] = 0.0
            continue
        kappa[i] = 2.0 * cross / denom
    # 3-tap moving average 평활화
    smoothed = list(kappa)
    for i in range(1, n - 1):
        smoothed[i] = (kappa[i - 1] + 2.0 * kappa[i] + kappa[i + 1]) / 4.0
    return smoothed


def _min_lateral_dist(polyline, x, y):
    """polyline 위 모든 점에 대한 (x, y) 까지의 최소 거리.

    classify_object_lane() 에서 객체의 lane 소속 판정에 사용. polyline 이
    비어 있으면 inf 반환.
    """
    if not polyline:
        return float('inf')
    best = float('inf')
    for px, py in polyline:
        d = math.hypot(x - px, y - py)
        if d < best:
            best = d
    return best


# ---------------------------------------------------------------------------
# LaneProvider
# ---------------------------------------------------------------------------

class LaneProvider:
    """매 tick CARLA 맵에서 운전자 좌표계 LaneModel 을 발행한다.

    update() 가 유일한 진입점이며, ego 위치 + 선택적 RouteContext 를 받아
    centerline / 좌·우 인접 차선 / 곡률 / lane width / LC 허용 여부를 한 번에
    채운 LaneModel 한 개를 만든다.
    """

    def __init__(self, carla_map):
        self.map = carla_map
        self.horizon_m = config.LOCAL_PATH_HORIZON_M
        self.step_m = max(1.0, config.LOCAL_PLANNING_SAMPLE_STEP_M)

    def _distance_to_junction_forward(self, wp):
        """현재 waypoint 에서 전방으로 walking 해 다음 junction 까지의 거리 산출.

        _polyline_from_waypoints 와 동일한 동일-차선 후보 우선 규칙을 사용해
        잘못된 분기로 인한 부정확한 짧은 거리 보고를 막는다. junction 이
        horizon × 1.5 m 안에 없으면 inf 반환.
        """
        if wp is None:
            return float('inf')
        cur = wp
        accumulated = 0.0
        steps_max = int(self.horizon_m / max(self.step_m, 0.5)) + 2
        for _ in range(steps_max):
            if cur.is_junction:
                return accumulated
            nxts = cur.next(self.step_m)
            if not nxts:
                return float('inf')
            same = [n for n in nxts
                    if n.lane_id * cur.lane_id > 0
                    and n.road_id == cur.road_id
                    and n.section_id == cur.section_id
                    and n.lane_id == cur.lane_id]
            cur = same[0] if same else next(
                (n for n in nxts if n.lane_id * cur.lane_id > 0), None
            )
            if cur is None:
                return float('inf')
            accumulated += self.step_m
            if accumulated > self.horizon_m * 1.5:
                return float('inf')
        return float('inf')

    def _smoothed_lane_width(self, wp, look_ahead_m=10.0):
        """현재 waypoint 부터 look_ahead_m 까지의 lane_width median 을 반환.

        OpenDRIVE poly3 width 함수의 계단형 점프를 평활화한다. 동일 차선
        후보를 우선해 next() walking 한다.
        """
        widths = [wp.lane_width]
        cur = wp
        accum = 0.0
        for _ in range(20):
            nxts = cur.next(1.0)
            if not nxts:
                break
            cand = [n for n in nxts
                    if n.lane_id == cur.lane_id and n.road_id == cur.road_id]
            cur = cand[0] if cand else nxts[0]
            widths.append(cur.lane_width)
            accum += 1.0
            if accum >= look_ahead_m:
                break
        try:
            return float(median(widths))
        except Exception:
            return float(wp.lane_width)

    def update(self, ego_state, route_context=None):
        """ego 위치 (+ 선택적 route hint) 로부터 한 tick 의 LaneModel 산출.

        다음을 한 번에 채운다.
          - 현재 차선 centerline (전방 horizon_m 까지) + 곡률
          - 평활화된 lane_width + 양 경계 polyline
          - 운전자 좌표계의 좌·우 인접 차선 centerline 과 가용성
          - junction 거리, LC 허용 여부, 차선 마킹 메타데이터

        ego 위치에 waypoint 가 잡히지 않거나 centerline 이 2 점 미만이면
        is_valid = False 인 빈 LaneModel 을 반환한다.

        ego_state     : 자차 상태
        route_context : RouteContext (선택). preview 점열을 walking 분기에
                        route hint 로 사용해 junction 안내가 정확해진다.
        반환          : LaneModel
        """
        loc = carla.Location(x=ego_state.x, y=ego_state.y, z=ego_state.z)
        wp = self.map.get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if wp is None:
            return LaneModel(is_valid=False)

        # Route hint 점열 — junction 분기 안내에 사용.
        route_hint_xy = None
        if route_context is not None and route_context.is_valid:
            route_hint_xy = route_context.preview_waypoints_world

        centerline = _polyline_from_waypoints(
            wp, self.horizon_m, self.step_m, route_hint_xy=route_hint_xy,
        )
        if len(centerline) < 2:
            return LaneModel(is_valid=False)

        lane_width = self._smoothed_lane_width(wp, look_ahead_m=10.0)
        half_w = lane_width * 0.5

        # 양 경계 polyline (양수 offset = 운전자 우측)
        right_boundary_pts = _offset_polyline_signed(centerline, +half_w)
        left_boundary_pts = _offset_polyline_signed(centerline, -half_w)

        # 횡방향 컨트롤러 FF 입력용 샘플별 곡률
        kappa = _menger_curvature(centerline)

        # ----- 인접 차선 (운전자 좌표계) -----
        driver_left_wp, driver_right_wp = _driver_left_right(wp)

        left_centerline = []
        right_centerline = []
        left_available = False
        right_available = False

        if driver_left_wp is not None \
                and driver_left_wp.lane_type == carla.LaneType.Driving \
                and (driver_left_wp.lane_id * wp.lane_id) > 0:
            left_centerline = _polyline_from_waypoints(
                driver_left_wp, self.horizon_m, self.step_m,
                route_hint_xy=route_hint_xy,
            )
            left_available = len(left_centerline) >= 2

        if driver_right_wp is not None \
                and driver_right_wp.lane_type == carla.LaneType.Driving \
                and (driver_right_wp.lane_id * wp.lane_id) > 0:
            right_centerline = _polyline_from_waypoints(
                driver_right_wp, self.horizon_m, self.step_m,
                route_hint_xy=route_hint_xy,
            )
            right_available = len(right_centerline) >= 2

        is_junction = wp.is_junction
        dist_to_junction = self._distance_to_junction_forward(wp)

        # LC 허용 여부도 lane_id 부호에 따라 좌우 교체된다.
        left_change_ok = (
            left_available
            and _driver_lane_change_allows(wp, "LEFT")
            and not is_junction
            and dist_to_junction > 20.0
        )
        right_change_ok = (
            right_available
            and _driver_lane_change_allows(wp, "RIGHT")
            and not is_junction
            and dist_to_junction > 20.0
        )

        # 차선 마킹도 운전자 좌표계로 교체. wp.left_lane_marking 은
        # get_left_lane 과 동일한 OpenDRIVE 방향 규약을 따른다.
        if wp.lane_id > 0:
            driver_left_marking = wp.right_lane_marking
            driver_right_marking = wp.left_lane_marking
        else:
            driver_left_marking = wp.left_lane_marking
            driver_right_marking = wp.right_lane_marking

        def _safe_str(x):
            """marking type / color 의 string 변환을 안전하게 — 실패 시 'Unknown'."""
            try:
                return str(x).split('.')[-1]
            except Exception:
                return "Unknown"

        return LaneModel(
            is_valid=True,
            current_lane_id=wp.lane_id,
            road_id=wp.road_id,
            section_id=wp.section_id,
            lane_width=lane_width,
            is_junction=is_junction,
            distance_to_junction=dist_to_junction,
            centerline=centerline,
            centerline_kappa=kappa,
            left_boundary=LaneBoundary(
                points_xy=left_boundary_pts,
                marking_type=_safe_str(getattr(driver_left_marking, 'type', 'Unknown')),
                marking_color=_safe_str(getattr(driver_left_marking, 'color', 'Unknown')),
                lane_change_allowed=left_change_ok,
            ),
            right_boundary=LaneBoundary(
                points_xy=right_boundary_pts,
                marking_type=_safe_str(getattr(driver_right_marking, 'type', 'Unknown')),
                marking_color=_safe_str(getattr(driver_right_marking, 'color', 'Unknown')),
                lane_change_allowed=right_change_ok,
            ),
            left_centerline=left_centerline,
            right_centerline=right_centerline,
            left_lane_available=left_available,
            right_lane_available=right_available,
            left_lane_change_allowed=left_change_ok,
            right_lane_change_allowed=right_change_ok,
        )

    def classify_object_lane(self, lane, x_world, y_world):
        """world 점 (x, y) 가 어느 차로에 속하는지 분류한다.

        현재 / 좌 / 우 centerline 까지의 최소 거리를 모두 본 뒤, lane.width
        의 절반 + 0.5 m margin 안에 있는 가장 가까운 차로를 반환. 모두 6 m
        밖이면 OUT_OF_ROAD, margin 안이지만 어느 한 차로로 단정 못 하면
        ADJACENT_UNKNOWN.

        lane            : 현재 LaneModel
        x_world, y_world: 검사할 점 (월드 좌표)
        반환            : "CURRENT_LANE" | "LEFT_LANE" | "RIGHT_LANE" |
                          "ADJACENT_UNKNOWN" | "OUT_OF_ROAD"
        """
        margin = lane.lane_width * 0.5 + 0.5
        cur_d = _min_lateral_dist(lane.centerline, x_world, y_world)
        left_d = _min_lateral_dist(lane.left_centerline, x_world, y_world) \
            if lane.left_lane_available else float('inf')
        right_d = _min_lateral_dist(lane.right_centerline, x_world, y_world) \
            if lane.right_lane_available else float('inf')
        best = min(cur_d, left_d, right_d)
        if best == cur_d and cur_d <= margin:
            return "CURRENT_LANE"
        if best == left_d and left_d <= margin:
            return "LEFT_LANE"
        if best == right_d and right_d <= margin:
            return "RIGHT_LANE"
        if min(cur_d, left_d, right_d) > 6.0:
            return "OUT_OF_ROAD"
        return "ADJACENT_UNKNOWN"
