"""
Global route 관리 모듈.

CARLA 의 GlobalRoutePlanner 를 wrapping 해 출발 위치 → 목적지의 고정된
waypoint 시퀀스를 생성하고, 매 tick ego 를 이 시퀀스에 재투영한다. 출력은
RouteContext 스냅샷이며, lane provider (junction 안내), local planner
(mandatory LC intent), BEV (preview) 가 소비한다.

CARLA 의 GlobalRoutePlanner 는 sample_resolution 이 기본 수 미터인 (Waypoint,
RoadOption) 쌍의 리스트를 반환한다. 본 모듈은 이를 다음과 같이 재샘플링 +
해석한다.

    - preview_waypoints_world : ego 전방 약 100 m 의 (x, y) 점열. BEV /
                                planner 가 소비.
    - target_lane_id          : 현재 도로 구간에서 경로가 지시하는 lane_id.
    - mandatory_lane_change_direction :
        동일 road 상에서 경로의 lane_id 가 L_now → L_next 로 바뀌면
        mandatory LC. 방향은 (L_next - L_now) 의 부호로 추론. CARLA 의
        lane_id 가 우측에서는 도로 중심에서 멀어질수록 증가하고 좌측에서는
        감소하므로, lane_id 부호로 운전자 좌·우를 정렬한다.

경로는 시작 시 1 회만 생성된다. 목적지 ARRIVAL_RADIUS_M 이내에 도달하면
RouteContext.is_finished 가 설정되어, planner 가 목적지 통과 후 충돌 대신
MIN_RISK 정지를 트리거할 수 있다.
"""
import math

import carla

from core.adas_types import RouteContext


# ----------------------------------------------------------------------------
# GlobalRoutePlanner import — 경로는 CARLA 버전에 따라 다름
# ----------------------------------------------------------------------------
# agents.navigation.global_route_planner 모듈은 CARLA 의 PythonAPI/carla/agents/...
# 경로에 함께 배포된다. 기본 상태에서 해당 디렉터리는 PYTHONPATH 에 포함되어
# 있지 않으며, 사용자가 직접 추가하거나 본 모듈이 자동 탐지한다.
_GRP_CLS = None
_GRP_IMPORT_ERROR = None


def _find_and_inject_carla_pythonapi():
    """일반적인 CARLA 설치 경로를 탐색해 PythonAPI/carla 디렉터리를 sys.path 에 추가.

    환경 변수 (CARLA_ROOT 등) → 표준 Windows 경로 → cwd 의 상위 디렉터리
    순으로 시도한다. 추가된 경로를 반환하며, 찾지 못하면 None.

    반환 : 성공 시 추가된 경로 문자열, 실패 시 None
    """
    import os
    import sys
    candidates = []
    # 1) 환경 변수 hint
    for env_var in ("CARLA_ROOT", "CARLA_DIR", "CARLA_PATH"):
        v = os.environ.get(env_var)
        if v:
            candidates.append(os.path.join(v, "PythonAPI", "carla"))
    # 2) 표준 Windows 설치 경로
    candidates.extend([
        r"C:\CARLA_0.9.15\PythonAPI\carla",
        r"C:\CARLA_0.9.16\PythonAPI\carla",
        r"C:\Program Files\CARLA_0.9.15\PythonAPI\carla",
        r"C:\Carla\PythonAPI\carla",
    ])
    # 3) cwd 상위 디렉터리 탐색
    cwd = os.path.abspath(os.getcwd())
    cur = cwd
    for _ in range(6):
        cand = os.path.join(cur, "PythonAPI", "carla")
        candidates.append(cand)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        target = os.path.join(c, "agents", "navigation", "global_route_planner.py")
        if os.path.isfile(target):
            if c not in sys.path:
                sys.path.insert(0, c)
            return c
    return None


try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner as _GRP_CLS  # type: ignore
except Exception:
    _added = _find_and_inject_carla_pythonapi()
    if _added is not None:
        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner as _GRP_CLS  # type: ignore
        except Exception as e:
            _GRP_IMPORT_ERROR = (
                f"agents.navigation found at {_added} but import still failed: {e}"
            )
    else:
        _GRP_IMPORT_ERROR = (
            "agents.navigation not found on PYTHONPATH or any common CARLA "
            "install location. Set CARLA_ROOT env var or run from the CARLA "
            "PythonAPI directory."
        )


ARRIVAL_RADIUS_M = 8.0
PREVIEW_HORIZON_M = 120.0


def _wp_xy(wp):
    """carla.Waypoint → (x, y) 월드 좌표 튜플."""
    loc = wp.transform.location
    return (loc.x, loc.y)


class RouteManager:
    """글로벌 경로를 1 회 trace 한 뒤 매 tick ego 진행도를 RouteContext 로 발행.

    set_route(start, end) 호출 후 반복적으로 update(ego_state) 를 부르면,
    ego 가 경로의 어느 지점에 있는지 / 도착 여부 / 다음 junction 거리 /
    mandatory LC 방향 등을 한 묶음으로 산출한다.
    """

    def __init__(self, world, sample_resolution_m=2.0):
        if _GRP_CLS is None:
            raise RuntimeError(
                f"agents.navigation.global_route_planner not importable. "
                f"Make sure CARLA's PythonAPI/carla/agents is on PYTHONPATH. "
                f"Original error: {_GRP_IMPORT_ERROR}"
            )
        self.world = world
        self.map = world.get_map()
        self.grp = _GRP_CLS(self.map, sample_resolution_m)
        self.sample_resolution_m = sample_resolution_m

        self._route = []
        self._route_xy = []
        self._cum_arc = []   # 누적 호 길이, route 와 동일 길이
        self._current_index = 0
        self.destination = None

    def set_route(self, start, end):
        """start → end 경로를 GlobalRoutePlanner 로 1 회 생성해 내부에 캐시.

        이후 tick 에서는 update() 가 ego 를 이 고정 plan 에 재투영할 뿐
        새 경로는 만들지 않는다.

        start  : 시작 위치 (carla.Location)
        end    : 목적지 위치 (carla.Location)
        반환   : 생성된 waypoint 개수 (int)
        """
        route = self.grp.trace_route(start, end)
        if not route:
            raise RuntimeError("GlobalRoutePlanner returned empty route")
        self._route = route
        self.destination = end
        self._route_xy = [_wp_xy(wp) for wp, _ in route]
        # 누적 호 길이 산출
        self._cum_arc = [0.0] * len(route)
        for i in range(1, len(route)):
            dx = self._route_xy[i][0] - self._route_xy[i - 1][0]
            dy = self._route_xy[i][1] - self._route_xy[i - 1][1]
            self._cum_arc[i] = self._cum_arc[i - 1] + math.hypot(dx, dy)
        self._current_index = 0
        return len(route)

    @property
    def total_length(self):
        """전체 경로 길이 [m]. 경로가 비어 있으면 0."""
        return self._cum_arc[-1] if self._cum_arc else 0.0

    @property
    def route_waypoints(self):
        """경로의 모든 carla.Waypoint 만 리스트로 반환 (RoadOption 제외)."""
        return [wp for wp, _ in self._route]

    def update(self, ego_state):
        """한 tick 의 RouteContext 산출 — ego 재투영 + 진행도 / mandatory LC 추론.

        다음을 한 번에 채운다.
          1) ego 를 경로에 재투영해 current_index 와 진행/잔여 거리 산출
          2) 목적지 ARRIVAL_RADIUS_M 안에 들어왔는지 도착 판정
          3) 전방 preview 점열 (BEV / lane provider 용)
          4) 다음 junction 까지의 거리
          5) target_lane_id 와 mandatory_lane_change_direction (동일 road 안
             에서 경로의 lane_id 변화로부터 추론)

        ego_state : 자차 상태
        반환      : RouteContext. 경로가 비어 있으면 is_valid=False.
        """
        if not self._route:
            return RouteContext(is_valid=False)

        # 1) ego 재투영 — 노이즈가 큰 localization spike 가 plan 전체를
        #    거슬러 snap 되는 것을 막기 위해 현재 index 주변 window 에서만 탐색.
        self._current_index = self._closest_index(
            ego_state.x, ego_state.y, search_radius=40
        )
        cur_wp = self._route[self._current_index][0]

        # 2) 진행 / 잔여 거리
        s_here = self._cum_arc[self._current_index]
        total = self.total_length
        dist_remaining = max(0.0, total - s_here)

        # 3) 도착 여부
        is_finished = False
        if self.destination is not None:
            dx = ego_state.x - self.destination.x
            dy = ego_state.y - self.destination.y
            if math.hypot(dx, dy) <= ARRIVAL_RADIUS_M:
                is_finished = True

        # 4) Preview window — ego 전방 horizon 까지의 점열
        preview = self._preview_points(self._current_index, PREVIEW_HORIZON_M)

        # 5) Junction 거리
        dist_to_junction = self._distance_to_next_junction(self._current_index)

        # 6) Target lane / mandatory LC 방향
        target_lane_id = cur_wp.lane_id
        mandatory_lc_dir = None
        dist_to_lc = None
        next_road_id = cur_wp.road_id

        # 경로를 따라 look-ahead 하며 다음 두 가지 중 하나가 발생하는 지점까지 진행:
        # (a) road_id 변화 (junction 전환)
        # (b) 동일 road 상에서 lane_id 변화 (mandatory LC)
        cum_d = 0.0
        prev_xy = (ego_state.x, ego_state.y)
        for j in range(self._current_index + 1, len(self._route)):
            wp_j = self._route[j][0]
            x_j, y_j = self._route_xy[j]
            cum_d += math.hypot(x_j - prev_xy[0], y_j - prev_xy[1])
            prev_xy = (x_j, y_j)
            if cum_d > PREVIEW_HORIZON_M:
                break
            if wp_j.road_id != cur_wp.road_id:
                next_road_id = wp_j.road_id
                break
            if wp_j.lane_id != cur_wp.lane_id:
                # 동일 road 에서 차선 변경 필요
                if cur_wp.lane_id * wp_j.lane_id <= 0:
                    # 중심선을 가로지름 → junction 급 상황으로 보고 종료
                    break
                if abs(wp_j.lane_id) < abs(cur_wp.lane_id):
                    # 도로 중심선 방향으로 이동
                    mandatory_lc_dir = "LEFT" if cur_wp.lane_id > 0 else "RIGHT"
                else:
                    mandatory_lc_dir = "RIGHT" if cur_wp.lane_id > 0 else "LEFT"
                target_lane_id = wp_j.lane_id
                dist_to_lc = cum_d
                break

        return RouteContext(
            is_valid=True,
            is_finished=is_finished,
            preview_waypoints_world=preview,
            target_lane_id=target_lane_id,
            next_road_id=next_road_id,
            distance_to_next_junction=dist_to_junction,
            distance_to_route_lane_change=dist_to_lc,
            mandatory_lane_change_direction=mandatory_lc_dir,
            distance_along_route=s_here,
            distance_remaining=dist_remaining,
            total_route_length=total,
        )

    def _closest_index(self, x, y, search_radius):
        """현재 index 주변 window 내에서 (x, y) 에 가장 가까운 경로 점 인덱스 반환.

        먼저 [current-5, current+search_radius] window 에서 검색. 거기서
        100 m 이상 벗어나 있으면 (ego 가 경로 이탈) safety net 으로 전체
        scan 수행.
        """
        lo = max(0, self._current_index - 5)
        hi = min(len(self._route), self._current_index + search_radius)
        best_i = self._current_index
        best_d2 = float('inf')
        for i in range(lo, hi):
            rx, ry = self._route_xy[i]
            d2 = (rx - x) ** 2 + (ry - y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        if best_d2 > 100.0 ** 2:
            for i in range(0, len(self._route)):
                rx, ry = self._route_xy[i]
                d2 = (rx - x) ** 2 + (ry - y) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_i = i
        return best_i

    def _preview_points(self, start_idx, horizon_m):
        """start_idx 부터 horizon_m 까지의 경로 점열을 (x, y) 리스트로 반환."""
        out = []
        if start_idx >= len(self._route_xy):
            return out
        out.append(self._route_xy[start_idx])
        accum = 0.0
        for i in range(start_idx + 1, len(self._route_xy)):
            px, py = self._route_xy[i]
            ax, ay = out[-1]
            accum += math.hypot(px - ax, py - ay)
            out.append((px, py))
            if accum >= horizon_m:
                break
        return out

    def _distance_to_next_junction(self, start_idx):
        """start_idx 부터 경로를 따라 다음 junction 까지의 누적 거리. 없으면 inf."""
        accum = 0.0
        for j in range(start_idx + 1, len(self._route)):
            wp_j = self._route[j][0]
            if wp_j.is_junction:
                return accum
            x_a, y_a = self._route_xy[j - 1]
            x_b, y_b = self._route_xy[j]
            accum += math.hypot(x_b - x_a, y_b - y_a)
            if accum > PREVIEW_HORIZON_M * 2:
                return float('inf')
        return float('inf')


# ----------------------------------------------------------------------------
# 목적지 선택 보조 (메인 루프 사용)
# ----------------------------------------------------------------------------

def pick_destination_spawn(
    world,
    ego_location,
    min_distance_m=250.0,
    max_distance_m=1500.0,
):
    """ego 로부터 [min, max] 거리 안에서 가장 먼 spawn point 를 목적지로 선택.

    경로가 너무 짧으면 의미 있는 LC / 추월 시나리오가 안 만들어지므로 일정
    거리 이상을 요구한다. 적합한 후보가 없으면 fallback 으로 ego 와 충분히
    떨어진 임의 spawn point 를 반환하고, 그것도 없으면 None.

    world         : carla.World
    ego_location  : 자차 위치 (carla.Location)
    min_distance_m: 목적지가 ego 로부터 가져야 할 최소 거리 [m]
    max_distance_m: 최대 거리 [m]
    반환          : carla.Transform 또는 None
    """
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        return None
    best_sp = None
    best_score = -1.0
    for sp in spawn_points:
        dx = sp.location.x - ego_location.x
        dy = sp.location.y - ego_location.y
        d = math.hypot(dx, dy)
        if d < min_distance_m or d > max_distance_m:
            continue
        if d > best_score:
            best_score = d
            best_sp = sp
    if best_sp is None:
        # fallback — ego 와 충분히 떨어진 임의 spawn point
        for sp in spawn_points:
            if abs(sp.location.x - ego_location.x) + abs(sp.location.y - ego_location.y) > 20.0:
                return sp
    return best_sp
