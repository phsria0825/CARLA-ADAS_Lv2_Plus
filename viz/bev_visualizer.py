"""
OpenCV 기반의 통합 visualizer로서, 좌측에서 우측으로 세 개의 panel을 배치한다.

    [ Camera View ] [ BEV ] [ Sidebar ]

  Camera View
    전방 카메라 센서의 실시간 RGB 영상을 BEV 높이에 맞추어 리사이즈한 panel이다.
    각 FusedObject는 3D bounding box로부터 카메라 pinhole 모델을 통해 투영된
    2D bounding box와 소형 track-id label로 오버레이된다. 카메라→world 변환과
    intrinsic은 매 tick마다 전달된다.

  BEV(우측)
    탑다운 ego frame 지도이다. 차선 centerline / 경계, route preview,
    fused object box, 현재 lead 하이라이트, local plan 궤적이 표시되며, 추가로
    실제 CARLA LiDAR로부터 획득한 raw point가 회색 픽셀로 시각화된 LiDAR 점군이
    바닥에 표시된다. 본 점군은 시각적 참조용이며, 추적에는 사용되지 않는다.

  Sidebar
    수치 telemetry(속도, 차선 정보, lead range/TTC, plan 상태, SCC 상태, 제어
    디버그)이다.

BEV panel 내부의 좌표 규약은 다음과 같다.
    +x_ego(차량 전방)  -> 이미지 상단
    +y_ego(차량 우측)  -> 이미지 우측
ego는 panel 하단 1/3 지점에 고정되어 있으며, 이를 통해 전방 약 100 m, 후방/측면
약 30 m 구간을 preview한다.

본 visualizer는 결측에 관대하다. 카메라 영상 또는 LiDAR point가 누락된 경우
해당 부분만 건너뛴다. cv2 자체가 사용 불가능한 경우 전체가 no-op로 동작한다.
"""
import math

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    _HAS_CV2 = False

import numpy as np

import config
from core.adas_types import (
    BehaviorState,
    ControlDebug,
    EgoState,
    FusedObject,
    LaneChangeState,
    LaneModel,
    LeadInfo,
    LeadInfoSet,
    LocalPlan,
    RouteContext,
)
from core.adas_utils import world_to_ego


# --- BEV grid 정의 ---------------------------------------------------------
BEV_X_MIN_M = -30.0
BEV_X_MAX_M = 100.0
BEV_Y_MIN_M = -30.0
BEV_Y_MAX_M = 30.0
BEV_RESOLUTION_M = 0.25

_H = int((BEV_X_MAX_M - BEV_X_MIN_M) / BEV_RESOLUTION_M)
_W = int((BEV_Y_MAX_M - BEV_Y_MIN_M) / BEV_RESOLUTION_M)

# 카메라 뷰 panel의 폭 budget(BEV 높이와 동일)이다. 실제 카메라 이미지는 종횡비를
# 유지하면서 이 폭에 맞게 리사이즈되며, 렌더링된 panel의 폭은 항상 _CAM_VIEW_W로
# 고정되므로 hstack 결과가 결정론적이다.
_CAM_VIEW_W = 720

# LiDAR raw point 오버레이용 설정이다.
_LIDAR_POINT_BGR = (110, 110, 110)
_LIDAR_GROUND_Z_MARGIN = 0.4   # ego-z 기준 본 margin 이내의 점은 도로면으로 간주하여 제외한다.


def _to_pixel(x_ego, y_ego):
    col = int((y_ego - BEV_Y_MIN_M) / BEV_RESOLUTION_M)
    row = _H - 1 - int((x_ego - BEV_X_MIN_M) / BEV_RESOLUTION_M)
    return col, row


def _in_bounds(x_ego, y_ego):
    return (BEV_X_MIN_M <= x_ego <= BEV_X_MAX_M
            and BEV_Y_MIN_M <= y_ego <= BEV_Y_MAX_M)


# 색상 정의(BGR)
_C_BG = (24, 24, 24)
_C_GRID = (48, 48, 48)
_C_EGO = (0, 255, 0)
_C_LANE_CTR = (180, 180, 80)
_C_LANE_BD = (200, 200, 200)
_C_LANE_ADJ = (110, 110, 110)
_C_OBJ = (0, 165, 255)
_C_OBJ_LEAD = (0, 0, 255)
# 인접 차선 lead 강조 색상 (BGR). 좌 = 마젠타, 우 = 시안.
# LeadInfoSet.left / right 의 시각적 디버깅 보조이며, MOBIL 평가가 인접 lead
# 를 어떻게 보고 있는지 한 눈에 확인하기 위한 보조 색상이다.
_C_OBJ_LEAD_LEFT = (255, 0, 200)
_C_OBJ_LEAD_RIGHT = (255, 200, 0)
_C_TRAJ = (255, 200, 0)
# Global route preview 색상이다(BGR; _draw_route는 교대 segment 점선으로 그린다).
_C_ROUTE = (0, 0, 255)
_C_TEXT = (240, 240, 240)
_C_TEXT_WARN = (0, 0, 255)
_C_ROUTE_OK = (0, 220, 220)


def _draw_rotated_box(img, cx, cy, length, width, yaw, color):
    """ego frame 의 (cx, cy) 중심에 length × width 사각형을 yaw 회전해서 그린다.

    객체 box 와 ego box 양쪽에서 동일하게 사용. 4 개 corner 를 회전 변환한
    뒤 cv2.polylines 로 닫힌 다각형을 그린다.
    """
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    hl = length * 0.5
    hw = width * 0.5
    corners_local = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    pts = []
    for lx, ly in corners_local:
        ex = cx + lx * cos_y - ly * sin_y
        ey = cy + lx * sin_y + ly * cos_y
        pts.append(_to_pixel(ex, ey))
    cv2.polylines(img, [np.array(pts, dtype=np.int32)], True, color, 1)


def _project_fused_bbox(
    obj, ego_state, world_to_sensor,
    fx, fy, cx_p, cy_p,
    in_w, in_h, scale, offset_u, disp_w,
):
    """obj 의 3D bbox 8 개 corner 를 핀홀 모델로 투영해 2D rectangle 산출.

    카메라 panel 에 객체의 2D bounding box 를 그리기 위해 사용. 입력 이미지
    의 리사이즈·crop 후 좌표계로 매핑한 (u_min, v_min, u_max, v_max) 를
    반환한다. 객체가 완전히 화면 밖이거나 카메라 후방에 있으면 None.

    obj          : 그릴 FusedObject (length / width 와 world pose 사용)
    ego_state    : 객체의 ground z 결정에 사용
    world_to_sensor : 4×4 변환 행렬
    fx, fy, cx_p, cy_p : 카메라 intrinsic
    in_w, in_h   : 원본 이미지 크기
    scale, offset_u, disp_w : 리사이즈·crop 후 좌표계 매핑 인자
    반환         : (u_min, v_min, u_max, v_max) 또는 None
    """
    # 3D bbox는 obj body frame에서 length × width × height를 가지며,
    # (x_world, y_world, ground_z + height/2)를 중심으로 yaw_world만큼 회전되어 있다.
    L = max(0.5, obj.length) * 0.5
    Wb = max(0.5, obj.width) * 0.5
    Hb = 1.5
    z_ground = ego_state.z
    cos_y = math.cos(obj.yaw_world)
    sin_y = math.sin(obj.yaw_world)

    uvs = []
    for lx_sign in (-1.0, 1.0):
        for ly_sign in (-1.0, 1.0):
            for lz in (0.0, Hb):
                lx = lx_sign * L
                ly = ly_sign * Wb
                wx = obj.x_world + lx * cos_y - ly * sin_y
                wy = obj.y_world + lx * sin_y + ly * cos_y
                wz = z_ground + lz
                p = world_to_sensor @ np.array(
                    [[wx], [wy], [wz], [1.0]], dtype=float)
                sx = float(p[0, 0])
                sy = float(p[1, 0])
                sz = float(p[2, 0])
                if sx <= 0.1:
                    continue
                u = cx_p + fx * sy / sx
                v = cy_p - fy * sz / sx
                uvs.append((u, v))
    if not uvs:
        return None

    u_min_raw = min(u for u, _ in uvs)
    u_max_raw = max(u for u, _ in uvs)
    v_min_raw = min(v for _, v in uvs)
    v_max_raw = max(v for _, v in uvs)
    # 리사이즈 및 crop된 panel 좌표계로 매핑한다.
    u_min = int(round(u_min_raw * scale + offset_u))
    u_max = int(round(u_max_raw * scale + offset_u))
    v_min = int(round(v_min_raw * scale))
    v_max = int(round(v_max_raw * scale))
    # panel 범위로 clipping을 수행한다.
    u_min = max(0, min(disp_w - 1, u_min))
    u_max = max(0, min(disp_w - 1, u_max))
    v_min = max(0, min(_H - 1, v_min))
    v_max = max(0, min(_H - 1, v_max))
    if u_max - u_min < 2 or v_max - v_min < 2:
        return None
    return u_min, v_min, u_max, v_max


def _pick_object_color(track_id, lead_info, adj_leads):
    """객체의 색상을 결정한다 — current / left / right lead 별로 다른 색.

    우선순위: current lead (적색) > left lead (마젠타) > right lead (시안)
    > 일반 객체 (주황). adj_leads 가 None 이면 좌·우 강조는 건너뛰고 일반
    색만 적용된다.

    track_id  : 색상을 결정할 객체의 트랙 id
    lead_info : 현재 차로 LeadInfo
    adj_leads : LeadInfoSet 또는 None
    반환      : BGR 튜플
    """
    if lead_info.detected and lead_info.track_id == track_id:
        return _C_OBJ_LEAD
    if adj_leads is not None:
        if (adj_leads.left is not None
                and adj_leads.left.detected
                and adj_leads.left.track_id == track_id):
            return _C_OBJ_LEAD_LEFT
        if (adj_leads.right is not None
                and adj_leads.right.detected
                and adj_leads.right.track_id == track_id):
            return _C_OBJ_LEAD_RIGHT
    return _C_OBJ


class BEVVisualizer:
    """3-panel OpenCV viewer — [Camera | BEV | Sidebar].

    cv2 가 import 안 되거나 디스플레이가 없는 환경 (headless) 에서는 모든
    update() 호출이 no-op 가 되도록 graceful degradation.
    """

    def __init__(self, window_name="ADAS Lv2+ Camera + BEV"):
        self.window_name = window_name
        self._enabled = _HAS_CV2
        self._warned = False
        if self._enabled:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                # 세 panel 을 좌→우로 배치: camera | BEV | sidebar
                cv2.resizeWindow(self.window_name, _CAM_VIEW_W + _W * 2, _H)
            except Exception as e:  # pragma: no cover - 디스플레이 환경에 종속
                print(f"[BEV] window init failed ({e}); running headless")
                self._enabled = False

    def close(self):
        """cv2 윈도우를 닫는다. 활성 상태가 아니면 no-op."""
        if self._enabled:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass

    # ------------------------------------------------------------------
    def update(
        self,
        ego_state,
        lane,
        fused_objects,
        local_plan,
        lead_info,
        control_debug,
        scc_status="",
        route_context=None,
        # ---- 카메라 + LiDAR feed(모두 optional이며, 누락 시 건너뛴다) ----
        camera_image=None,
        camera_intrinsic=None,
        camera_sensor_to_world=None,
        lidar_raw_points=None,
        lidar_sensor_offset=None,
        # 인접 차선 lead 시각화 (선택). 좌/우 lead 의 track_id 를 색상으로
        # 강조해 MOBIL 평가가 어느 객체를 보고 있는지 확인하기 쉽게 만든다.
        # lead_info 와 일관되게 보통 lead_set.current 와 동일한 객체가
        # lead_info 로 들어 온다.
        adj_leads=None,
    ):
        """한 tick 의 카메라 + BEV + sidebar 통합 패널을 그려 윈도우에 표시한다.

        ego_state         : 자차 상태
        lane              : 현재 LaneModel
        fused_objects     : confirmed FusedObject 리스트
        local_plan        : LocalPlan (trajectory 그리기에 사용)
        lead_info         : 현재 차선 lead (적색 강조)
        control_debug     : ControlDebug (sidebar 텔레메트리)
        scc_status        : SCC 상태 문자열
        route_context     : RouteContext (preview 그리기, 선택)
        camera_image      : 전방 카메라 BGR 이미지 (선택)
        camera_intrinsic  : 카메라 intrinsic dict (선택)
        camera_sensor_to_world : 카메라 pose 4×4 (선택)
        lidar_raw_points  : LiDAR sensor frame raw 점 리스트 (선택)
        lidar_sensor_offset : ego frame 의 LiDAR 마운트 (x, y, z) (선택)
        adj_leads         : LeadInfoSet — 좌/우 lead 색상 강조 (선택)
        """
        if not self._enabled:
            return

        img = np.full((_H, _W, 3), _C_BG, dtype=np.uint8)
        self._draw_grid(img)
        # LiDAR point는 모든 요소의 아래에 배치되어, 점군 위에서 차선/객체/경로의
        # 가독성이 유지되도록 한다.
        if lidar_raw_points:
            self._draw_lidar_points(
                img, lidar_raw_points,
                lidar_sensor_offset or (
                    config.LIDAR_POS_X, config.LIDAR_POS_Y, config.LIDAR_POS_Z,
                ),
                ego_state,
            )
        self._draw_lane(img, ego_state, lane)
        if route_context is not None and route_context.is_valid:
            self._draw_route(img, ego_state, route_context)
        if local_plan is not None and local_plan.trajectory:
            self._draw_trajectory(img, ego_state, local_plan)
        self._draw_objects(img, ego_state, fused_objects, lead_info, adj_leads)
        self._draw_ego(img)
        sidebar = self._draw_sidebar(
            ego_state, lane, lead_info, local_plan, control_debug,
            scc_status, route_context,
        )
        camera_panel = self._render_camera_view(
            camera_image, camera_intrinsic, camera_sensor_to_world,
            ego_state, fused_objects, lead_info, adj_leads,
        )
        canvas = np.hstack([camera_panel, img, sidebar])
        try:
            cv2.imshow(self.window_name, canvas)
            cv2.waitKey(1)
        except Exception as e:  # pragma: no cover
            if not self._warned:
                print(f"[BEV] imshow failed ({e}); disabling")
                self._warned = True
                self._enabled = False

    # ------------------------------------------------------------------
    def _draw_grid(self, img):
        # 10 m 간격 grid를 그린다.
        step = int(10.0 / BEV_RESOLUTION_M)
        for r in range(0, _H, step):
            cv2.line(img, (0, r), (_W - 1, r), _C_GRID, 1)
        for c in range(0, _W, step):
            cv2.line(img, (c, 0), (c, _H - 1), _C_GRID, 1)

    def _draw_lane(self, img, ego, lane):
        if not lane.is_valid:
            return

        def _world_polyline_to_pixels(poly):
            out = []
            for px, py in poly:
                xe, ye = world_to_ego(px, py, ego.x, ego.y, ego.yaw_rad)
                if _in_bounds(xe, ye):
                    out.append(_to_pixel(xe, ye))
            return out

        # centerline(점선)을 그린다.
        cl_px = _world_polyline_to_pixels(lane.centerline)
        for i in range(0, len(cl_px) - 1, 2):
            cv2.line(img, cl_px[i], cl_px[i + 1], _C_LANE_CTR, 1)
        # 경계선을 그린다.
        for bd, color in (
            (lane.left_boundary.points_xy, _C_LANE_BD),
            (lane.right_boundary.points_xy, _C_LANE_BD),
        ):
            pts = _world_polyline_to_pixels(bd)
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1], color, 1)
        # 인접 차선의 centerline을 그린다.
        for adj, color in (
            (lane.left_centerline, _C_LANE_ADJ),
            (lane.right_centerline, _C_LANE_ADJ),
        ):
            pts = _world_polyline_to_pixels(adj)
            for i in range(0, len(pts) - 1, 2):
                cv2.line(img, pts[i], pts[i + 1], color, 1)

    def _draw_route(self, img, ego, route):
        """Global path preview를 빨강 점선 polyline으로 그린다.

        RouteManager가 경로를 약 2 m 해상도로 샘플링하므로, segment를 하나
        걸러 그리면 약 2 m dash, 약 2 m 간격이 형성되어 BEV에서 뚜렷한 점선
        모양이 된다. 라인에는 안티에일리어싱이 적용되고 차선 centerline보다
        약간 두껍게 그려져 global path가 시각적으로 두드러진다.
        """
        pts_px = []
        for px, py in route.preview_waypoints_world:
            xe, ye = world_to_ego(px, py, ego.x, ego.y, ego.yaw_rad)
            if _in_bounds(xe, ye):
                pts_px.append(_to_pixel(xe, ye))
        # segment index를 2 간격으로 진행하면 dash on, dash off가 반복된다.
        for i in range(0, len(pts_px) - 1, 2):
            cv2.line(img, pts_px[i], pts_px[i + 1], _C_ROUTE, 2,
                     lineType=cv2.LINE_AA)
        # 가시 범위에서 가장 먼 route 점에 head marker를 그린다.
        if pts_px:
            cv2.circle(img, pts_px[-1], 4, _C_ROUTE, -1,
                       lineType=cv2.LINE_AA)

    def _draw_trajectory(self, img, ego, plan):
        pts_px = []
        for p in plan.trajectory:
            xe, ye = world_to_ego(p.x, p.y, ego.x, ego.y, ego.yaw_rad)
            if _in_bounds(xe, ye):
                pts_px.append(_to_pixel(xe, ye))
        for i in range(len(pts_px) - 1):
            cv2.line(img, pts_px[i], pts_px[i + 1], _C_TRAJ, 2)

    def _draw_objects(self, img, ego, objs, lead_info, adj_leads=None):
        for o in objs:
            if not _in_bounds(o.x_ego, o.y_ego):
                continue
            color = _pick_object_color(o.track_id, lead_info, adj_leads)
            # 객체의 yaw(ego 기준)에 정렬된 사각형을 그린다.
            _draw_rotated_box(
                img, o.x_ego, o.y_ego, o.length, o.width, o.yaw_ego, color
            )
            # 속도 화살표를 그린다.
            arrow_len = 0.5 * math.hypot(o.vx_ego, o.vy_ego)
            if arrow_len > 0.2:
                tip_x = o.x_ego + o.vx_ego * 0.5
                tip_y = o.y_ego + o.vy_ego * 0.5
                if _in_bounds(tip_x, tip_y):
                    cv2.arrowedLine(
                        img,
                        _to_pixel(o.x_ego, o.y_ego),
                        _to_pixel(tip_x, tip_y),
                        color, 1, tipLength=0.3,
                    )

    def _draw_ego(self, img):
        # ego frame에서 ego는 (0, 0)에 위치한다. 4.7 x 1.85 m 사각형을 그린다.
        _draw_rotated_box(img, 0.0, 0.0, 4.7, 1.85, 0.0, _C_EGO)
        # 방향 표시 tick을 그린다.
        cv2.line(img, _to_pixel(0.0, 0.0), _to_pixel(3.5, 0.0), _C_EGO, 2)

    # ------------------------------------------------------------------
    # BEV 상의 LiDAR raw point 오버레이
    # ------------------------------------------------------------------
    def _draw_lidar_points(
        self,
        img,
        raw_points,           # LiDAR sensor frame 상의 (sx, sy, sz) 리스트
        lidar_offset,         # ego frame에서의 LiDAR 장착 위치 (x, y, z)
        ego_state,
    ):
        """sensor frame -> ego frame -> 픽셀로 변환한 뒤, 회색 점으로 일괄
        도장(bulk paint)한다.

        config 상 LiDAR는 ego에 대해 회전이 없으므로, sensor frame은 ego body
        frame과 translation만큼만 차이가 난다. 또한 점군이 도로면을 가리지
        않도록 지면 인근 반사 점은 필터링한다.
        """
        if not raw_points:
            return
        try:
            arr = np.asarray(raw_points, dtype=np.float32)
        except Exception:
            return
        if arr.ndim != 2 or arr.shape[1] < 3:
            return
        off_x, off_y, off_z = lidar_offset
        xs = arr[:, 0] + off_x
        ys = arr[:, 1] + off_y
        zs = arr[:, 2] + off_z
        # ego_z ± margin 범위(도로면)에 해당하는 점은 제외한다.
        keep = (np.abs(zs - ego_state.z) > _LIDAR_GROUND_Z_MARGIN)
        # BEV window 외부의 점은 제외한다.
        keep &= (xs >= BEV_X_MIN_M) & (xs <= BEV_X_MAX_M)
        keep &= (ys >= BEV_Y_MIN_M) & (ys <= BEV_Y_MAX_M)
        xs = xs[keep]; ys = ys[keep]
        if xs.size == 0:
            return
        cols = ((ys - BEV_Y_MIN_M) / BEV_RESOLUTION_M).astype(np.int32)
        rows = (_H - 1 - ((xs - BEV_X_MIN_M) / BEV_RESOLUTION_M)).astype(np.int32)
        # 일괄 도장(bulk paint)을 수행한다. 수천 개의 점에 대해 점별 cv2.circle 호출보다 훨씬 저렴하다.
        cols = np.clip(cols, 0, _W - 1)
        rows = np.clip(rows, 0, _H - 1)
        img[rows, cols] = _LIDAR_POINT_BGR

    # ------------------------------------------------------------------
    # Camera view panel: 실시간 RGB + 투영된 2D bounding box
    # ------------------------------------------------------------------
    def _render_camera_view(
        self,
        image,
        intrinsic,
        sensor_to_world,
        ego_state,
        fused_objects,
        lead_info,
        adj_leads=None,
    ):
        """(_H, _CAM_VIEW_W, 3) 크기의 BGR panel을 반환한다. 누락된 입력이 있는
        경우, layout이 안정적으로 유지되도록 label이 표시된 검은 placeholder를
        반환한다."""
        if image is None or intrinsic is None or sensor_to_world is None:
            placeholder = np.zeros((_H, _CAM_VIEW_W, 3), dtype=np.uint8)
            cv2.putText(placeholder, "no camera feed",
                        (20, _H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (160, 160, 160), 1, cv2.LINE_AA)
            return placeholder

        # 입력 이미지를 panel에 맞도록 리사이즈한다.
        h_in, w_in = image.shape[:2]
        if h_in == 0 or w_in == 0:
            return np.zeros((_H, _CAM_VIEW_W, 3), dtype=np.uint8)
        # 우선 panel의 HEIGHT에 맞추어 fit한 뒤, 수평 방향으로 중앙 letterbox를 적용한다.
        scale = _H / float(h_in)
        new_w = max(1, int(round(w_in * scale)))
        cam = cv2.resize(image, (new_w, _H), interpolation=cv2.INTER_AREA)
        if new_w > _CAM_VIEW_W:
            # 수평 방향으로 중앙 crop을 수행한다.
            cut = (new_w - _CAM_VIEW_W) // 2
            cam = cam[:, cut:cut + _CAM_VIEW_W]
            offset_u = -cut
            disp_w = _CAM_VIEW_W
        else:
            # 수평 방향으로 padding을 적용한다.
            pad_total = _CAM_VIEW_W - new_w
            left = pad_total // 2
            right = pad_total - left
            cam = cv2.copyMakeBorder(cam, 0, 0, left, right,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
            offset_u = left
            disp_w = _CAM_VIEW_W

        # world->camera 변환을 1회 합성한다.
        try:
            world_to_sensor = np.linalg.inv(sensor_to_world)
        except np.linalg.LinAlgError:
            return cam
        fx = float(intrinsic['fx']); fy = float(intrinsic['fy'])
        cx_p = float(intrinsic['cx']); cy_p = float(intrinsic['cy'])

        for obj in fused_objects:
            color = _pick_object_color(obj.track_id, lead_info, adj_leads)
            box = _project_fused_bbox(
                obj, ego_state, world_to_sensor, fx, fy, cx_p, cy_p,
                w_in, h_in, scale, offset_u, disp_w,
            )
            if box is None:
                continue
            u_min, v_min, u_max, v_max = box
            cv2.rectangle(cam, (u_min, v_min), (u_max, v_max), color, 2)
            label = f"T{obj.track_id}"
            cv2.putText(cam, label, (u_min, max(12, v_min - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # 카메라 panel 상단에 라벨 banner를 표시한다.
        cv2.putText(cam, "Camera View (front)", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
                    cv2.LINE_AA)
        return cam

    def _draw_sidebar(
        self,
        ego,
        lane,
        lead,
        plan,
        debug,
        scc_status,
        route=None,
    ):
        sb = np.full((_H, _W, 3), (32, 32, 32), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 24
        def line(text, color=_C_TEXT):
            nonlocal y
            cv2.putText(sb, text, (10, y), font, 0.45, color, 1, cv2.LINE_AA)
            y += 20

        line("ADAS Lv2+", _C_EGO)
        line(f"speed   : {ego.speed_kmh:5.1f} km/h")
        line(f"set_spd : {config.SET_SPEED:5.1f} km/h")
        line(f"accel   : {ego.accel_mps2:+5.2f} m/s^2")
        line(f"yaw_rate: {math.degrees(ego.yaw_rate_rad_s):+5.1f} deg/s")
        y += 4
        line("--- Route ---", _C_ROUTE)
        if route is not None and route.is_valid:
            line(f"progress: {route.distance_along_route:6.0f} m")
            line(f"remain  : {route.distance_remaining:6.0f} m")
            line(f"to_junc : {route.distance_to_next_junction:5.0f} m")
            if route.mandatory_lane_change_direction:
                tag = (f"LC->{route.mandatory_lane_change_direction} "
                       f"@{route.distance_to_route_lane_change:.0f}m"
                       if route.distance_to_route_lane_change is not None
                       else f"LC->{route.mandatory_lane_change_direction}")
                line(tag, _C_TEXT_WARN)
            if route.is_finished:
                line("ARRIVED", _C_ROUTE_OK)
        else:
            line("no route")
        y += 4
        line("--- Lane ---", _C_LANE_CTR)
        if lane.is_valid:
            line(f"lane_id : {lane.current_lane_id}")
            line(f"width   : {lane.lane_width:4.2f} m")
            line(f"junction: {'YES' if lane.is_junction else 'no'}")
            line(f"to_junc : {lane.distance_to_junction:5.1f} m")
            line(f"L avail : {'Y' if lane.left_lane_available else 'n'}  LC:{'Y' if lane.left_lane_change_allowed else 'n'}")
            line(f"R avail : {'Y' if lane.right_lane_available else 'n'}  LC:{'Y' if lane.right_lane_change_allowed else 'n'}")
        else:
            line("INVALID", _C_TEXT_WARN)
        y += 4
        line("--- Lead ---", _C_OBJ)
        if lead.detected:
            line(f"track   : {lead.track_id}")
            line(f"range   : {lead.range:5.1f} m")
            line(f"d_range : {lead.range_rate:+5.2f} m/s")
            ttc_str = f"{lead.ttc:5.2f} s" if math.isfinite(lead.ttc) else "  inf"
            line(f"TTC     : {ttc_str}")
            line(f"lead_v  : {lead.lead_speed_mps * 3.6:5.1f} km/h")
        else:
            line("no lead")
        y += 4
        line("--- Plan ---", _C_TRAJ)
        if plan is not None:
            line(f"behavior: {plan.behavior_state.value}")
            line(f"LC state: {plan.lane_change_state.value}")
            line(f"des_spd : {plan.desired_speed_kmh:5.1f} km/h")
            if not plan.is_safe:
                line(f"unsafe  : {plan.safety_reason}", _C_TEXT_WARN)
        y += 4
        line("--- Ctrl ---")
        line(f"SCC     : {scc_status[:24]}")
        line(f"steer   : {debug.steer:+5.3f}")
        # LQR mode에서 debug.delta_pp는 Ackermann 곡률 FF를 보유하며, PP-only
        # fallback에서는 실제 PP 명령을 보유한다. HMI는 이를 일관되게 feedforward
        # 성분으로 표시한다.
        ff_label = "FF      " if abs(debug.delta_lqr) > 1e-6 else "PP      "
        line(f"{ff_label}: {debug.delta_pp:+5.3f}")
        line(f"LQR FB  : {debug.delta_lqr:+5.3f}")
        line(f"e_y     : {debug.e_y:+5.3f} m")
        line(f"e_psi   : {math.degrees(debug.e_psi):+5.1f} deg")
        line(f"throttle: {debug.throttle:4.2f}")
        line(f"brake   : {debug.brake:4.2f}")
        return sb
