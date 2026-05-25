"""
ADAS Lv2+ 스택 전반에서 사용되는 수학 / 기하 헬퍼.

CARLA 의존성을 배제하기 위해 numpy 외 외부 의존성을 두지 않는다. 따라서
controller, planner, visualizer 등에서 CARLA 를 끌어들이지 않고 자유롭게
import 할 수 있다.
"""
import math

import numpy as np


def clamp(value, lo, hi):
    """``value`` 를 [lo, hi] 구간으로 잘라낸다.

    value : 자를 값
    lo    : 하한
    hi    : 상한
    반환  : lo ≤ x ≤ hi 를 만족하는 값
    """
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def normalize_angle(angle):
    """입력 각도를 반열린 구간 [-π, π) 로 wrap 한다.

    angle : 임의의 라디안 값
    반환  : 동일한 회전을 나타내는 [-π, π) 안의 라디안
    """
    a = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if a <= -math.pi:
        a += 2.0 * math.pi
    return a


def deg2rad(deg):
    """도(degree) → 라디안 변환."""
    return deg * math.pi / 180.0


def rad2deg(rad):
    """라디안 → 도(degree) 변환."""
    return rad * 180.0 / math.pi


def lerp(a, b, t):
    """두 값 ``a``, ``b`` 사이를 ``t`` 비율(0~1)로 선형 보간한다."""
    return a + (b - a) * t


def distance_xy(ax, ay, bx, by):
    """평면 두 점 (ax, ay), (bx, by) 사이의 유클리드 거리."""
    return math.hypot(bx - ax, by - ay)


def world_to_ego(px, py, ego_x, ego_y, ego_yaw_rad):
    """월드 좌표계 점을 ego 차량 좌표계로 변환한다.

    본 스택 전반에서 사용하는 ego 좌표계 컨벤션:
        x : 차량 전방
        y : 차량 우측 (CARLA 의 left-handed Y 와 일치)
        yaw: world +x 에서 ego +x 로의 회전 (CCW 양수)

    px, py             : 월드 좌표계 점
    ego_x, ego_y       : ego 위치 (월드 좌표)
    ego_yaw_rad        : ego heading [rad]
    반환               : (fx, fy) — ego 전방·우측 거리 [m]
    """
    dx = px - ego_x
    dy = py - ego_y
    cos_y = math.cos(-ego_yaw_rad)
    sin_y = math.sin(-ego_yaw_rad)
    fx = dx * cos_y - dy * sin_y
    fy = dx * sin_y + dy * cos_y
    return fx, fy


def ego_to_world(fx, fy, ego_x, ego_y, ego_yaw_rad):
    """ego 좌표계 점을 월드 좌표계로 변환한다.

    fx, fy        : ego 전방·우측 거리 [m]
    ego_x, ego_y  : ego 위치 (월드 좌표)
    ego_yaw_rad   : ego heading [rad]
    반환          : (wx, wy) — 월드 좌표계 점
    """
    cos_y = math.cos(ego_yaw_rad)
    sin_y = math.sin(ego_yaw_rad)
    wx = ego_x + fx * cos_y - fy * sin_y
    wy = ego_y + fx * sin_y + fy * cos_y
    return wx, wy


def signed_lateral_offset(px, py, ref_x, ref_y, ref_yaw_rad):
    """기준 pose 의 전방축에 대한 점 (px, py) 까지의 부호 있는 횡거리.

    양수이면 점이 기준 heading 의 우측에 위치함을 의미한다. CARLA 의
    left-handed 월드 좌표계에서 우측 단위벡터는 (sin(yaw), -cos(yaw)) 이다.

    px, py        : 검사할 점 (월드 좌표)
    ref_x, ref_y  : 기준 pose 위치 (월드 좌표)
    ref_yaw_rad   : 기준 heading [rad]
    반환          : 부호 있는 횡거리 [m] (우측 양수)
    """
    dx = px - ref_x
    dy = py - ref_y
    right_x = math.sin(ref_yaw_rad)
    right_y = -math.cos(ref_yaw_rad)
    return dx * right_x + dy * right_y


def project_onto_polyline(points, px, py):
    """점 (px, py) 를 polyline 에 최단 거리로 투영한다.

    각 segment 에 대해 점의 local 파라미터 t 를 구해 가장 가까운 segment 를
    찾고, 그 segment 의 시작점에서 투영점까지의 호 길이까지 누적해 반환한다.
    polyline 이 비어 있으면 모두 0 으로 반환한다.

    points : [(x, y), ...] 형태의 점열 (월드 좌표)
    px, py : 투영할 점
    반환   : (seg_index, t_clamped, s)
             - seg_index : 가장 가까운 segment 의 시작점 인덱스
             - t_clamped : 그 segment 상의 국부 파라미터 [0, 1]
             - s         : points[0] 부터 투영점까지의 누적 호 길이 [m]
    """
    pts = list(points)
    if len(pts) < 2:
        if not pts:
            return 0, 0.0, 0.0
        return 0, 0.0, 0.0

    best_i = 0
    best_t = 0.0
    best_d2 = float('inf')
    best_proj_x = pts[0][0]
    best_proj_y = pts[0][1]

    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        sx = bx - ax
        sy = by - ay
        seg_len2 = sx * sx + sy * sy
        if seg_len2 < 1e-9:
            continue
        t = ((px - ax) * sx + (py - ay) * sy) / seg_len2
        t_clamped = clamp(t, 0.0, 1.0)
        proj_x = ax + sx * t_clamped
        proj_y = ay + sy * t_clamped
        dx = px - proj_x
        dy = py - proj_y
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
            best_t = t_clamped
            best_proj_x = proj_x
            best_proj_y = proj_y

    s = 0.0
    for i in range(best_i):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        s += math.hypot(bx - ax, by - ay)
    ax, ay = pts[best_i]
    s += math.hypot(best_proj_x - ax, best_proj_y - ay)
    return best_i, best_t, s


def polyline_heading_at(points, idx):
    """polyline 의 segment ``idx`` 시작점에서의 heading 을 반환한다.

    points : [(x, y), ...] 점열
    idx    : segment 시작점 인덱스. 끝점이면 직전 segment 의 heading 으로
             fallback 한다.
    반환   : atan2(dy, dx) 형태의 heading [rad]
    """
    pts = list(points)
    if len(pts) < 2:
        return 0.0
    i = clamp(idx, 0, len(pts) - 2)
    i = int(i)
    ax, ay = pts[i]
    bx, by = pts[i + 1]
    return math.atan2(by - ay, bx - ax)


def lateral_error_to_polyline(points, px, py, yaw_rad):
    """polyline 기준으로 (px, py, yaw_rad) 의 부호 있는 횡오차와 heading 오차.

    CARLA 의 left-handed 좌표계에서는 수학 좌표계 기준 cross product 부호가
    시각적 "좌측" 과 반대이므로, segment 직선에 대한 부호 있는 수직거리를
    그대로 RIGHT-positive 오차로 사용할 수 있다.

    부호 컨벤션:
        e_y > 0  → ego 가 경로의 우측에 있음 → 좌측 조향으로 보정
        e_y < 0  → ego 가 경로의 좌측에 있음 → 우측 조향으로 보정
        e_psi    → normalize_angle(yaw_ego - yaw_ref)

    LQR 피드백 항은 u = -K @ x_err 을 사용하며 K[0,0] > 0 이므로, 양의 e_y 가
    음의 조향 명령(좌회전) 을 생성해 올바른 방향으로 보정된다.

    points  : 추종 polyline (월드 좌표)
    px, py  : ego 위치 (월드 좌표)
    yaw_rad : ego heading [rad]
    반환    : (e_y, e_psi, seg_i) — 횡오차 [m], heading 오차 [rad],
              가장 가까운 segment 의 인덱스
    """
    pts = list(points)
    seg_i, _t, _s = project_onto_polyline(pts, px, py)
    ref_yaw = polyline_heading_at(pts, seg_i)
    ax, ay = pts[seg_i]
    bx, by = pts[min(seg_i + 1, len(pts) - 1)]
    sx = bx - ax
    sy = by - ay
    cross = sx * (py - ay) - sy * (px - ax)
    seg_len = math.hypot(sx, sy)
    if seg_len < 1e-6:
        return 0.0, 0.0, seg_i
    e_y_right_positive = cross / seg_len
    e_psi = normalize_angle(yaw_rad - ref_yaw)
    return e_y_right_positive, e_psi, seg_i


def kmh_to_mps(kmh):
    """km/h → m/s 변환."""
    return kmh / 3.6


def mps_to_kmh(mps):
    """m/s → km/h 변환."""
    return mps * 3.6
