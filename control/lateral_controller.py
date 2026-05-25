"""
횡방향 제어 — Ackermann 곡률 feedforward + 동역학 bicycle LQR feedback.

조향 명령은 두 항의 합으로 산출한다.

    δ_cmd = δ_ff + δ_lqr
    δ_ff  = arctan(L · κ_ref)             (운동학적 Ackermann FF)
    δ_lqr = -K · x_err                    (x_err = [e_y, e_ẏ, e_ψ, e_ψ̇])

LQR 게인 K 는 동역학 bicycle 모델의 A 행렬이 속도 v 에 의존하므로 매 tick
재계산된다 (20 Hz × 4×4 DARE 풀이가 수십 µs 라 부담 없음). DARE 풀이는
가능하면 scipy.linalg.solve_discrete_are 를 우선 사용하며, scipy 가 없을
경우 Kleinman fixed-point 반복으로 fallback 한다. ZOH 이산화도 scipy.expm
을 우선 사용하고 없으면 forward Euler 로 fallback 한다.
"""
import math

import numpy as np

try:
    from scipy.linalg import expm as _scipy_expm
    from scipy.linalg import solve_discrete_are as _scipy_dare
    _HAS_SCIPY = True
except Exception:
    _scipy_expm = None
    _scipy_dare = None
    _HAS_SCIPY = False

import config
from core.adas_types import ControlDebug
from core.adas_utils import (
    clamp,
    normalize_angle,
    polyline_heading_at,
    project_onto_polyline,
)


def _solve_dare(A, B, Q, R, max_iter=200, tol=1e-6):
    """이산 대수 Riccati 방정식 (DARE) 의 해 P 를 구한다.

    scipy 가 있으면 solve_discrete_are (Schur 분해 기반) 를 우선 사용해
    stiff 한 동역학 bicycle A 행렬에 대해서도 수치적으로 안정적인 해를
    얻는다. scipy 가 없으면 Kleinman fixed-point 반복으로 fallback 한다.

    A, B    : 이산 상태공간 행렬
    Q, R    : LQR 비용 가중치
    max_iter: Kleinman 반복 횟수 한계
    tol     : 수렴 임계 (max element-wise 차이)
    반환    : 해 P. 풀이 실패 시 None.
    """
    if _HAS_SCIPY:
        try:
            return _scipy_dare(A, B, Q, R)
        except Exception:
            pass  # 반복 solver 로 fall through
    P = Q.copy()
    for _ in range(max_iter):
        BPB = B.T @ P @ B
        BPA = B.T @ P @ A
        try:
            K = np.linalg.solve(R + BPB, BPA)
        except np.linalg.LinAlgError:
            return None
        P_next = A.T @ P @ A - A.T @ P @ B @ K + Q
        if np.max(np.abs(P_next - P)) < tol:
            return P_next
        P = P_next
    return P


def _zoh_discretize(Ac, Bc, dt):
    """연속 시간 (Ac, Bc) 를 dt 구간에 대해 zero-order hold 이산화한다.

    scipy 가 있으면 블록 행렬 expm 기법 (정확한 ZOH) 을 사용하고, 없으면
    forward Euler 로 fallback 한다. 동역학 bicycle 모델은 stiff 한 경향
    이라 가능하면 expm 쪽이 정확하다.

    Ac, Bc : 연속 시간 상태공간 행렬
    dt     : 시간 간격 [s]
    반환   : (Ad, Bd) 이산 시간 상태공간 행렬
    """
    if _HAS_SCIPY and _scipy_expm is not None:
        n = Ac.shape[0]
        m = Bc.shape[1]
        M = np.zeros((n + m, n + m))
        M[:n, :n] = Ac
        M[:n, n:] = Bc
        Mexp = _scipy_expm(M * dt)
        Ad = Mexp[:n, :n]
        Bd = Mexp[:n, n:]
        return Ad, Bd
    # Forward Euler fallback
    Ad = np.eye(Ac.shape[0]) + Ac * dt
    Bd = Bc * dt
    return Ad, Bd


class LateralController:
    """Ackermann FF + 동역학 LQR FB 조향 제어기.

    compute() 가 유일한 진입점이며, ego 와 trajectory 를 받아 한 tick 의
    steer 명령과 ControlDebug 를 반환한다. trajectory 가 비어 있거나 점이
    2 개 미만이면 직전 steer 를 그대로 반환해 과도기 0 출력을 막는다.
    """

    _MIN_SPEED_FOR_LQR_MPS = 1.5

    def __init__(self):
        self._last_steer = 0.0
        self._last_K = None
        # 차량 동역학 파라미터 (Tesla Model 3 근사값)
        self.m = config.VEHICLE_MASS
        self.Iz = config.VEHICLE_IZ
        self.lf = config.VEHICLE_LF
        self.lr = config.VEHICLE_LR
        self.Cf = config.VEHICLE_CF
        self.Cr = config.VEHICLE_CR
        self.L = config.WHEELBASE
        # LQR cost 가중치는 상수이지만 Ac 가 v 에 의존하므로 K 는 매 tick 재계산.
        self.Q = np.asarray(config.LQR_Q, dtype=float)
        self.R = np.asarray(config.LQR_R, dtype=float)
        if self.Q.shape != (4, 4):
            raise ValueError(
                "config.LQR_Q must be 4x4 for the dynamic bicycle error model"
            )
        if self.R.shape != (1, 1):
            raise ValueError("config.LQR_R must be 1x1")

    def _solve_K(self, v, dt):
        """현재 속도 v 에서의 LQR 게인 K (shape (1, 4)) 를 산출한다.

        동역학 bicycle 모델의 4-상태 오차 시스템 (e_y, e_y_dot, e_psi,
        e_psi_dot) 을 ZOH 이산화한 뒤 DARE 를 풀어 K = (R + Bᵀ P B)⁻¹ Bᵀ P A
        로 얻는다. 풀이 실패 시 None.

        v   : 현재 ego 종방향 속도 [m/s] (최소 1.5 m/s 로 clamp)
        dt  : 시간 간격 [s]
        반환: shape (1, 4) 게인 행렬, 풀이 실패 시 None
        """
        v = max(v, self._MIN_SPEED_FOR_LQR_MPS)
        m, Iz, lf, lr, Cf, Cr = self.m, self.Iz, self.lf, self.lr, self.Cf, self.Cr

        a22 = -(2 * Cf + 2 * Cr) / (m * v)
        a23 = (2 * Cf + 2 * Cr) / m
        a24 = (-2 * Cf * lf + 2 * Cr * lr) / (m * v)
        a42 = -(2 * Cf * lf - 2 * Cr * lr) / (Iz * v)
        a43 = (2 * Cf * lf - 2 * Cr * lr) / Iz
        a44 = -(2 * Cf * lf * lf + 2 * Cr * lr * lr) / (Iz * v)

        Ac = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, a22, a23, a24],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, a42, a43, a44],
        ])
        Bc = np.array([
            [0.0],
            [2 * Cf / m],
            [0.0],
            [2 * Cf * lf / Iz],
        ])
        Ad, Bd = _zoh_discretize(Ac, Bc, dt)
        P = _solve_dare(Ad, Bd, self.Q, self.R)
        if P is None:
            return None
        try:
            K = np.linalg.solve(self.R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
        except np.linalg.LinAlgError:
            return None
        return K   # shape (1, 4)

    def compute(self, ego_state, trajectory, dt):
        """한 tick 의 steer 명령과 ControlDebug 를 산출한다.

        파이프라인:
            1) ego 를 trajectory 에 투영해 (e_y, e_psi, kappa_ref) 산출
            2) e_y_dot / e_psi_dot 도 동역학 식으로 산출 (차체 슬립 포함)
            3) Ackermann FF: δ_ff = L · κ_ref
            4) LQR FB: δ_lqr = -K · x_err (속도 의존, 매 tick K 재풀이)
            5) 합산 후 [-MAX_STEER, +MAX_STEER] 로 포화

        ego_state  : 자차 상태
        trajectory : 월드 좌표 TrajectoryPoint 리스트 (2 점 이상 필요)
        dt         : 시간 간격 [s]
        반환       : (steer ∈ [-MAX_STEER, +MAX_STEER], ControlDebug)
        """
        dbg = ControlDebug()

        if not trajectory or len(trajectory) < 2:
            dbg.steer = self._last_steer
            return self._last_steer, dbg

        max_rad = float(getattr(config, "MAX_STEER_ANGLE_RAD", 0.61))

        # ---- ego 를 trajectory 에 투영해 e_y / e_psi / κ_ref 산출 ----
        pts = [(p.x, p.y) for p in trajectory]
        seg_i, t_seg, _s = project_onto_polyline(pts, ego_state.x, ego_state.y)
        ref_yaw = polyline_heading_at(pts, seg_i)

        # CARLA 의 left-handed frame 에서 우측 양수인 부호 있는 횡오차.
        ax, ay = pts[seg_i]
        bx, by = pts[min(seg_i + 1, len(pts) - 1)]
        sx = bx - ax
        sy = by - ay
        seg_len = math.hypot(sx, sy)
        if seg_len < 1e-6:
            e_y = 0.0
        else:
            e_y = (sx * (ego_state.y - ay) - sy * (ego_state.x - ax)) / seg_len

        e_psi = normalize_angle(ego_state.yaw_rad - ref_yaw)

        # 투영점에서의 기준 곡률 κ_ref 를 양쪽 점 곡률의 선형 보간으로 산출.
        if seg_i + 1 < len(trajectory):
            k0 = trajectory[seg_i].curvature
            k1 = trajectory[seg_i + 1].curvature
            kappa_ref = k0 + (k1 - k0) * t_seg
        else:
            kappa_ref = trajectory[seg_i].curvature

        # ---- 4-상태 동역학 error state 산출 ----
        # vy_body 는 v·sin(e_psi) 만으로는 표현 안 되는 차체 슬립을 반영.
        # vx_body · κ_ref 는 기준 yaw rate 로, 이를 빼야 e_psi_dot 이 단순한
        # ego yaw rate 가 아닌 "yaw-rate ERROR" 로 의미를 갖는다.
        vx_b = ego_state.vx_body if ego_state.vx_body != 0.0 else ego_state.speed_mps
        vy_b = ego_state.vy_body
        e_y_dot = vx_b * math.sin(e_psi) + vy_b
        psi_dot_ref = vx_b * kappa_ref
        e_psi_dot = ego_state.yaw_rate_rad_s - psi_dot_ref

        # ---- Ackermann 곡률 feedforward ----
        # δ_ff = L · κ_ref 는 경로를 zero error 로 유지하는 데 필요한 운동학적
        # 정상 상태 조향. 이 항 덕분에 LQR 은 잔류 오차만 억제하면 된다.
        delta_ff_curv_rad = self.L * kappa_ref
        delta_ff_curv = clamp(
            delta_ff_curv_rad / max_rad,
            -config.MAX_STEER,
            config.MAX_STEER,
        )

        # ---- LQR FB ----
        # K 는 v 에 의존하므로 매 tick 재계산.
        delta_lqr = 0.0
        if ego_state.speed_mps > self._MIN_SPEED_FOR_LQR_MPS:
            K = self._solve_K(ego_state.speed_mps, dt)
            if K is not None:
                self._last_K = K
                x_err = np.array([[e_y], [e_y_dot], [e_psi], [e_psi_dot]])
                u = -float((K @ x_err).item())   # rad; +u 이면 우회전
                delta_lqr = clamp(u / max_rad,
                                  -config.MAX_STEER, config.MAX_STEER)

        # ---- 합산 + 포화 ----
        delta_cmd = clamp(
            delta_ff_curv + delta_lqr,
            -config.MAX_STEER,
            config.MAX_STEER,
        )
        self._last_steer = delta_cmd

        dbg.steer = delta_cmd
        dbg.e_y = e_y
        dbg.e_psi = e_psi
        
        # ControlDebug.delta_pp 필드를 HMI 의 feedforward 슬롯으로 재사용.
        # 본 컨트롤러는 여기에 Ackermann 곡률 FF 를 실어 보낸다.
        dbg.delta_pp = delta_ff_curv
        dbg.delta_lqr = delta_lqr
        dbg.lookahead_dist = 0.0
        return delta_cmd, dbg
