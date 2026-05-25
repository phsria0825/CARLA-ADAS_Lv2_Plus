"""
SFOT EKF 가 사용하는 모션 모델 — 상태 천이 + Jacobian + 프로세스 노이즈.

두 종류의 production-standard 모델을 제공한다.

  CV  (Constant Velocity, 6D)
      state : x = [px, py, pz, vx, vy, vz]^T
      F     : 선형 — position += velocity·dt
      용도  : 단순한 직선 추적의 baseline. 가벼우나 곡선에서 직선 외삽 드리프트.

  CTRV (Constant Turn Rate and Velocity, 7D)
      state : x = [px, py, v, psi, psi_dot, pz, vz]^T
      F     : (px, py, psi) 에 대해 비선형 — EKF Jacobian 제공
      수직 성분 (pz, vz) 은 CV 가정 (차량은 평면 이동 + 수직 성분 미미)
      프로세스 노이즈: heading 방향 가속도 + yaw-rate-rate 를 두 개의 white-
      noise 원천으로 사용하는 상태 의존적 G 행렬.
      용도: 차량 MOT 에서 *예측* 단계가 차량을 yaw 호 위에 유지시키므로
            곡선 도로에서 직선 외삽 오차가 없다.

각 모델이 노출하는 인터페이스:

    dim_state              : int                                 (상태 크기)
    position_indices       : list[int]                           ([px, py, pz] 컬럼)
    initial_state(pos, yaw_w)
    initial_covariance(R_pos_world)
    predict_state(x, dt)
    jacobian(x, dt)
    process_noise(x, dt)
    extract_position(x)        -> (px, py, pz)
    extract_velocity_world(x)  -> (vx, vy, vz)  world 좌표계
    extract_yaw_world(x)       -> float 또는 None
    extract_yaw_rate(x)        -> float 또는 None

Sensor 의 측정 Jacobian 은 position_indices 를 사용해 선형화된 행을 올바른
상태 컬럼에 배치하므로, CV ↔ CTRV 전환 시 sfot_measurements.py 의 변경은
필요하지 않다.
"""
import math

import numpy as np


class MotionModel:
    """모든 모션 모델의 추상 기반 클래스 — 서브클래스가 6 개 메서드를 채운다."""

    name = "abstract"
    dim_state = 0
    position_indices = []   # 상태의 어느 컬럼이 [px, py, pz] 를 담는지 지정

    def initial_state(self, pos_world, yaw_world):
        """LiDAR 측정 위치 (월드 좌표) 와 초기 yaw 로부터 초기 상태 벡터를 만든다."""
        raise NotImplementedError

    def initial_covariance(self, R_pos_world):
        """측정 위치 공분산으로부터 초기 상태 공분산 P 를 만든다."""
        raise NotImplementedError

    def predict_state(self, x, dt):
        """상태 x 를 dt 만큼 시간 전파한다 (f(x, dt) 계산)."""
        raise NotImplementedError

    def jacobian(self, x, dt):
        """현 상태 x 에서 dt 시간 전파의 Jacobian ∂f/∂x 를 반환한다."""
        raise NotImplementedError

    def process_noise(self, x, dt):
        """dt 시간 전파에 대응하는 프로세스 노이즈 Q 행렬을 반환한다."""
        raise NotImplementedError

    def extract_position(self, x):
        """상태에서 (px, py, pz) world 좌표를 뽑아낸다."""
        raise NotImplementedError

    def extract_velocity_world(self, x):
        """상태에서 (vx, vy, vz) world 좌표 속도를 뽑아낸다."""
        raise NotImplementedError

    def extract_yaw_world(self, x):
        """world frame heading 을 반환한다. 모델이 heading 을 추적하지 않으면 None."""
        return None

    def extract_yaw_rate(self, x):
        """yaw rate 를 반환한다. 모델이 추적하지 않으면 None."""
        return None


# ---------------------------------------------------------------------------
# CV — Constant Velocity 모델 (6 상태)
# ---------------------------------------------------------------------------

class CVMotionModel(MotionModel):
    """6 상태 등속도 모델 — 위치 / 속도 분리, 선형 동역학."""

    name = "CV"
    dim_state = 6
    position_indices = [0, 1, 2]

    def __init__(self,
                 sigma_vel_init_xy=50.0,
                 sigma_vel_init_z=10.0,
                 q_accel=3.0):
        self.sigma_vel_xy = sigma_vel_init_xy
        self.sigma_vel_z = sigma_vel_init_z
        self.q = q_accel

    def initial_state(self, pos_world, yaw_world):
        """첫 LiDAR 측정 위치로 위치 채우고 속도는 0 으로 시작."""
        x = np.zeros((6, 1), dtype=float)
        x[0, 0], x[1, 0], x[2, 0] = pos_world
        return x

    def initial_covariance(self, R_pos_world):
        """측정 위치 공분산 + 큰 속도 prior 분산으로 P 초기화."""
        P = np.zeros((6, 6), dtype=float)
        P[0:3, 0:3] = R_pos_world
        P[3, 3] = self.sigma_vel_xy ** 2
        P[4, 4] = self.sigma_vel_xy ** 2
        P[5, 5] = self.sigma_vel_z ** 2
        return P

    def _F(self, dt):
        """상수 transition matrix — pos += vel·dt 의 6×6 형태."""
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    def predict_state(self, x, dt):
        """선형 상태 전파 x ← F·x."""
        return self._F(dt) @ x

    def jacobian(self, x, dt):
        """CV 는 선형이므로 Jacobian = transition matrix."""
        return self._F(dt)

    def process_noise(self, x, dt):
        """white acceleration 모델 기반 6×6 Q (위치/속도 블록 대칭)."""
        q = self.q
        q3 = (dt ** 3) * q / 3.0
        q2 = (dt ** 2) * q / 2.0
        q1 = dt * q
        Q = np.zeros((6, 6), dtype=float)
        for i in range(3):
            Q[i, i] = q3
            Q[i + 3, i + 3] = q1
            Q[i, i + 3] = q2
            Q[i + 3, i] = q2
        return Q

    def extract_position(self, x):
        """상태 인덱스 [0, 1, 2] 가 (px, py, pz)."""
        return float(x[0, 0]), float(x[1, 0]), float(x[2, 0])

    def extract_velocity_world(self, x):
        """상태 인덱스 [3, 4, 5] 가 (vx, vy, vz) world 좌표."""
        return float(x[3, 0]), float(x[4, 0]), float(x[5, 0])


# ---------------------------------------------------------------------------
# CTRV — Constant Turn-Rate and Velocity 모델 (7 상태, planar CTRV + 수직 CV)
# ---------------------------------------------------------------------------

class CTRVMotionModel(MotionModel):
    """7 상태 평면 CTRV + 수직 CV 모델 — 차량 곡선 추적의 표준 선택지.

    State: x = [px, py, v, psi, psi_dot, pz, vz]^T  (7D)

    Planar 갱신식 (psi_dot != 0):
        psi(k+1)    = psi + psi_dot · dt
        px(k+1)     = px + (v / psi_dot) · (sin(psi(k+1)) - sin(psi))
        py(k+1)     = py + (v / psi_dot) · (-cos(psi(k+1)) + cos(psi))
        v(k+1)      = v
        psi_dot(k+1)= psi_dot

    Planar 갱신식 (psi_dot ≈ 0 — 직선 운동이며 0 으로 나누는 것을 회피):
        px(k+1)     = px + v · cos(psi) · dt
        py(k+1)     = py + v · sin(psi) · dt
        psi(k+1)    = psi

    수직 성분 (pz, vz) 은 CV 를 따른다.
        pz(k+1) = pz + vz · dt;  vz(k+1) = vz

    Jacobian 은 해석적으로 계산되며, 프로세스 노이즈는 표준 CTRV G-matrix
    형태로 heading 방향 가속도와 yaw-rate-rate 두 white-noise 원천을 사용
    한다. (pz, vz) 에는 독립적인 수직 가속도 노이즈가 추가로 적용된다.
    """

    name = "CTRV"
    dim_state = 7
    position_indices = [0, 1, 5]   # px=0, py=1, pz=5
    _PSI_DOT_EPS = 1e-4

    def __init__(self,
                 sigma_a=2.0,
                 sigma_psi_dotdot=0.5,
                 sigma_az=1.0,
                 sigma_v_init=20.0,
                 sigma_psi_init=1.5,
                 sigma_psi_dot_init=0.8,
                 sigma_vz_init=5.0):
        self.sigma_a = sigma_a
        self.sigma_psi_dotdot = sigma_psi_dotdot
        self.sigma_az = sigma_az
        self.sigma_v_init = sigma_v_init
        self.sigma_psi_init = sigma_psi_init
        self.sigma_psi_dot_init = sigma_psi_dot_init
        self.sigma_vz_init = sigma_vz_init

    def initial_state(self, pos_world, yaw_world):
        """위치는 LiDAR 측정 그대로, heading 은 측정 yaw, 속도 / yaw rate 는 0 으로 시작."""
        x = np.zeros((7, 1), dtype=float)
        x[0, 0] = pos_world[0]
        x[1, 0] = pos_world[1]
        x[2, 0] = 0.0                                       # v
        x[3, 0] = yaw_world if yaw_world is not None else 0.0   # psi
        x[4, 0] = 0.0                                       # psi_dot
        x[5, 0] = pos_world[2]                              # pz
        x[6, 0] = 0.0                                       # vz
        return x

    def initial_covariance(self, R_pos_world):
        """측정 평면/수직 위치 공분산 + 동적 상태 prior 로 P 초기화."""
        P = np.zeros((7, 7), dtype=float)
        # 측정으로부터 얻은 평면 위치
        P[0:2, 0:2] = R_pos_world[0:2, 0:2]
        # 측정으로부터 얻은 수직 위치 (행/열 5)
        P[5, 5] = R_pos_world[2, 2]
        # 동적 상태에 대한 prior
        P[2, 2] = self.sigma_v_init ** 2
        P[3, 3] = self.sigma_psi_init ** 2
        P[4, 4] = self.sigma_psi_dot_init ** 2
        P[6, 6] = self.sigma_vz_init ** 2
        return P

    def predict_state(self, x, dt):
        """CTRV 비선형 상태 전파. yaw rate 가 작으면 직선 운동식으로 분기."""
        px = float(x[0, 0]); py = float(x[1, 0])
        v = float(x[2, 0]); psi = float(x[3, 0]); psi_dot = float(x[4, 0])
        pz = float(x[5, 0]); vz = float(x[6, 0])

        if abs(psi_dot) < self._PSI_DOT_EPS:
            px_new = px + v * math.cos(psi) * dt
            py_new = py + v * math.sin(psi) * dt
            psi_new = psi
        else:
            psi_new = psi + psi_dot * dt
            px_new = px + (v / psi_dot) * (math.sin(psi_new) - math.sin(psi))
            py_new = py + (v / psi_dot) * (-math.cos(psi_new) + math.cos(psi))

        return np.array(
            [[px_new], [py_new], [v], [psi_new], [psi_dot],
             [pz + vz * dt], [vz]],
            dtype=float,
        )

    def jacobian(self, x, dt):
        """CTRV 상태 전파의 7×7 Jacobian. yaw rate 분기에 따라 평면 블록이 달라진다."""
        v = float(x[2, 0]); psi = float(x[3, 0]); psi_dot = float(x[4, 0])
        F = np.eye(7)

        if abs(psi_dot) < self._PSI_DOT_EPS:
            # 직선 운동에 대한 선형화 Jacobian
            F[0, 2] = math.cos(psi) * dt
            F[0, 3] = -v * math.sin(psi) * dt
            F[1, 2] = math.sin(psi) * dt
            F[1, 3] = v * math.cos(psi) * dt
            # (극한에서 ∂px/∂psi_dot 항은 존재하지 않는다)
        else:
            psi_new = psi + psi_dot * dt
            inv_pd = 1.0 / psi_dot
            inv_pd2 = inv_pd * inv_pd
            sp = math.sin(psi); cp = math.cos(psi)
            spn = math.sin(psi_new); cpn = math.cos(psi_new)

            # px 에 대한 편미분
            F[0, 2] = (spn - sp) * inv_pd
            F[0, 3] = (v * inv_pd) * (cpn - cp)
            F[0, 4] = (-v * inv_pd2) * (spn - sp) + (v * dt * inv_pd) * cpn
            # py 에 대한 편미분
            F[1, 2] = (-cpn + cp) * inv_pd
            F[1, 3] = (v * inv_pd) * (spn - sp)
            F[1, 4] = (-v * inv_pd2) * (-cpn + cp) + (v * dt * inv_pd) * spn

        # psi(k+1) = psi + psi_dot · dt
        F[3, 4] = dt
        # pz(k+1) = pz + vz · dt
        F[5, 6] = dt
        return F

    def process_noise(self, x, dt):
        """상태 의존적 G-matrix 형식의 7×7 Q.

        세 가지 white noise 원천:
            ν_a    : heading 방향 종방향 가속도 노이즈   σ_a
            ν_psi  : yaw-rate-rate 노이즈                σ_psi_dotdot
            ν_az   : 수직 가속도 노이즈 (독립)            σ_az

        G_planar (5 × 2):
            [0.5 dt² cos ψ,   0      ]
            [0.5 dt² sin ψ,   0      ]
            [ dt,             0      ]
            [ 0,              0.5 dt²]
            [ 0,              dt     ]
        Q_planar = G · diag(σ_a², σ_ψ̈²) · Gᵀ
        Q_z      = σ_az² · [[dt⁴/4, dt³/2], [dt³/2, dt²]]
        """
        psi = float(x[3, 0])
        cp = math.cos(psi); sp = math.sin(psi)
        dt2 = dt * dt
        sa2 = self.sigma_a ** 2
        sppdd2 = self.sigma_psi_dotdot ** 2
        saz2 = self.sigma_az ** 2

        G = np.array([
            [0.5 * dt2 * cp, 0.0],
            [0.5 * dt2 * sp, 0.0],
            [dt,             0.0],
            [0.0,            0.5 * dt2],
            [0.0,            dt       ],
        ])
        Q_planar = G @ np.diag([sa2, sppdd2]) @ G.T   # (5, 5)

        # 수직 성분: 2 상태 CV-z white-acc 노이즈
        Q_z = np.array([
            [(dt2 * dt2) / 4.0,  (dt2 * dt) / 2.0],
            [(dt2 * dt) / 2.0,    dt2            ],
        ]) * saz2

        Q = np.zeros((7, 7), dtype=float)
        Q[0:5, 0:5] = Q_planar
        Q[5:7, 5:7] = Q_z
        return Q

    def extract_position(self, x):
        """상태에서 (px, py, pz) 추출 — pz 는 인덱스 5."""
        return float(x[0, 0]), float(x[1, 0]), float(x[5, 0])

    def extract_velocity_world(self, x):
        """heading 방향 속도를 world (vx, vy) 로 분해 + vz 그대로."""
        v = float(x[2, 0]); psi = float(x[3, 0])
        return v * math.cos(psi), v * math.sin(psi), float(x[6, 0])

    def extract_yaw_world(self, x):
        """CTRV 는 heading 을 직접 추적하므로 상태에서 그대로 반환."""
        return float(x[3, 0])

    def extract_yaw_rate(self, x):
        """CTRV 의 ω = psi_dot 을 그대로 반환."""
        return float(x[4, 0])


# ---------------------------------------------------------------------------
# 팩토리
# ---------------------------------------------------------------------------

def build_motion_model(name):
    """이름 문자열로 적절한 MotionModel 인스턴스를 생성한다.

    name : "CV" 또는 "CTRV" (대소문자 무시). None 이면 "CTRV" 로 기본화.
    반환 : CVMotionModel 또는 CTRVMotionModel
    """
    name = (name or "CTRV").upper()
    if name == "CV":
        return CVMotionModel()
    if name == "CTRV":
        return CTRVMotionModel()
    raise ValueError(f"Unknown motion model: {name!r}; expected 'CV' or 'CTRV'")
