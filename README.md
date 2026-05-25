# carla-adas-l2plus

CARLA 0.9.16 시뮬레이터 위에 구현한 자율주행 보조 시스템 (ADAS Lv2+).
**Perception → Localization → Planning → Control** 4-계층 파이프라인을
20 Hz 로 돌리며, 차선 유지·차간 거리 제어·차선 변경·자동 긴급 제동을
하나의 스택에서 통합 처리한다.

---

## 지원 기능

| 기능 | 영문 약어 | 구현 모듈 |
|------|---------|---------|
| 차선 유지 | LKAS | `local_planner` + `lateral_controller` |
| 차선 변경 / 추월 | LC / Overtake | `behavior_planner` + `local_planner` |
| 차간 거리 제어 | ACC GAP | `scc_controller` GAP_CTRL |
| 정속 주행 | Cruise | `scc_controller` SPEED_CTRL |
| 저속 추종 / 정지 | Stop & Go | `scc_controller` STOP_GO |
| 자동 긴급 제동 | AEB | `scc_controller` AEB |

---

## 시스템 아키텍처

```
┌─────────────────── main_adas_loop.py (20 Hz, 11-step tick) ────────────────┐
│                                                                              │
│  EgoStateProvider ──→ RouteManager ──→ LaneProvider ──→ FusionManager        │
│       (자차 상태)         (경로 진행)       (차선 모델)        (EKF 트래커)     │
│                                                                              │
│                            ↓ FusedObject 리스트                              │
│                                                                              │
│  ConstantVelocityPredictor ──→ LeadVehicleSelector (current / left / right)  │
│       (1.5 s 미래 예측)                                                       │
│                                                                              │
│                            ↓                                                 │
│                                                                              │
│  BehaviorPlanner (MOBIL + 8-state LC FSM)  ──→  LocalPlanner (quintic LC)    │
│                                                                              │
│                            ↓                                                 │
│                                                                              │
│  LateralController (LQR + Ackermann FF)    SCCController (4-mode FSM)        │
│                            ↓                                                 │
│                     ActuatorArbiter (AEB 우선) ──→ carla.VehicleControl       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 디렉터리 구조

```
.
├── main_adas_loop.py     # 메인 진입점 (11 단계 tick 루프)
├── config.py             # 전역 설정 hub
├── core/                 # CARLA 비의존 데이터 타입 + 수학 유틸
├── localization/         # 자차·경로·차선
├── perception/           # 센서 + EKF 트래커 (SFOT)
├── planning/             # 의사결정 + 궤적 + 예측
├── control/              # 횡·종 제어
└── viz/                  # BEV 3-panel 시각화
```

---

## 처리 흐름

각 tick (50 ms) 마다 다음 11 단계가 순차 실행된다.

| 단계 | 모듈 | 역할 |
|------|------|------|
| 1 | EgoStateProvider | 자차의 위치·yaw·속도·가속도 갱신 |
| 2 | RouteManager | 전역 경로 진행 상태, target lane 결정 |
| 3 | LaneProvider | 현재·좌·우 차선의 중심선·곡률·경계 추출 |
| 4 | FusionManager | LiDAR + 카메라 측정을 EKF 로 융합해 트랙 갱신 |
| 5 | CV Predictor | 모든 트랙의 1.5 s 미래 trajectory 외삽 |
| 6 | LeadVehicleSelector | 현재/좌/우 차로 각각의 lead 차량 산출 |
| 7 | BehaviorPlanner | MOBIL 평가 + LC 상태머신 진행 |
| 8 | LocalPlanner | 차선 유지 또는 quintic LC 궤적 생성 |
| 9 | LateralController | Ackermann FF + LQR FB 로 조향 명령 |
| 10 | SCCController | 4-mode FSM 으로 throttle / brake 결정 |
| 11 | ActuatorArbiter | AEB 우선 조정 후 `carla.VehicleControl` 전송 |

---

## 4 계층별 동작 원리

### 1. Perception — SFOT (Sensor Fusion Object Tracking)

- LiDAR 와 카메라가 각자의 측정 모델로 객체를 검출한다.
- 각 트랙에 대해 EKF predict (CV 또는 CTRV 모션 모델) → Mahalanobis χ² 99.5 % gating → update 의 표준 칼만 사이클을 수행한다.
- score-based confirm / delete 로직으로 신규 트랙은 일정 점수 이상에서 confirmed이 되고, 분실되면 점진적으로 삭제된다.
- 출력은 `FusedObject` 리스트 (track_id, 위치, 속도, yaw, 크기, 신뢰도).

### 2. Localization — Ego · Lane · Route

- **EgoStateProvider** — CARLA actor 의 world 좌표·속도·가속도를 body frame 으로 투영.
- **LaneProvider** — OpenDRIVE waypoint 로부터 현재·좌·우 차선의 중심선 polyline, Menger 곡률, 경계선을 추출.
- **RouteManager** — `GlobalRoutePlanner` 로 출발→목적지 경로를 1 회 계산하고, tick 마다 ego 진행도를 재투영해 RouteContext 갱신.

### 3. Planning — 행동 결정 + 궤적

**(a) 예측 (CV Prediction)**
- 모든 confirmed 트랙에 대해 등속 가정으로 1.5 s 미래 trajectory 를 외삽한다.
- 차선 곡률이 큰 구간에서는 horizon 을 0.8 s 로 자동 단축해 직선 가정의 오차를 제한한다.

**(b) Lead 선정 (3-lane)**
- 현재·좌·우 차로 각각에 대해 자차 전방의 가장 가까운 차량을 lead 로 선정.
- LeadInfoSet (current/left/right) 로 출력해 MOBIL 평가와 SCC 가 동시에 활용.

**(c) 행동 결정 (BehaviorPlanner)**
- **MOBIL 모델** — safety criterion + incentive criterion 으로 LC 후보 (LEFT/RIGHT) 를 평가.
- 6 개 가속도 항 ($a_c, \tilde{a}_c, a_n, \tilde{a}_n, a_o, \tilde{a}_o$) 모두 IDM 식으로 계산해 before vs after 비교의 일관성을 유지.
- 우측통행 환경에서 RIGHT LC 는 낮은 threshold, LEFT LC 는 높은 threshold 의 비대칭 keep-right bias 적용.
- **8-state LC FSM** — `IDLE → REQUESTED → PREPARE → EXECUTE → COMPLETE → COOLDOWN` (실패 시 `ABORT/CANCEL` 경유). 5 s hysteresis, PREPARE→EXECUTE 전 후방 안전 재검증.

**(d) 궤적 생성 (LocalPlanner)**
- 차선 유지 시: lane centerline 을 그대로 추종.
- 차선 변경 시: 5 차 다항식 (quintic polynomial, $10\tau^3 - 15\tau^4 + 6\tau^5$) 로 횡 offset 함수를 생성. 양 끝점에서 위치·속도·가속도가 모두 0 인 6 경계 조건.
- LC 거리는 횡가속·jerk 한계 ($a_y \leq 1.2\,\text{m/s}^2$, $j \leq 0.8\,\text{m/s}^3$) 를 동시 만족하는 최소 길이로 동적 계산.

### 4. Control — 횡·종

**(a) Lateral (조향)**
- $\delta_{\text{cmd}} = \delta_{\text{ff}} + \delta_{\text{lqr}}$
- **Ackermann FF**: $\delta_{\text{ff}} = \arctan(L \cdot \kappa_{\text{path}})$ — 경로 곡률을 휠베이스로 환산.
- **LQR FB**: 4-state bicycle 동역학 모델 (lateral error $e_y$, lateral rate $\dot{e}_y$, heading error $e_\psi$, yaw rate error $\dot{e}_\psi$) 에 대한 LQR.
- 매 tick ZOH 이산화 → DARE 풀이로 속도-의존 게인 $K$ 재계산.

**(b) Longitudinal (SCC 4-mode FSM)**

| 모드 | 진입 조건 | 동작 |
|------|---------|------|
| SPEED_CTRL | lead 없음 | set speed 추종 (PID) |
| GAP_CTRL | lead 거리 < 70 m | desired gap 추종 (gap_error + 0.5·range_rate PID) |
| STOP_GO | speed ≤ 15 km/h **AND** range ≤ 10 m | latch 정지 유지, lead 출발 시 follow 재개 |
| AEB | TTC ≤ 1.2 s **OR** (range ≤ 6 m AND closing > 0.3 m/s) | full brake, 해제는 TTC ≥ 1.8 s 까지 hold |

ActuatorArbiter 가 AEB 출력을 최우선으로 적용해 다른 모드와 충돌 시에도 제동을 보장한다.

---

## 핵심 알고리즘 요약

| 영역 | 알고리즘 |
|------|---------|
| 객체 추적 | EKF 6/7-state (CV / CTRV), Mahalanobis χ² 99.5 % gating, score-based confirm |
| 종방향 모델 | IDM (Intelligent Driver Model) |
| 차선 변경 결정 | MOBIL (safety + incentive, 비대칭 keep-right bias) |
| 횡방향 제어 | Ackermann 곡률 FF + 동역학 bicycle LQR FB (4-state, ZOH 이산화, DARE 풀이) |
| 종방향 제어 | 4-mode SCC FSM (SPEED → GAP → STOP_GO → AEB) |
| LC 궤적 | Quintic polynomial (ay / jerk 한계 기반 동적 lc_length) |
| 운동 예측 | Constant Velocity (1.5 s horizon, 곡률 시 0.8 s 자동 단축) |

---

## 실행

```bash
# 1) CARLA 서버
./CarlaUE4.sh -quality-level=Low

# 2) ADAS 스택
python main_adas_loop.py                       # 기본 (OVERTAKE 시나리오)
python main_adas_loop.py --scenario free       # 자유 주행
python main_adas_loop.py --traffic-count 60    # ambient NPC 60대
python main_adas_loop.py --cut-in              # 결정적 cut-in 이벤트
```

**환경**: CARLA 0.9.16 (Town04), Python 3.13, `numpy` · `scipy` · `opencv-python`.

---

## License

[MIT License](LICENSE).
