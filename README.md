# carla-adas-l2plus

[![Status](https://img.shields.io/badge/status-WIP-orange.svg)](#status)
[![Maturity](https://img.shields.io/badge/maturity-research%20prototype-blue.svg)](#status)
[![SAE Level](https://img.shields.io/badge/SAE-L2%2B%20(functional)-green.svg)](#status)
[![CARLA](https://img.shields.io/badge/CARLA-0.9.16-lightgrey.svg)](https://carla.readthedocs.io/)
[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)

> CARLA 0.9.16 시뮬레이터 위에 구현한 **고전 자율주행 스택** —
> SFOT (EKF / CTRV) + MOBIL / IDM + LQR + Quintic LC + AEB.
> 양산 ADAS 의 알고리즘 레퍼런스를 만들고 검증하기 위한 **연구 시제품**.

---

## Status

본 프로젝트는 **연구·교육 목적의 알고리즘 시제품**이며, **아직 진행 중 (WIP)**.
양산 ADAS 가 아닌 **알고리즘 매트릭스 데모** 임을 명확히 한다.

| 측면 | 상태 |
|------|------|
| **기능 매트릭스 (Functional)** | **SAE L2+** — 조향 + 가감속 + 자동 LC + AEB 통합 |
| **운행 범위 (ODD)** | 고속도로 / 구조화된 도로 (보행자·신호등 인지 없음) |
| **시뮬레이션 환경** | CARLA 0.9.16, Town04 기본, 20 Hz 메인 루프 |
| **검증 상태** | 회귀 smoke test 통과, 양산 검증 (HARA / FMEA) 없음 |
| **현재 단계** | PR1 ~ PR5 적용 완료, 보행자·신호등 인지 / DMS / MRM 강화는 다음 사이클 |

본 시스템은 양산 Lv2 ADAS 와 동일한 **알고리즘 기능 매트릭스**(조향+가감속+LC+AEB)
를 제공하지만, **HMI / DMS / 인지 범위 / 안전 검증** 측면에서는 양산 대비 격차가
있는 연구 단계임을 명확히 밝힌다.

---

## ADAS 기능 (구현 완료)

| 기능 | 영문 약어 | 구현 모듈 |
|------|---------|---------|
| 차선 유지 | LKAS | `local_planner` + `lateral_controller` |
| 차선 변경 / 추월 | LC / Overtake | `behavior_planner` (MOBIL) + `local_planner` (Quintic) |
| 차간 거리 제어 | ACC GAP | `scc_controller` GAP_CTRL |
| 정속 주행 | Cruise | `scc_controller` SPEED_CTRL |
| 저속 추종 / 정지 | Stop & Go | `scc_controller` STOP_GO |
| 자동 긴급 제동 | AEB | `scc_controller` AEB |

---

## 시스템 아키텍처

```
┌───────────────────────── main_adas_loop.py (20 Hz) ─────────────────────────┐
│                                                                              │
│  EgoStateProvider ──→ RouteManager ──→ LaneProvider ──→ FusionManager        │
│       (자차)            (경로 진행)        (차선 모델)         (SFOT EKF)        │
│                                                                              │
│                            ↓ FusedObject 리스트                              │
│                                                                              │
│  ConstantVelocityPredictor ──→ LeadVehicleSelector (current/left/right)      │
│       (1.5s 미래)                                                            │
│                                                                              │
│                            ↓                                                 │
│                                                                              │
│  BehaviorPlanner (MOBIL + LC FSM)  ──→  LocalPlanner (lane / quintic LC)     │
│                                                                              │
│                            ↓                                                 │
│                                                                              │
│  LateralController (LQR + Ackermann FF)   SCCController (4-mode FSM)         │
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
├── localization/         # ego / lane / route
├── perception/           # 센서 + SFOT (EKF / CTRV) 트래커
├── planning/             # 의사결정 (MOBIL) + 궤적 + CV 예측
├── control/              # 횡 (LQR) + 종 (SCC FSM)
└── viz/                  # BEV 3-panel 시각화
```

---

## 핵심 알고리즘

| 영역 | 알고리즘 |
|------|---------|
| 객체 추적 | EKF 6/7-state (CV / CTRV), Mahalanobis χ² 99.5% gating, score-based confirmed |
| 종방향 모델 | IDM (Intelligent Driver Model) |
| 차선 변경 결정 | MOBIL (safety + incentive, 비대칭 keep-right bias) |
| 횡방향 제어 | Ackermann 곡률 FF + 동역학 bicycle LQR FB (4-state, ZOH 이산화, DARE 풀이) |
| 종방향 제어 | 4-mode SCC FSM (SPEED → GAP → STOP_GO → AEB) |
| LC 궤적 | Quintic polynomial (ay / jerk 한계 기반 동적 lc_length) |
| Motion Prediction | Constant Velocity (1.5s horizon, 곡률 시 0.8s 자동 단축) |

---

## Quick Start

### 환경 준비

- CARLA 0.9.16 (Town04 사용)
- Python 3.13 (Anaconda 권장)
- 의존성: `numpy`, `scipy` (선택), `opencv-python` (BEV 시각화)

### 실행

```bash
# 1) CARLA 서버 실행
./CarlaUE4.sh  -quality-level=Low

# 2) ADAS 스택 실행
python main_adas_loop.py                       # OVERTAKE 시나리오 (기본)
python main_adas_loop.py --scenario free       # 자유 주행
python main_adas_loop.py --stop-on-arrival     # 목적지 도달 시 종료
python main_adas_loop.py --traffic-count 60    # ambient NPC 60대
python main_adas_loop.py --cut-in              # 결정적 cut-in 이벤트
```

### 주요 플래그

| 플래그 | 의미 |
|--------|------|
| `--map Town04` | 맵 선택 |
| `--set-speed 70` | set speed [km/h] |
| `--no-bev` | BEV 시각화 비활성 |
| `--no-npc` | NPC spawn 비활성 |
| `--scenario overtake \| free` | 시나리오 선택 |
| `--cut-in --cut-in-side LEFT\|RIGHT` | 결정적 cut-in NPC 배치 |
| `--traffic-seed N` | 재현 가능한 NPC 배치 |

---

## 최근 적용된 고도화 (PR1 ~ PR5)

| PR | 변경 |
|----|------|
| PR1 | Constant Velocity Predictor 모듈 신설 (1.5s horizon) |
| PR2 | LeadVehicleSelector 3-lane 확장 (`LeadInfoSet` = current / left / right) |
| PR3 | BehaviorPlanner MOBIL 평가에 worst-case (현재 vs 1초 후) gap 주입 |
| PR4 | PREPARE → EXECUTE 직전 후방 안전 재검증 (failure 시 REQUESTED 회귀) |
| PR5 | LC 궤적 smoothstep → quintic polynomial 교체, ay/jerk 한계 기반 동적 lc_length |

세부 회귀 결과: PR2 (1,2,3) / PR3 (30→26 m) / PR4 (REAR_GAP_FUTURE_17m<18) / PR5 (85m, 6.4s).

추가로 코드 스타일 리팩터링 (dataclass → 일반 class, typing 표기 제거,
`@staticmethod` → 모듈 함수, 학술 인용 / PR 마커 / spec § 정리) 완료.

---

## Roadmap

다음 사이클에서 추가 예정 (양산 Lv2 수준으로 끌어올리기 위해):

- [ ] **보행자 / 자전거 인지** — `FusedObject.object_type` 분기 + 검출 확장
- [ ] **신호등 인지 (TSR)** — 카메라 색 검출 + CARLA `get_traffic_lights()`
- [ ] **MRM (Minimal Risk Maneuver)** — 갓길 정차 + 비상등 알람 강화
- [ ] **ODD monitor** — 속도 / 도로 종류 / 날씨 모니터링해 자동 deactivate
- [ ] **HMI 모듈** — BEV sidebar 시각 알람 + 시뮬 청각 알람
- [ ] **MPC LC** — 현재 quintic 외에 cvxpy 기반 corridor MPC 시범 적용 (warm-start 필요)
- [ ] **CA prediction 옵션** — 가속도 추정 가능 시 CV 대체

---

## License

본 프로젝트는 [MIT License](LICENSE) 로 배포된다. CARLA 0.9.16 자체도 MIT.
참고자료 (강의 PDF / 외부 실습 코드) 는 본 리포에 포함되지 않는다.
