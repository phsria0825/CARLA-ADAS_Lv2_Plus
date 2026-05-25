"""
ADAS Lv2+ 메인 루프 (CARLA 0.9.15).

본 모듈은 메인 루프의 매 tick 처리 순서를 다음과 같이 구현한다.
    world.tick(): CARLA actor에서 ego 상태 추출
        -> RouteManager (선택)        : 전역 경로 진행도 갱신
        -> LaneProvider               : waypoint 기반 차선 모델 + curvature
        -> FusionManager (SFOT)       : 카메라 + LiDAR -> EKF -> FusedObject
        -> LeadVehicleSelector        : 현 차선 lead 선정
        -> BehaviorPlanner (MOBIL)    : LC 의사결정 + 상태머신
        -> LocalPlanner               : trajectory 생성 (lane-keep / LC EXECUTE): Ackermann FF + 동역학 LQR FB
        -> SCCController              : 종방향 (SPEED / GAP / SnG / AEB): AEB 최우선 결정
        -> ego.apply_control(): 카메라 뷰 + BEV + 사이드바

실행 예:
    python main_adas_loop.py                       # OVERTAKE 시나리오 (기본)
    python main_adas_loop.py --scenario free       # 자유 주행 모드
    python main_adas_loop.py --stop-on-arrival     # 목적지 도달 시 종료
    python main_adas_loop.py --traffic-count 60    # 더 빡빡한 ambient 트래픽
    python main_adas_loop.py --traffic-count 0     # ambient 트래픽 없음 (결정적)
    python main_adas_loop.py --cut-in              # 결정적 cut-in 이벤트

선택 플래그:
    --map Town04 --set-speed 70 --no-bev --no-npc
                    --scenario overtake|free
                    --destination-index N    ('free' 시나리오에서만 사용)
                    --no-route               (RouteManager 비활성)
                    --traffic-count N        (기본 40; 75% lane-walking + 25% 배경)
                    --traffic-seed N         (>=0 = 재현 가능한 배치)
                    --cut-in --cut-in-side LEFT|RIGHT --cut-in-after-s 6.0

OVERTAKE 시나리오 레이아웃:
    - ego는 인접 동방향 차선이 있는 다차선 직선 도로에 spawn된다.
    - 35 m 전방 동일 차선에 35 km/h 수준의 느린 NPC가 배치된다.
    - 목적지는 동일 차선 600 m 전방 (사이에 junction 없음)이다.
    전역 경로가 직선으로 유지되므로 차선을 떠날 유일한 동기는 추월이다.
    MOBIL이 5초 안정 추종 + IDM 인센티브 임계 통과 조건을 만족하면
    LC가 발동하여 추월-복귀 시퀀스를 자동으로 수행한다.
"""
import argparse
import math
import random
import sys
import time

import carla

import config
from core.adas_types import BehaviorState, ControlDebug, LaneChangeState, LeadInfo
from planning.behavior_planner import BehaviorPlanner
from viz.bev_visualizer import BEVVisualizer
from localization.ego_state_provider import EgoStateProvider
from perception.fusion_manager import FusionManager
from control.lateral_controller import LateralController
from localization.lane_provider import LaneProvider
from planning.lead_vehicle_selector import LeadVehicleSelector
from planning.local_planner import LocalPlanner
from planning.predictors import ConstantVelocityPredictor
from localization.route_manager import RouteManager
from control.scc_controller import SCCController


# ----------------------------------------------------------------------------
# CARLA 보조 함수
# ----------------------------------------------------------------------------

def _connect(host, port, timeout):
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    return client


def _set_sync(world, fixed_delta):
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 10
    world.apply_settings(settings)
    return original


def _restore_sync(world, original):
    try:
        world.apply_settings(original)
    except Exception as e:
        print(f"[CLEAN] restore settings failed: {e}")


def _pick_ego_spawn(world):
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        return None
    # 차선변경 시나리오 도달성을 위해 다차선 도로의 spawn point를 우선 선정한다.
    carla_map = world.get_map()
    for sp in spawn_points:
        wp = carla_map.get_waypoint(sp.location, project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
        if wp is None or wp.is_junction:
            continue
        # 인접 차선이 최소 1개 이상 존재해야 LC 의사결정의 의미가 있다.
        if wp.get_left_lane() is not None or wp.get_right_lane() is not None:
            return sp
    return spawn_points[0]


# ----------------------------------------------------------------------------
# 시나리오: OVERTAKE (discretionary lane change 테스트베드)
#
# 본 시나리오는 ego가 차선을 떠날 유일한 동기가 "느린 lead 차량 추월"이 되도록
# 설계된다. 전역 경로가 동일 차선을 따라 직선으로 유지되므로, LC 로직이 정상
# 동작하지 않을 경우 ego는 GAP_CTRL로 NPC 뒤에 영구 정착한다. MOBIL 안정-추종
# 조건이 충족되는 순간 이 동일 장면이 자동으로 추월-복귀 시퀀스를 유도한다.
#
# stable-follow 게이트를 충족하도록 다음 파라미터가 튜닝되어 있다.
#   - ego set-speed     : 70 km/h   (set_speed − ego_speed ≥ 1.5 m/s 보장)
#   - NPC 감속 비율     : speed_limit 대비 55% 감속 (80 km/h 도로에서 약 35 km/h)
#   - NPC 초기 간격     : 35 m       (LC_LEAD_RANGE_MIN..MAX 윈도우 안으로 수렴)
#   - 목적지            : 동일 차선 600 m 전방 (사이 junction 없음)
#   - 요구 헤드룸       : 600 m 직선·無 junction 구간으로 LC 진입·통과·복귀가
#                         도로 토폴로지 변동 없이 완료되도록 보장
# ----------------------------------------------------------------------------

OVERTAKE_NPC_AHEAD_M = 35.0
OVERTAKE_DEST_AHEAD_M = 600.0
OVERTAKE_HEADROOM_M = 600.0
OVERTAKE_NPC_SLOW_PCT = 55.0       # percentage_speed_difference: + means slower
OVERTAKE_RECOMMENDED_SET_SPEED_KMH = 70.0


def _find_overtake_spawn(world):
    """
    모든 spawn point를 순회하면서 다음 조건을 모두 만족하는 첫 항목을 반환한다.
      - Driving 차선 위에 있으며 junction 내부가 아닐 것
      - 동일 방향의 인접 Driving 차선이 최소 한쪽 이상 존재할 것 (LC 가능)
      - 전방 OVERTAKE_HEADROOM_M 거리까지 junction이 없을 것 — 느린 NPC 배치,
        LC 진입, 추월, 원 차선 복귀 전 과정이 토폴로지 변동 없이 끝나도록
        보장한다.
    조건을 만족하는 후보가 없으면 None을 반환한다.
    """
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    # rank candidates: prefer ones with BOTH adjacent lanes (more flexibility)
    primary = []
    secondary = []
    STEP_M = 5.0

    for sp in spawn_points:
        wp = carla_map.get_waypoint(sp.location, project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
        if wp is None or wp.is_junction:
            continue

        left = wp.get_left_lane()
        right = wp.get_right_lane()
        left_ok = (left is not None
                   and left.lane_type == carla.LaneType.Driving
                   and (left.lane_id * wp.lane_id) > 0)
        right_ok = (right is not None
                    and right.lane_type == carla.LaneType.Driving
                    and (right.lane_id * wp.lane_id) > 0)
        if not (left_ok or right_ok):
            continue

        # headroom walk
        cur = wp
        accumulated = 0.0
        ok = True
        while accumulated < OVERTAKE_HEADROOM_M:
            nxts = cur.next(STEP_M)
            if not nxts:
                ok = False
                break
            cur = nxts[0]
            if cur.is_junction:
                ok = False
                break
            accumulated += STEP_M
        if not ok:
            continue

        if left_ok and right_ok:
            primary.append(sp)
        else:
            secondary.append(sp)

    if primary:
        return primary[0]
    if secondary:
        return secondary[0]
    return None


def _setup_overtake_scenario(
    world,
    traffic_manager,
):
    """
    Spawn ego + slow NPC and compute the downstream destination.

    Returns:
        (ego, npc_lead, destination_transform). destination_transform is
        None if the headroom walk fell short (very unlikely once
        _find_overtake_spawn passed).
    """
    carla_map = world.get_map()
    bp_lib = world.get_blueprint_library()

    ego_sp = _find_overtake_spawn(world)
    if ego_sp is None:
        raise RuntimeError(
            "OVERTAKE scenario: no spawn point with adjacent lane and "
            f"{OVERTAKE_HEADROOM_M:.0f} m straight headroom in this map. "
            "Try a different map (Town04 highway loop is recommended)."
        )

    # --- ego ---
    ego_bp = bp_lib.find(config.VEHICLE_BP)
    ego_bp.set_attribute('role_name', 'hero')
    if ego_bp.has_attribute('color'):
        ego_bp.set_attribute('color', '0,120,255')
    ego = world.try_spawn_actor(ego_bp, ego_sp)
    if ego is None:
        raise RuntimeError("OVERTAKE scenario: failed to spawn ego at chosen point")

    ego_wp = carla_map.get_waypoint(ego_sp.location, project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
    left = ego_wp.get_left_lane()
    right = ego_wp.get_right_lane()
    adj = []
    if left is not None and left.lane_type == carla.LaneType.Driving \
            and (left.lane_id * ego_wp.lane_id) > 0:
        adj.append("LEFT")
    if right is not None and right.lane_type == carla.LaneType.Driving \
            and (right.lane_id * ego_wp.lane_id) > 0:
        adj.append("RIGHT")

    print(f"[SCENARIO] === OVERTAKE TEST ===")
    print(f"[SCENARIO] ego @ ({ego_sp.location.x:.1f}, {ego_sp.location.y:.1f}) "
          f"road={ego_wp.road_id} lane={ego_wp.lane_id} "
          f"adjacent={'/'.join(adj) if adj else 'NONE'}")

    # --- slow NPC, same lane, OVERTAKE_NPC_AHEAD_M ahead ---
    ahead = ego_wp.next(OVERTAKE_NPC_AHEAD_M)
    if not ahead:
        print("[SCENARIO] WARNING: no waypoint found for NPC spawn; aborting NPC")
        return ego, None, None
    npc_wp = ahead[0]

    # Use a recognizable, slightly smaller car so the BEV makes the role clear.
    npc_bp = bp_lib.find('vehicle.audi.tt') if bp_lib.find('vehicle.audi.tt') else \
             random.choice(bp_lib.filter('vehicle.*'))
    if npc_bp.has_attribute('color'):
        npc_bp.set_attribute('color', '220,40,40')
    npc_bp.set_attribute('role_name', 'slow_lead')

    npc_tf = npc_wp.transform
    npc_tf.location.z += 0.3
    npc = world.try_spawn_actor(npc_bp, npc_tf)
    if npc is None:
        print("[SCENARIO] WARNING: NPC spawn collided; retrying further ahead")
        ahead2 = ego_wp.next(OVERTAKE_NPC_AHEAD_M + 10.0)
        if ahead2:
            npc_tf = ahead2[0].transform
            npc_tf.location.z += 0.3
            npc = world.try_spawn_actor(npc_bp, npc_tf)

    if npc is not None:
        npc.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.vehicle_percentage_speed_difference(npc, OVERTAKE_NPC_SLOW_PCT)
        traffic_manager.auto_lane_change(npc, False)
        traffic_manager.distance_to_leading_vehicle(npc, 5.0)
        # Ignore traffic lights so it doesn't accidentally stop mid-overtake
        traffic_manager.ignore_lights_percentage(npc, 100.0)
        print(f"[SCENARIO] slow NPC @ ({npc_tf.location.x:.1f}, {npc_tf.location.y:.1f}) "
              f"speed_diff=+{OVERTAKE_NPC_SLOW_PCT:.0f}%  (~35 km/h on 80 km/h roads)")
    else:
        print("[SCENARIO] WARNING: could not spawn slow NPC; running solo")

    # --- destination: same lane, OVERTAKE_DEST_AHEAD_M further ---
    dest_wp = ego_wp
    accumulated = 0.0
    while accumulated < OVERTAKE_DEST_AHEAD_M:
        nxts = dest_wp.next(10.0)
        if not nxts:
            break
        dest_wp = nxts[0]
        accumulated += 10.0
    dest_tf = dest_wp.transform
    print(f"[SCENARIO] destination @ ({dest_tf.location.x:.1f}, {dest_tf.location.y:.1f}) "
          f"({accumulated:.0f} m down road={dest_wp.road_id} lane={dest_wp.lane_id})")
    print(f"[SCENARIO] recommended --set-speed {OVERTAKE_RECOMMENDED_SET_SPEED_KMH:.0f} "
          f"so speed_gap >= 1.5 m/s triggers the LC stable-follow gate")

    return ego, npc, dest_tf


def _spawn_ego(world):
    bp_lib = world.get_blueprint_library()
    bp = bp_lib.find(config.VEHICLE_BP)
    bp.set_attribute('role_name', 'hero')
    if bp.has_attribute('color'):
        bp.set_attribute('color', '0,120,255')
    sp = _pick_ego_spawn(world)
    if sp is None:
        raise RuntimeError("No spawn points available")
    ego = world.try_spawn_actor(bp, sp)
    if ego is None:
        # fall back: any spawn point
        for alt in world.get_map().get_spawn_points():
            ego = world.try_spawn_actor(bp, alt)
            if ego is not None:
                break
    if ego is None:
        raise RuntimeError("Failed to spawn ego")
    print(f"[SPAWN] ego @ ({sp.location.x:.1f}, {sp.location.y:.1f})")
    return ego


def _spawn_npc_ahead(world, ego,
                     traffic_manager):
    bp_lib = world.get_blueprint_library()
    bp = random.choice(bp_lib.filter('vehicle.*'))
    if bp.has_attribute('color'):
        bp.set_attribute('color', '255, 80, 80')
    bp.set_attribute('role_name', 'npc_lead')

    carla_map = world.get_map()
    ego_tf = ego.get_transform()
    ego_wp = carla_map.get_waypoint(ego_tf.location, project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
    if ego_wp is None:
        return None

    # walk 50 m ahead on the same lane
    ahead = ego_wp.next(50.0)
    if not ahead:
        return None
    spawn_wp = ahead[0]
    spawn_tf = spawn_wp.transform
    spawn_tf.location.z += 0.3  # avoid ground collision
    npc = world.try_spawn_actor(bp, spawn_tf)
    if npc is None:
        return None
    npc.set_autopilot(True, traffic_manager.get_port())
    traffic_manager.vehicle_percentage_speed_difference(npc, 30.0)  # 30% slower than limit
    traffic_manager.auto_lane_change(npc, False)
    print(f"[SPAWN] NPC lead @ ({spawn_tf.location.x:.1f}, {spawn_tf.location.y:.1f})")
    return npc


# ----------------------------------------------------------------------------
# Ambient traffic
# ----------------------------------------------------------------------------
#
# Strategy:
#
#   1) ~75% of NPCs are spawned by LANE-WALKING from ego's waypoint along all
#      same-direction lanes (current + up to 3 adjacent each side). Each lane
#      is sampled both forward and backward at random ~20-50 m steps so the
#      NPCs share ego's direction of travel and are within perception range
#      from t=0.
#
#   2) ~25% of NPCs are spawned at random map spawn points for background
#      atmosphere.
#
#   3) Per-vehicle TM tweaks inject speed variance and a high auto-lane-change
#      probability, so faster NPCs naturally overtake slower ones and cut in
#      across ego's lane during the simulation.
#
#   4) Optional scripted cut-in: a designated NPC in the adjacent lane that
#      we manually fire tm.force_lane_change() on after a few seconds, for a
#      deterministic interaction event when reproducibility is desired.

_TRAFFIC_AVOID_RADIUS_M = 25.0
_SWARM_MIN_STEP_M = 20.0
_SWARM_MAX_STEP_M = 50.0
_SWARM_MAX_FORWARD_STEPS = 8
_SWARM_MAX_BACKWARD_STEPS = 4
_SWARM_MAX_SIDE_LANES = 3


def _same_direction_lanes(ego_wp):
    """Collect ego's lane + up to _SWARM_MAX_SIDE_LANES same-direction
    adjacent lanes on each side. Uses the OpenDRIVE sign rule (same lane_id
    sign = same travel direction) to reject oncoming lanes.

    Driver-perspective LEFT/RIGHT swap is NOT needed here because we're
    collecting BOTH sides anyway — the swarm doesn't care which side is
    the driver's left."""
    out = [ego_wp]
    # walk LEFT (OpenDRIVE direction)
    cur = ego_wp
    for _ in range(_SWARM_MAX_SIDE_LANES):
        nxt = cur.get_left_lane()
        if (nxt is None
                or nxt.lane_type != carla.LaneType.Driving
                or nxt.lane_id * ego_wp.lane_id <= 0):
            break
        out.append(nxt)
        cur = nxt
    # walk RIGHT
    cur = ego_wp
    for _ in range(_SWARM_MAX_SIDE_LANES):
        nxt = cur.get_right_lane()
        if (nxt is None
                or nxt.lane_type != carla.LaneType.Driving
                or nxt.lane_id * ego_wp.lane_id <= 0):
            break
        out.append(nxt)
        cur = nxt
    return out


def _swarm_transforms(
    world,
    ego,
    avoid_locations,
    n_target,
):
    """Build transforms for the swarm by lane-walking forward and backward
    on every same-direction lane around ego."""
    carla_map = world.get_map()
    ego_wp = carla_map.get_waypoint(
        ego.get_transform().location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if ego_wp is None:
        return []

    lanes = _same_direction_lanes(ego_wp)
    transforms = []

    def _far_enough(loc):
        for av in avoid_locations:
            if math.hypot(loc.x - av.x, loc.y - av.y) < _TRAFFIC_AVOID_RADIUS_M:
                return False
        return True

    for lane_wp in lanes:
        # Forward chain
        cur = lane_wp
        for _ in range(_SWARM_MAX_FORWARD_STEPS):
            step = random.uniform(_SWARM_MIN_STEP_M, _SWARM_MAX_STEP_M)
            nxts = cur.next(step)
            if not nxts:
                break
            # Prefer same-(road, section, lane); reject opposite direction.
            same_lane = [n for n in nxts
                         if n.lane_id * cur.lane_id > 0
                         and n.road_id == cur.road_id
                         and n.section_id == cur.section_id
                         and n.lane_id == cur.lane_id]
            cur = same_lane[0] if same_lane else next(
                (n for n in nxts if n.lane_id * cur.lane_id > 0), None
            )
            if cur is None:
                break
            tf = cur.transform
            tf.location.z += 0.5
            if not _far_enough(tf.location):
                continue
            transforms.append(tf)

        # Backward chain — Town04 같은 cyclic topology 에서 무한 루프를 피하기
        # 위해 반복 횟수를 cap 한다. single-step previous(d) 자체는 안전하지만
        # safety net 으로 cap 을 둔다.
        cur = lane_wp
        for _ in range(_SWARM_MAX_BACKWARD_STEPS):
            step = random.uniform(_SWARM_MIN_STEP_M, _SWARM_MAX_STEP_M)
            prevs = cur.previous(step)
            if not prevs:
                break
            same_lane = [p for p in prevs
                         if p.lane_id * cur.lane_id > 0
                         and p.road_id == cur.road_id
                         and p.section_id == cur.section_id
                         and p.lane_id == cur.lane_id]
            cur = same_lane[0] if same_lane else next(
                (p for p in prevs if p.lane_id * cur.lane_id > 0), None
            )
            if cur is None:
                break
            tf = cur.transform
            tf.location.z += 0.5
            if not _far_enough(tf.location):
                continue
            transforms.append(tf)

    # Shuffle and trim to budget so we get an even spread across the lanes.
    random.shuffle(transforms)
    return transforms[:n_target]


def _spawn_batch(
    client,
    world,
    traffic_manager,
    transforms,
    role,
):
    """Batch-spawn 4-wheel vehicles at the given transforms under TM autopilot."""
    if not transforms:
        return []
    bp_lib = world.get_blueprint_library()
    candidates = [
        b for b in bp_lib.filter('vehicle.*')
        if b.has_attribute('number_of_wheels')
        and int(b.get_attribute('number_of_wheels')) == 4
        and b.id != config.VEHICLE_BP
    ]
    if not candidates:
        candidates = list(bp_lib.filter('vehicle.*'))

    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    batch = []
    for tf in transforms:
        bp = random.choice(candidates)
        if bp.has_attribute('color'):
            colors = bp.get_attribute('color').recommended_values
            if colors:
                bp.set_attribute('color', random.choice(colors))
        if bp.has_attribute('driver_id'):
            ids = bp.get_attribute('driver_id').recommended_values
            if ids:
                bp.set_attribute('driver_id', random.choice(ids))
        bp.set_attribute('role_name', role)
        batch.append(
            SpawnActor(bp, tf).then(SetAutopilot(FutureActor, True,
                                                traffic_manager.get_port()))
        )

    spawned = []
    for resp in client.apply_batch_sync(batch, True):
        if resp.error:
            continue
        actor = world.get_actor(resp.actor_id)
        if actor is not None:
            spawned.append(actor)
    return spawned


def _apply_traffic_personality(
    traffic_manager,
    actor,
    auto_lc_prob = 0.75,
):
    """Per-vehicle TM tweaks. Higher auto_lc_prob for swarm than background
    so cut-ins naturally happen as faster vehicles pass slower ones."""
    roll = random.random()
    if roll < 0.15:
        sd = random.uniform(20.0, 40.0)    # slower (becomes a lead)
    elif roll < 0.85:
        sd = random.uniform(-5.0, 15.0)    # near limit
    else:
        sd = random.uniform(-25.0, -5.0)   # speeder (will overtake)
    traffic_manager.vehicle_percentage_speed_difference(actor, sd)
    traffic_manager.auto_lane_change(actor, random.random() < auto_lc_prob)
    traffic_manager.distance_to_leading_vehicle(actor, random.uniform(1.5, 4.0))
    traffic_manager.ignore_lights_percentage(actor, random.uniform(0.0, 10.0))


def _spawn_ambient_traffic(
    client,
    world,
    traffic_manager,
    ego,
    n_npcs,
    avoid_locations = None,
):
    """Spawn n_npcs NPCs split between (a) lane-walking swarm around ego and
    (b) random background placement, then apply per-vehicle TM personalities."""
    if n_npcs <= 0:
        return []

    avoid = list(avoid_locations or [])
    avoid.append(ego.get_transform().location)

    # 1) Lane-walking swarm — 75% of budget
    n_swarm = int(round(n_npcs * 0.75))
    swarm_tfs = _swarm_transforms(world, ego, avoid, n_swarm)
    swarm_actors = _spawn_batch(client, world, traffic_manager,
                                swarm_tfs, role='swarm')

    # Update avoid set with spawn-points already used
    avoid.extend([a.get_transform().location for a in swarm_actors])

    # 2) Random background — remainder
    n_bg = max(0, n_npcs - len(swarm_actors))
    spawn_points = list(world.get_map().get_spawn_points())
    random.shuffle(spawn_points)
    bg_tfs = []
    for sp in spawn_points:
        if len(bg_tfs) >= n_bg:
            break
        if any(math.hypot(sp.location.x - av.x, sp.location.y - av.y)
               < _TRAFFIC_AVOID_RADIUS_M for av in avoid):
            continue
        bg_tfs.append(sp)
    bg_actors = _spawn_batch(client, world, traffic_manager,
                             bg_tfs, role='background')

    all_actors = swarm_actors + bg_actors
    # Personalities: swarm gets higher auto-LC probability so it generates
    # cut-ins naturally as speeders overtake slow vehicles.
    for actor in swarm_actors:
        _apply_traffic_personality(traffic_manager, actor, auto_lc_prob=0.75)
    for actor in bg_actors:
        _apply_traffic_personality(traffic_manager, actor, auto_lc_prob=0.5)
    return all_actors


# ----------------------------------------------------------------------------
# Scripted cut-in: a designated NPC in the adjacent lane that we explicitly
# fire tm.force_lane_change() on after a few seconds of sim time. Useful for
# deterministic interaction testing on top of the stochastic swarm.
# ----------------------------------------------------------------------------

def _spawn_cut_in_npc(
    world,
    traffic_manager,
    ego,
    side = 'left',
    ahead_m = 35.0,
):
    """
    Spawn an NPC in the adjacent lane ahead of ego, configured so a single
    later call to tm.force_lane_change(actor, to_right_bool) will cut it
    into ego's lane.

    `side` is the DRIVER-perspective side of ego where the NPC starts. The
    returned `to_right_for_cut_in` is the bool to pass to force_lane_change
    when the cut-in is triggered — already corrected for the OpenDRIVE vs
    driver direction mirroring (CARLA 가 OpenDRIVE 기준선 좌·우를 따라
    동작하므로, lane_id 부호에 따라 직접 변환).
    """
    carla_map = world.get_map()
    ego_wp = carla_map.get_waypoint(
        ego.get_transform().location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if ego_wp is None:
        return None, None

    # Pick driver-LEFT or driver-RIGHT adjacent waypoint (driver-frame swap)
    if side == 'left':
        adj_wp = (ego_wp.get_right_lane() if ego_wp.lane_id > 0
                  else ego_wp.get_left_lane())
    else:
        adj_wp = (ego_wp.get_left_lane() if ego_wp.lane_id > 0
                  else ego_wp.get_right_lane())
    if (adj_wp is None
            or adj_wp.lane_type != carla.LaneType.Driving
            or adj_wp.lane_id * ego_wp.lane_id <= 0):
        return None, None

    ahead = adj_wp.next(ahead_m)
    if not ahead:
        return None, None
    spawn_wp = ahead[0]

    bp = world.get_blueprint_library().find('vehicle.audi.tt')
    if bp is None:
        bp = world.get_blueprint_library().filter('vehicle.*')[0]
    if bp.has_attribute('color'):
        bp.set_attribute('color', '255, 200, 0')  # distinct yellow
    bp.set_attribute('role_name', 'cut_in')

    tf = spawn_wp.transform
    tf.location.z += 0.5
    actor = world.try_spawn_actor(bp, tf)
    if actor is None:
        return None, None
    actor.set_autopilot(True, traffic_manager.get_port())
    # Slightly slower so ego catches up alongside before the cut-in fires
    traffic_manager.vehicle_percentage_speed_difference(actor, 15.0)
    traffic_manager.auto_lane_change(actor, False)
    traffic_manager.distance_to_leading_vehicle(actor, 3.0)

    # NPC needs to move toward ego, i.e. opposite of where it sits relative
    # to ego. CARLA TM force_lane_change(actor, to_right=True) means "move
    # to the OpenDRIVE-right side of this lane". For lanes with lane_id > 0
    # this is mirrored relative to driver direction (OpenDRIVE 기준선 효과).
    #
    # If side='left' (NPC is to driver's LEFT of ego): NPC must move to
    # driver's RIGHT to cut in. For lane_id < 0: driver-right = OD-right
    # -> True. For lane_id > 0: driver-right = OD-left -> False.
    npc_wp = carla_map.get_waypoint(
        actor.get_transform().location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    driver_dir = 'right' if side == 'left' else 'left'
    if npc_wp.lane_id > 0:
        to_right_od = (driver_dir == 'left')
    else:
        to_right_od = (driver_dir == 'right')
    return actor, to_right_od


def _follow_ego_with_spectator(world, ego):
    spectator = world.get_spectator()
    tf = ego.get_transform()
    import math as _math
    yaw_rad = _math.radians(tf.rotation.yaw)
    back_x = tf.location.x - 8.0 * _math.cos(yaw_rad)
    back_y = tf.location.y - 8.0 * _math.sin(yaw_rad)
    cam_tf = carla.Transform(
        carla.Location(x=back_x, y=back_y, z=tf.location.z + 4.0),
        carla.Rotation(pitch=-15.0, yaw=tf.rotation.yaw, roll=0.0),
    )
    spectator.set_transform(cam_tf)


# ----------------------------------------------------------------------------
# Actuator arbitration
# ----------------------------------------------------------------------------

def _arbitrate(steer, throttle, brake, scc_state):
    """AEB wins outright. Throttle/brake mutually exclusive."""
    ctrl = carla.VehicleControl()
    if scc_state.value == 'AEB':
        ctrl.steer = float(steer)
        ctrl.throttle = 0.0
        ctrl.brake = float(max(brake, 0.0))
    else:
        ctrl.steer = float(steer)
        ctrl.throttle = float(max(throttle, 0.0)) if brake <= 1e-3 else 0.0
        ctrl.brake = float(max(brake, 0.0))
    ctrl.hand_brake = False
    ctrl.reverse = False
    ctrl.manual_gear_shift = False
    return ctrl


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default=config.CARLA_HOST)
    parser.add_argument('--port', type=int, default=config.CARLA_PORT)
    parser.add_argument('--map', default=config.MAP_NAME)
    parser.add_argument('--set-speed', type=float, default=config.SET_SPEED,
                        help='SCC cruise set speed [km/h]')
    parser.add_argument('--no-bev', action='store_true',
                        help='disable OpenCV BEV window')
    parser.add_argument('--no-npc', action='store_true',
                        help='do not spawn a slow NPC ahead')
    parser.add_argument('--ticks', type=int, default=0,
                        help='exit after N ticks; 0 = run until Ctrl-C')
    parser.add_argument('--no-route', action='store_true',
                        help='do not generate a global route')
    parser.add_argument('--destination-index', type=int, default=-1,
                        help='spawn-point index to use as destination; '
                             '-1 picks the farthest reachable one '
                             '(ignored when --scenario=overtake)')
    parser.add_argument('--stop-on-arrival', action='store_true',
                        help='exit the loop when the route is complete')
    parser.add_argument('--scenario', choices=('overtake', 'free'), default='overtake',
                        help="'overtake' (default): multi-lane straight + slow "
                             "NPC in same lane + destination 600 m ahead in same "
                             "lane, designed to provoke a discretionary lane change. "
                             "'free': random spawn + 50 m NPC + arbitrary far "
                             "destination.")
    parser.add_argument('--traffic-count', type=int, default=40,
                        help='Number of ambient traffic NPCs to spawn (0 = '
                             'disabled). 75% are lane-walking from ego (guaranteed '
                             'same-direction interaction); 25% background across '
                             'the map. Speed variance + 75% auto-LC produces '
                             'natural cut-ins as faster cars pass slower ones.')
    parser.add_argument('--traffic-seed', type=int, default=-1,
                        help='Random seed for traffic spawning. -1 = unseeded.')
    parser.add_argument('--cut-in', action='store_true',
                        help='Spawn one scripted cut-in NPC in the adjacent lane '
                             'and fire tm.force_lane_change() ~6 s after start. '
                             'Use for a deterministic interaction event.')
    parser.add_argument('--cut-in-side', choices=('left', 'right'), default='left',
                        help='Driver-perspective side of ego where the cut-in '
                             'NPC starts. It will then move toward ego.')
    parser.add_argument('--cut-in-after-s', type=float, default=6.0,
                        help='Sim seconds after start to fire the cut-in LC.')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable detailed BP/SCC diagnostic prints. '
                             'Default output is one compact status line per '
                             'second plus LC REQUESTED / CANCELLED / COMPLETE '
                             'event markers.')
    args = parser.parse_args()
    # Push the verbose flag into config so behavior_planner / scc_controller
    # / fusion_manager can gate their diagnostic prints by it.
    config.LOG_VERBOSE = bool(args.verbose)

    print(f"[CARLA] connecting to {args.host}:{args.port}")
    client = _connect(args.host, args.port, config.CARLA_TIMEOUT)

    # load the requested map if needed
    world = client.get_world()
    current_map_name = world.get_map().name.split('/')[-1]
    if args.map and current_map_name != args.map:
        print(f"[CARLA] loading {args.map} (current: {current_map_name})")
        world = client.load_world(args.map)

    original_settings = _set_sync(world, config.FIXED_DELTA)

    tm = client.get_trafficmanager(config.TM_PORT)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)

    ego = None
    npc = None
    bev = None
    fusion = None
    scenario_dest_tf = None
    traffic_npcs = []
    try:
        # Seed RNG for reproducible traffic placement when requested
        if args.traffic_seed >= 0:
            random.seed(args.traffic_seed)
            tm.set_random_device_seed(args.traffic_seed)

        # ---- scenario setup ----
        if args.scenario == 'overtake':
            ego, npc, scenario_dest_tf = _setup_overtake_scenario(world, tm)
            world.tick()
            if args.set_speed < OVERTAKE_RECOMMENDED_SET_SPEED_KMH - 1.0:
                print(f"[SCENARIO] note: --set-speed={args.set_speed} is lower than "
                      f"the recommended {OVERTAKE_RECOMMENDED_SET_SPEED_KMH:.0f} for "
                      f"this scenario; the speed gap to the NPC may be marginal.")
        else:
            ego = _spawn_ego(world)
            world.tick()
            if not args.no_npc:
                npc = _spawn_npc_ahead(world, ego, tm)
                world.tick()

        # ---- ambient traffic flow ----
        # Spawn after the scenario actors so we know which locations to avoid.
        cut_in_npc = None
        cut_in_to_right = None
        cut_in_trigger_t = None
        if args.traffic_count > 0:
            avoid = []
            if npc is not None:
                avoid.append(npc.get_transform().location)
            traffic_npcs = _spawn_ambient_traffic(
                client, world, tm, ego,
                n_npcs=args.traffic_count,
                avoid_locations=avoid,
            )
            print(f"[TRAFFIC] {len(traffic_npcs)}/{args.traffic_count} ambient NPCs spawned "
                  f"(~75% lane-walking around ego + ~25% background)")
            world.tick()

        if args.cut_in:
            cut_in_npc, cut_in_to_right = _spawn_cut_in_npc(
                world, tm, ego,
                side=args.cut_in_side, ahead_m=35.0,
            )
            if cut_in_npc is not None:
                cut_in_trigger_t = args.cut_in_after_s
                print(f"[CUT-IN] scripted NPC spawned to ego's {args.cut_in_side}; "
                      f"force_lane_change(to_right={cut_in_to_right}) "
                      f"will fire at t={cut_in_trigger_t:.1f}s")
                # Also include the cut-in NPC in the cleanup list
                traffic_npcs.append(cut_in_npc)
                world.tick()
            else:
                print("[CUT-IN] could not place cut-in NPC (no suitable adjacent lane)")

        # ----- module wiring -----
        ego_state_provider = EgoStateProvider(ego)
        lane_provider = LaneProvider(world.get_map())
        # SFOT (Sensor Fusion and Object Tracking) — 6-state KF (x,y,z,vx,vy,
        # vz) 또는 7-state CTRV, LiDAR 3D + Camera 2D pixel measurements via
        # pinhole projection, score-based track lifecycle (window=8, conf=0.7,
        # del=0.4), LiDAR-only track init, sequential
        # per-sensor association. Real CARLA sensors are spawned for pose
        # accuracy and BEV viz; detections are actor-truth filtered through
        # FOV/range and noised per spec.
        fusion = FusionManager(world, ego)
        lead_selector = LeadVehicleSelector()
        # 등속 가정 1차 motion predictor. 1.5 s horizon, 0.1 s step. LaneModel
        # 곡률이 임계 이상이면 horizon 을 0.8 s 로 자동 단축한다.
        predictor = ConstantVelocityPredictor()
        behavior_planner = BehaviorPlanner()
        local_planner = LocalPlanner()
        lateral_controller = LateralController()
        scc = SCCController()
        scc.set_cruise_speed(args.set_speed, announce=True)
        local_planner.set_speed_kmh = args.set_speed

        # ----- global route -----
        route_manager = None
        if not args.no_route:
            try:
                route_manager = RouteManager(world, sample_resolution_m=2.0)
                ego_loc = ego.get_transform().location

                # Scenario-supplied destination wins over CLI/auto-pick.
                if scenario_dest_tf is not None:
                    dest_tf = scenario_dest_tf
                elif args.destination_index >= 0:
                    sps = world.get_map().get_spawn_points()
                    if args.destination_index >= len(sps):
                        raise RuntimeError(
                            f"--destination-index {args.destination_index} out of range "
                            f"(have {len(sps)} spawn points)")
                    dest_tf = sps[args.destination_index]
                else:
                    from localization.route_manager import pick_destination_spawn
                    dest_tf = pick_destination_spawn(
                        world, ego_loc, min_distance_m=250.0, max_distance_m=1500.0,
                    )
                    if dest_tf is None:
                        raise RuntimeError("No suitable destination spawn point found")
                n_wp = route_manager.set_route(ego_loc, dest_tf.location)
                print(f"[ROUTE] destination @ ({dest_tf.location.x:.1f}, "
                      f"{dest_tf.location.y:.1f}) | {n_wp} waypoints | "
                      f"{route_manager.total_length:.0f} m total")
            except Exception as e:
                print(f"[ROUTE] disabled: {e}")
                route_manager = None

        if not args.no_bev:
            bev = BEVVisualizer()

        # ----- loop -----
        tick_count = 0
        sim_time = 0.0
        last_status_print = 0.0
        dt = config.FIXED_DELTA
        print("[RUN] entering control loop (Ctrl-C to stop)")

        while True:
            world.tick()
            tick_count += 1
            sim_time += dt

            # Scripted cut-in trigger: fire once when sim_time crosses the
            # configured threshold. tm.force_lane_change is a one-shot — it
            # commands a single LC and TM handles the actual motion.
            if (cut_in_npc is not None
                    and cut_in_trigger_t is not None
                    and sim_time >= cut_in_trigger_t
                    and cut_in_npc.is_alive):
                tm.force_lane_change(cut_in_npc, cut_in_to_right)
                print(f"[CUT-IN] t={sim_time:.2f}s  force_lane_change fired")
                cut_in_trigger_t = None  # one-shot

            # 1. ego state
            ego_state = ego_state_provider.update(sim_time)

            # 2. route context (must come BEFORE lane_provider so the lane
            #    provider can use the route hint at branching points)
            route_ctx = None
            if route_manager is not None:
                route_ctx = route_manager.update(ego_state)

            # 3. lane model
            lane = lane_provider.update(ego_state, route_context=route_ctx)

            # 4. SFOT perception pipeline:
            #    LidarManager + CameraManager pull actor-truth detections
            #    in their respective sensor frames, then FusionManager runs
            #    sequential association+update on its 6-state KF and emits
            #    confirmed tracks as FusedObject (ego frame, 2D projection).
            fused = fusion.update(ego_state, sim_time)
            # 등속 가정으로 모든 confirmed 트랙의 1.5 초 미래 위치를 예측.
            # BehaviorPlanner 의 MOBIL 평가가 worst-case (현재 vs 미래) gap
            # 으로 인접 차로의 cut-in 위협을 한 박자 먼저 잡는 데 사용된다.
            predictions = predictor.predict(fused, sim_time, lane=lane)

            # LeadVehicleSelector 가 현재 + 좌·우 인접 차선 lead 를 동시 산출.
            # SCC / 종방향 제어는 ``lead_set.current`` 만 소비해 기존 인터페이스
            # 를 유지하고, BehaviorPlanner 는 좌·우 lead 도 함께 활용한다.
            lead_set = lead_selector.update(ego_state, lane, fused, sim_time)
            lead_info = lead_set.current

            # 5. behavior decision (utility-based lane scoring + LC state machine)
            decision = behavior_planner.update(
                t=sim_time,
                ego=ego_state,
                lane=lane,
                lead=lead_info,
                scc_state_value=scc.state.value,
                set_speed_kmh=args.set_speed,
                fused_objects=fused,
                route_context=route_ctx,
                lead_set=lead_set,
                predictions=predictions,
            )

            # 6. local plan (consumes BehaviorDecision for LC trajectory)
            plan = local_planner.plan(
                ego_state, lane, lead_info,
                route_context=route_ctx,
                behavior_decision=decision,
            )

            # 7. lateral control
            steer, ctrl_dbg = lateral_controller.compute(
                ego_state, plan.trajectory, dt
            )

            # 8. longitudinal control.
            # SCC keeps tracking the lead via lane_provider/lead_vehicle_selector;
            # once ego is re-projected onto the target lane mid-LC the lead
            # naturally falls out of the current-lane filter and SCC moves to
            # SPEED_CTRL on its own.
            throttle, brake = scc.update(
                speed_kmh=ego_state.speed_kmh,
                speed_ms=ego_state.speed_mps,
                actual_accel=ego_state.accel_mps2,
                lead_info=lead_info,
                dt=dt,
                desired_speed_kmh=plan.desired_speed_kmh,
            )
            ctrl_dbg.throttle = throttle
            ctrl_dbg.brake = brake
            ctrl_dbg.desired_speed_kmh = plan.desired_speed_kmh
            ctrl_dbg.scc_state = scc.state.value

            # 8. arbitration & apply
            cmd = _arbitrate(steer, throttle, brake, scc.state)
            ego.apply_control(cmd)

            # 9. spectator + BEV
            if tick_count % 2 == 0:
                _follow_ego_with_spectator(world, ego)
            if bev is not None:
                # Pull live camera frame + LiDAR raw points from the
                # FusionManager's sensor managers so the visualizer can
                # show the camera image with object 2D bboxes (left panel)
                # and the LiDAR point cloud overlaid on the BEV (right).
                if fusion is not None:
                    cam_image = fusion.camera_mgr.get_image()
                    cam_intrinsic = fusion.camera_mgr.get_intrinsic()
                    cam_sensor_to_world = fusion.camera_mgr.get_sensor_to_world()
                    lidar_points = fusion.lidar_mgr.get_raw_points()
                else:
                    cam_image = None
                    cam_intrinsic = None
                    cam_sensor_to_world = None
                    lidar_points = None
                bev.update(
                    ego_state=ego_state,
                    lane=lane,
                    fused_objects=fused,
                    local_plan=plan,
                    lead_info=lead_info,
                    control_debug=ctrl_dbg,
                    scc_status=scc.get_status_string(),
                    route_context=route_ctx,
                    camera_image=cam_image,
                    camera_intrinsic=cam_intrinsic,
                    camera_sensor_to_world=cam_sensor_to_world,
                    lidar_raw_points=lidar_points,
                    adj_leads=lead_set,
                )

            # 10. periodic console status — one compact line per second
            if sim_time - last_status_print >= 1.0:
                last_status_print = sim_time
                # Lead: just range (drop range-rate/TTC noise from the main view)
                lead_txt = (f"lead {lead_info.range:5.1f}m"
                            if lead_info.detected else "no lead")
                # Route: along/total + optional mandatory LC hint
                if route_ctx is not None and route_ctx.is_valid:
                    route_txt = (
                        f"rt {route_ctx.distance_along_route:5.0f}/"
                        f"{route_ctx.total_route_length:5.0f}m"
                        + (" [LC->{}]".format(route_ctx.mandatory_lane_change_direction)
                           if route_ctx.mandatory_lane_change_direction else "")
                    )
                else:
                    route_txt = "no route"
                # LC state: include direction + progress % only when active
                lc_txt = decision.lc_state.value
                if decision.lc_direction:
                    lc_txt += f"->{decision.lc_direction}"
                if decision.lc_state == LaneChangeState.EXECUTE:
                    lc_txt += f" {decision.lc_progress_ratio*100:.0f}%"
                # SCC: just state name, not the full debug string
                scc_state = scc.state.value
                print(
                    f"[t={sim_time:6.1f}s] {ego_state.speed_kmh:5.1f}km/h "
                    f"| SCC {scc_state:8s} | LC {lc_txt:18s} "
                    f"| {lead_txt:14s} | {route_txt}"
                )
                if args.verbose:
                    # Verbose: tack on controller debug + lead detail (current + adj lanes)
                    extra_lead = (f"dR {lead_info.range_rate:+5.2f}"
                                  if lead_info.detected else "")
                    def _adj(lead):
                        if lead is None:
                            return "x"
                        return f"{lead.range:5.1f}" if lead.detected else "-"
                    adj_txt = f"L{_adj(lead_set.left)} R{_adj(lead_set.right)}"
                    print(
                        f"            verbose ={steer:+.3f} "
                        f"thr={throttle:.2f} brk={brake:.2f}  "
                        f"{extra_lead}  adj[{adj_txt}]"
                    )

            # 11. arrival termination
            if (route_ctx is not None and route_ctx.is_finished
                    and ego_state.speed_kmh < 1.0 and args.stop_on_arrival):
                print(f"[RUN] arrived at destination, exiting")
                break

            if args.ticks > 0 and tick_count >= args.ticks:
                print(f"[RUN] reached --ticks={args.ticks}, exiting")
                break

    except KeyboardInterrupt:
        print("\n[RUN] Ctrl-C received, shutting down")
    except Exception as e:
        import traceback
        print(f"[RUN] error: {e}")
        traceback.print_exc()
    finally:
        if bev is not None:
            bev.close()
        # FusionManager owns real CARLA sensors; destroy them before the
        # ego actor goes away so they don't dangle.
        if fusion is not None:
            try:
                fusion.destroy()
            except Exception:
                pass
        # Destroy ambient traffic first via apply_batch — much faster than
        # per-actor destroy() when you have dozens of NPCs to remove.
        if traffic_npcs:
            try:
                client.apply_batch(
                    [carla.command.DestroyActor(a) for a in traffic_npcs if a is not None]
                )
                print(f"[CLEAN] destroyed {len(traffic_npcs)} ambient NPCs")
            except Exception as e:
                print(f"[CLEAN] traffic destroy failed: {e}")
        for actor in (npc, ego):
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    pass
        tm.set_synchronous_mode(False)
        _restore_sync(world, original_settings)
        print("[CLEAN] done")


if __name__ == '__main__':
    sys.exit(main() or 0)
