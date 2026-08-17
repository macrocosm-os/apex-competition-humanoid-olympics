"""Shared MuJoCo physics and event-neutral gates for Humanoid Olympics."""

from __future__ import annotations

import gc
import math
import pathlib
from dataclasses import dataclass, field
from typing import Mapping

import mujoco
import numpy as np

from .course import (EVENT_MAX_STEPS, EVENTS, GEOM_PREFIX, OVERHEAD_GROUP, PLINTH_TOP,
                     TRACK_HALF_W, WORLD_GROUP, EventLayout, build_event, course_xml_fragment,
                     sample_frictions)

ASSETS = pathlib.Path(__file__).parent / "assets"

ACT_DIM = 12
STATE_DIM = 256
SCAN_NX, SCAN_NY = 9, 5
OVERHEAD_N = 7
OBS_DIM = 52 + SCAN_NX * SCAN_NY + OVERHEAD_N

# At 6 m/s the inherited 1.6 m horizon was only 0.27 seconds. These scans retain
# the 104-wide signature while letting a fast policy see its next hurdle or bend.
SCAN_X = np.linspace(-0.4, 6.0, SCAN_NX)
SCAN_Y = np.linspace(-0.7, 0.7, SCAN_NY)
OVERHEAD_X = np.linspace(0.0, 4.0, OVERHEAD_N)
SCAN_CLIP = 2.0

PHYS_DT = 0.002
FRAME_SKIP = 10
ACTION_SCALE = 0.25
QVEL_GLITCH_LIMIT = 100.0
RESET_NOISE = 0.01
FALL_CLEARANCE = 0.45
UPRIGHT_MIN = 0.40
RAY_FROM_ABOVE = 3.0
AIR_DENSITY = 1.204
WIND_MAX_MS = 8.0

KP = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], np.float64)
KD = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], np.float64)
DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                           -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], np.float64)
GAIT_PERIOD = 0.8

HIGH_JUMP_BARS = (0.80, 0.92, 1.04, 1.16)


class InvalidAction(ValueError):
    """The player returned something other than a finite 12-vector."""


@dataclass(frozen=True)
class StepResult:
    obs: np.ndarray
    terminal_reason: str | None


@dataclass(frozen=True)
class InstanceParams:
    event: str
    attempt: int
    seed: int
    friction_level: float
    wind_speed: float
    wind_dir: float
    challenge: Mapping[str, float] = field(default_factory=dict)

    @property
    def wind(self) -> tuple[float, float, float]:
        return (-self.wind_speed * float(np.cos(self.wind_dir)),
                -self.wind_speed * float(np.sin(self.wind_dir)), 0.0)


def event_instances(instances_per_event: int, seed: int,
                    wind_max: float = WIND_MAX_MS) -> list[InstanceParams]:
    """A balanced, seed-derived meet: each discipline gets the same trial count."""
    if instances_per_event < 1:
        raise ValueError("instances_per_event must be >= 1")
    # Rotating the event order changes presentation but keeps models grouped, so
    # the evaluator needs at most one compiled G1 scene in memory at a time.
    offset = int(seed % len(EVENTS))
    order = EVENTS[offset:] + EVENTS[:offset]
    return [instance_spec(event, attempt, seed, wind_max, attempts=instances_per_event)
            for event in order for attempt in range(instances_per_event)]


def instance_spec(event: str, attempt: int, seed: int, wind_max: float = WIND_MAX_MS,
                  attempts: int = 4) -> InstanceParams:
    """Deterministically stratified conditions for one event attempt."""
    if event not in EVENTS:
        raise ValueError(f"unknown Olympic event {event!r}")
    event_index = EVENTS.index(event)
    if not 0 <= attempt < attempts:
        raise ValueError(f"attempt must be in [0, {attempts}), got {attempt}")
    rng = np.random.default_rng([seed, event_index, 0x0A11])
    # A shifted lattice is steadier than independent draws while still producing
    # a fresh suite each round. Wind direction uses an antithetic pair.
    phase = float(rng.uniform(0.0, 1.0))
    friction = ((attempt + phase) / attempts) % 1.0
    wind_speed = wind_max * (((attempt + 0.5 + phase) / attempts) % 1.0)
    wind_dir = (2.0 * math.pi * phase + (attempt // 2) * math.pi / 2 +
                (attempt % 2) * math.pi) % (2.0 * math.pi)
    challenge: dict[str, float] = {}
    if event == "high_jump":
        challenge["bar_height_m"] = HIGH_JUMP_BARS[attempt % len(HIGH_JUMP_BARS)]
    episode_rng = np.random.default_rng([seed, event_index, attempt, 0x5151])
    return InstanceParams(event=event, attempt=attempt, seed=int(episode_rng.integers(1 << 31)),
                          friction_level=friction,
                          wind_speed=wind_speed, wind_dir=wind_dir, challenge=challenge)


def _scene_xml(layout: EventLayout) -> str:
    robot = (ASSETS / "g1_12dof.xml").read_text()
    floor = (f'    <geom name="floor" type="plane" size="180 180 0.1" pos="0 0 0" '
             f'condim="3" group="{WORLD_GROUP}" rgba=".15 .16 .19 1"/>\n')
    return robot.replace("</worldbody>", floor + course_xml_fragment(layout) + "\n  </worldbody>")


_MODEL: mujoco.MjModel | None = None
_MODEL_KEY: str | None = None
_COURSE_GEOMS: list[int] = []


def _shared_model(layout: EventLayout) -> tuple[mujoco.MjModel, list[int]]:
    """Compile only the current event; six G1 mesh models would exceed the cap."""
    global _MODEL, _MODEL_KEY, _COURSE_GEOMS
    # Challenge values are runtime fields (the high-jump bar position), so one
    # compiled scene per event is sufficient even when attempts use new bars.
    key = layout.event
    if _MODEL is None or _MODEL_KEY != key:
        _MODEL = None
        _COURSE_GEOMS = []
        gc.collect()
        model = mujoco.MjModel.from_xml_string(_scene_xml(layout), _mesh_assets())
        model.opt.timestep = PHYS_DT
        model.opt.density = AIR_DENSITY
        _MODEL = model
        _MODEL_KEY = key
        _COURSE_GEOMS = [model.geom(f"{GEOM_PREFIX}{i}").id for i in range(len(layout.surfaces))]
    return _MODEL, _COURSE_GEOMS


class OlympicsSim:
    """One attempt in one Olympic discipline.

    Course-relative heading/lateral channels retain the old public ONNX shape,
    but now describe the tangent of the active event's route rather than +x.
    """

    def __init__(self, params: InstanceParams):
        self.params = params
        self.layout = build_event(params.event, params.challenge)
        rng = np.random.default_rng([params.seed, 0xC0FFEE])
        self.frictions = sample_frictions(self.layout, params.friction_level, rng)
        self.model, geoms = _shared_model(self.layout)
        for gid, mu in zip(geoms, self.frictions):
            self.model.geom_friction[gid, 0] = mu
        if self.event == "high_jump":
            bar_index = next(i for i, surface in enumerate(self.layout.surfaces) if surface.kind == "bar")
            self.model.geom_pos[geoms[bar_index], 2] = PLINTH_TOP + float(
                self.layout.challenge["bar_height_m"])
        self.model.opt.wind[:] = params.wind
        self.data = mujoco.MjData(self.model)
        self._pelvis = self.model.body("pelvis").id
        self._foot_bodies = {
            self.model.body("left_ankle_roll_link").id: "left",
            self.model.body("right_ankle_roll_link").id: "right",
        }
        self._surface_geom_ids = {geoms[i] for i, surface in enumerate(self.layout.surfaces)
                                  if surface.walkable} | {self.model.geom("floor").id}
        self._obstacle_geom_ids = {
            surface.kind: geoms[i] for i, surface in enumerate(self.layout.surfaces)
            if not surface.walkable
        }
        self._ray_mask = np.zeros(6, np.uint8)
        self._ray_mask[WORLD_GROUP] = 1
        self._up_mask = np.zeros(6, np.uint8)
        self._up_mask[WORLD_GROUP] = 1
        self._up_mask[OVERHEAD_GROUP] = 1
        self._geomid = np.zeros(1, np.int32)
        self._action = np.zeros(ACT_DIM)
        self.steps = 0
        self.max_x = self.layout.start_x
        self._circle_prev = 0.0
        self._circle_distance = 0.0
        self._best_clearance = 0.0
        self._jump_distance = 0.0
        self._jump_landed = False
        self._triple_phase = 0
        self._hop_foot: str | None = None
        self._event_reason: str | None = None

    @property
    def event(self) -> str:
        return self.params.event

    @property
    def max_steps(self) -> int:
        return EVENT_MAX_STEPS[self.event]

    @property
    def progress(self) -> float:
        if self.layout.is_circular:
            return float(np.clip(self._circle_distance / self.layout.finish, 0.0, 1.0))
        start = self.layout.start_x
        return float(np.clip((self.max_x - start) / max(self.layout.finish - start, 1e-6), 0.0, 1.0))

    @property
    def distance_m(self) -> float:
        """Maximum route distance, rather than a world-axis coordinate."""
        if self.layout.is_circular:
            return max(0.0, self._circle_distance)
        return max(0.0, self.max_x - self.layout.start_x)

    @property
    def metrics(self) -> dict[str, float]:
        return {"bar_height_m": float(self.layout.challenge.get("bar_height_m", 0.0)),
                "best_clearance_m": self._best_clearance,
                "jump_distance_m": self._jump_distance,
                "triple_phase": float(self._triple_phase)}

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng([self.params.seed, 0xBADA55])
        self.data.qpos[0] = self.layout.start_x + float(rng.uniform(-RESET_NOISE, RESET_NOISE))
        self.data.qpos[1] = self.layout.start_y + float(rng.uniform(-RESET_NOISE, RESET_NOISE))
        self.data.qpos[2] = PLINTH_TOP + 0.793
        half = self.layout.start_yaw / 2.0
        self.data.qpos[3:7] = [math.cos(half), 0.0, 0.0, math.sin(half)]
        self.data.qpos[7:] = DEFAULT_ANGLES + rng.uniform(-RESET_NOISE, RESET_NOISE, ACT_DIM)
        self.data.qvel[:] = rng.uniform(-RESET_NOISE, RESET_NOISE, self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.max_x = float(self.data.qpos[0])
        self._circle_prev = math.atan2(float(self.data.qpos[1]), float(self.data.qpos[0]))
        self._circle_distance = 0.0
        self._best_clearance = 0.0
        self._jump_distance = 0.0
        self._jump_landed = False
        self._triple_phase = 0
        self._hop_foot = None
        self._event_reason = None
        return self._obs()

    def step(self, action, max_steps: int | None = None) -> StepResult:
        a = np.asarray(action, dtype=np.float64).ravel()
        if a.shape != (ACT_DIM,):
            raise InvalidAction(f"action must be {ACT_DIM} floats, got shape {a.shape}")
        if not np.all(np.isfinite(a)):
            raise InvalidAction("action must contain only finite floats")
        self._action = np.clip(a, -10.0, 10.0)
        target = self._action * ACTION_SCALE + DEFAULT_ANGLES
        for _ in range(FRAME_SKIP):
            self.data.ctrl[:] = (target - self.data.qpos[7:]) * KP - self.data.qvel[6:] * KD
            mujoco.mj_step(self.model, self.data)
            self._observe_physics_step()
            if self._event_reason is not None:
                break
        self.steps += 1
        reason = self._event_reason or self._terminal(max_steps or self.max_steps)
        return StepResult(obs=self._obs(), terminal_reason=reason)

    # -- route and event rules ------------------------------------------------------------

    def _observe_physics_step(self) -> None:
        x, y, z = (float(v) for v in self.data.qpos[:3])
        self.max_x = max(self.max_x, x)
        if self.layout.is_circular:
            angle = math.atan2(y, x)
            delta = (angle - self._circle_prev + math.pi) % (2 * math.pi) - math.pi
            self._circle_distance += (400.0 / (2 * math.pi)) * delta
            self._circle_prev = angle
        if abs(self._route()[1]) > TRACK_HALF_W:
            self._event_reason = "out_of_bounds"
            return
        if self.event == "hurdles_100" and self._hits("hurdle"):
            self._event_reason = "hurdle_hit"
        elif self.event == "high_jump":
            self._observe_high_jump(x, z)
        elif self.event in {"long_jump", "triple_jump"}:
            self._observe_jump(x)
        elif self.event in {"sprint_100", "hurdles_100"} and x >= self.layout.finish:
            self._event_reason = "completed"
        elif self.event == "sprint_400" and self._circle_distance >= 400.0:
            self._event_reason = "completed"

    def _observe_high_jump(self, x: float, z: float) -> None:
        bar_x = float(self.layout.challenge["bar_x_m"])
        height = float(self.layout.challenge["bar_height_m"])
        if self._hits("bar"):
            self._event_reason = "bar_hit"
            return
        if abs(x - bar_x) <= 0.12:
            self._best_clearance = max(self._best_clearance, z - PLINTH_TOP)
        if x >= bar_x + 0.45:
            self._event_reason = "cleared" if self._best_clearance >= height + 0.06 else "bar_missed"

    def _observe_jump(self, x: float) -> None:
        takeoff = float(self.layout.challenge["takeoff_x_m"])
        landing = float(self.layout.challenge["landing_x_m"])
        contacts = self._foot_contacts()
        if self.event == "triple_jump":
            self._observe_triple_phase(x, contacts)
        # The final sand is the only terminal landing. Feet are used rather than
        # pelvis height so a clean flight through the edge is not scored early.
        if x >= landing and contacts and (self.event != "triple_jump" or self._triple_phase >= 2):
            if not self._jump_landed:
                self._jump_distance = max(0.0, x - takeoff)
                self._jump_landed = True
                self._event_reason = "landed"

    def _observe_triple_phase(self, x: float, contacts: set[str]) -> None:
        if not contacts:
            return
        if self._triple_phase == 0 and 13.5 <= x <= 15.5:
            self._hop_foot = sorted(contacts)[0]
            self._triple_phase = 1
        elif self._triple_phase == 1 and 17.0 <= x <= 18.8:
            if self._hop_foot is not None and any(side != self._hop_foot for side in contacts):
                self._triple_phase = 2

    def _foot_contacts(self) -> set[str]:
        contacts: set[str] = set()
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[contact.geom1])
            b2 = int(self.model.geom_bodyid[contact.geom2])
            if b1 in self._foot_bodies and contact.geom2 in self._surface_geom_ids:
                contacts.add(self._foot_bodies[b1])
            if b2 in self._foot_bodies and contact.geom1 in self._surface_geom_ids:
                contacts.add(self._foot_bodies[b2])
        return contacts

    def _hits(self, kind: str) -> bool:
        gid = self._obstacle_geom_ids.get(kind)
        if gid is None:
            return False
        return any(self.data.contact[i].geom1 == gid or self.data.contact[i].geom2 == gid
                   for i in range(self.data.ncon))

    # -- perception -----------------------------------------------------------------------

    def _yaw(self) -> float:
        qw, qx, qy, qz = self.data.qpos[3:7]
        return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))

    def _route(self) -> tuple[float, float, float]:
        """(target yaw, signed cross-track offset, remaining metres)."""
        x, y = (float(v) for v in self.data.qpos[:2])
        if self.layout.is_circular:
            radius = float(self.layout.challenge["radius_m"])
            theta = math.atan2(y, x)
            return theta + math.pi / 2, math.hypot(x, y) - radius, max(0.0, 400.0 - self._circle_distance)
        return 0.0, y, max(0.0, self.layout.finish - x)

    def _ray_down(self, x: float, y: float, z_from: float) -> float:
        d = mujoco.mj_ray(self.model, self.data, np.array([x, y, z_from]),
                          np.array([0.0, 0.0, -1.0]), self._ray_mask, 1, -1, self._geomid)
        return z_from - d if d >= 0 else -SCAN_CLIP

    def _ray_up(self, x: float, y: float, z_from: float) -> float:
        d = mujoco.mj_ray(self.model, self.data, np.array([x, y, z_from]),
                          np.array([0.0, 0.0, 1.0]), self._up_mask, 1, -1, self._geomid)
        return SCAN_CLIP if d < 0 else min(d, SCAN_CLIP)

    def _obs(self) -> np.ndarray:
        d, yaw = self.data, self._yaw()
        px, py, pz = (float(v) for v in d.qpos[:3])
        c, s = np.cos(yaw), np.sin(yaw)
        route_yaw, lateral, remaining = self._route()
        heading_error = (yaw - route_yaw + math.pi) % (2 * math.pi) - math.pi
        rot = np.array(d.xmat[self._pelvis]).reshape(3, 3)
        lin = rot.T @ d.qvel[:3]
        ang = rot.T @ d.qvel[3:6]
        grav = rot.T @ np.array([0.0, 0.0, -1.0])
        scan = np.empty(SCAN_NX * SCAN_NY)
        k = 0
        for dx in SCAN_X:
            for dy in SCAN_Y:
                wx, wy = px + c * dx - s * dy, py + s * dx + c * dy
                scan[k] = self._ray_down(wx, wy, pz + RAY_FROM_ABOVE) - pz
                k += 1
        np.clip(scan, -SCAN_CLIP, SCAN_CLIP, out=scan)
        over = np.array([self._ray_up(px + c * dx, py + s * dx, pz + 0.05) for dx in OVERHEAD_X])
        ground = self._ray_down(px, py, pz + RAY_FROM_ABOVE)
        phase = (self.steps * PHYS_DT * FRAME_SKIP % GAIT_PERIOD) / GAIT_PERIOD
        return np.concatenate([
            grav, ang * 0.25, lin * 2.0, d.qpos[7:] - DEFAULT_ANGLES, d.qvel[6:] * 0.05,
            self._action, [np.sin(2 * math.pi * phase), np.cos(2 * math.pi * phase)],
            [np.sin(heading_error), np.cos(heading_error)], [lateral, remaining / 10.0,
             np.clip(pz - ground, -SCAN_CLIP, SCAN_CLIP)], scan, over,
        ]).astype(np.float32)

    # -- common gates ---------------------------------------------------------------------

    def _terminal(self, max_steps: int) -> str | None:
        qpos, qvel = self.data.qpos, self.data.qvel
        if not (np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel))):
            return "physics_glitch"
        if np.max(np.abs(qvel)) > QVEL_GLITCH_LIMIT:
            return "physics_glitch"
        px, py, pz = (float(v) for v in qpos[:3])
        _, lateral, _ = self._route()
        if abs(lateral) > TRACK_HALF_W:
            return "out_of_bounds"
        if float(self.data.xmat[self._pelvis].reshape(3, 3)[2, 2]) < UPRIGHT_MIN:
            return "fell"
        if pz - self._ray_down(px, py, pz + RAY_FROM_ABOVE) < FALL_CLEARANCE:
            return "fell"
        if self.event in {"sprint_100", "hurdles_100"} and px >= self.layout.finish:
            return "completed"
        if self.event == "sprint_400" and self._circle_distance >= 400.0:
            return "completed"
        if self.steps >= max_steps:
            return "timeout"
        return None


def _mesh_assets() -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in (ASSETS / "meshes").glob("*.STL")}
