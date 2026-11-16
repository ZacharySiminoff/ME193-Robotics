"""mjlab task: teach the Onshape two-wheel robot to drive forward in a straight line, ready for the real robot.

The robot is the one exported to ../09_onshape_robot/onshape/ (it is not copied here: re-export it there and this
task picks up the new version). It starts at the origin pointing a little off to one side (random yaw) and must
drive forward at TARGET_SPEED while staying on its start line and keeping its heading.

The motors behave like the real LEGO Double Motor, as measured by ../12_lego_motor/characterize.py and saved in
motor_model.json: commands are whole speed percents, arrive after the measured delay, and the wheel speed follows
with the measured time constant. If there is no motor_model.json, typical values are used (see DEFAULT_MOTOR).

The policy (actor) only sees what the real robot can measure, the way the real robot measures it:
  heading      sin and cos of the yaw from the motor's gyro (rounded to the gyro's resolution, a little noise)
  wheel_speed  each wheel's speed from its encoder: whole degrees, differenced over one control step (rad/s)
  last_action  what the policy asked for last step
Readings arrive late by the measured delay. The critic (only used in training) also sees the true velocities and
the sideways offset. ../12_lego_motor/deploy.py builds the same three observations on the real robot.

"Forward" is FORWARD below, in the robot's own frame as exported from Onshape. For the current export the wheel
axles point along x, so the robot drives along y; spinning both wheels forward moves it towards -y.

Importing this file registers the task "Drive-Forward" (see train.py, make_video.py, train_and_video.py next to it).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator import BuiltinPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  action_rate_l2,
  bad_orientation,
  base_ang_vel,
  base_lin_vel,
  joint_vel_rel,
  last_action,
  reset_root_state_uniform,
  time_out,
)
from mjlab.envs.mdp.actions import JointVelocityActionCfg
from mjlab.envs.mdp.actions.actions import JointVelocityAction
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# ---- constants you may want to change ----------------------------------------------------------------
# The Onshape export: next to this file when run by runIt.py (--with 09_onshape_robot/onshape), else in the repo.
_HERE = Path(__file__).resolve().parent
_CANDIDATES = (_HERE / "onshape" / "robot.xml", _HERE.parent / "09_onshape_robot" / "onshape" / "robot.xml")
ROBOT_XML = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])
LEFT, RIGHT = "left_wheel", "right_wheel"       # wheel joint names in robot.xml
FORWARD = (0.0, -1.0)       # the robot's forward direction (x, y) in its own frame. Both wheels forward -> -y.
TARGET_SPEED = 0.04         # m/s the robot is rewarded for going forward
MAX_WHEEL_SPEED_WANTED = 6.0  # rad/s; a policy output of 1.0 means this wheel speed (capped at the motor's top speed)
FLOOR_FRICTION = 1.5        # sliding friction of the floor (MuJoCo's default is 1.0; was 2.0)
WHEEL_KD = 0.05             # how hard the wheel joint is pushed toward the motor's speed (N*m per rad/s)
WHEEL_TORQUE_LIMIT = 0.3    # N*m
START_YAW = 0.4             # rad; the robot starts pointing up to this far off FORWARD
START_HEIGHT = 0.01         # m; the robot is dropped from this high (its lowest point is about at its origin)
MAX_OFF_LINE = 0.5          # m; the episode ends if the robot wanders this far sideways from its start line
EPISODE_S = 10.0
PHYSICS_DT = 0.005          # s
# Randomized every episode so the policy copes with a motor that is a bit different from the measured one:
TAU_SCALE = (0.7, 1.4)      # time constant x this
GAIN_SCALE = (0.9, 1.1)     # actual wheel speed x this
YAW_NOISE = 0.005           # rad, gyro yaw noise
# -------------------------------------------------------------------------------------------------------

# ---- the real motor: ../12_lego_motor/motor_model.json (sent to the cluster with --with) ----------------
_MODEL_CANDIDATES = (_HERE / "motor_model.json", _HERE.parent / "12_lego_motor" / "motor_model.json")
DEFAULT_MOTOR = {           # typical values, used until you have measured your own motor
  "control_dt": 0.05, "rad_s_per_percent": 0.14, "deadband_percent": 0.0, "max_rad_s": 14.0,
  "action_delay_s": 0.04, "action_delay_jitter_s": 0.015, "tau_s": 0.06, "tau_up_s": 0.06, "tau_down_s": 0.06,
  "obs_delay_s": 0.03, "notify_interval_s": 0.02, "encoder_resolution_deg": 1.0,
  "imu_yaw_rad_per_unit": math.pi / 1800, "imu_yaw_resolution_rad": math.pi / 1800,
}


def _load_motor_model() -> tuple[dict, str]:
  for path in _MODEL_CANDIDATES:
    if path.exists():
      return {**DEFAULT_MOTOR, **json.loads(path.read_text())}, str(path)
  return dict(DEFAULT_MOTOR), "defaults (no motor_model.json found)"


MOTOR, MOTOR_SOURCE = _load_motor_model()
CONTROL_DT = float(MOTOR["control_dt"])
DECIMATION = max(1, round(CONTROL_DT / PHYSICS_DT))
MAX_WHEEL_SPEED = min(MAX_WHEEL_SPEED_WANTED, 0.95 * float(MOTOR["max_rad_s"]))
# the order of the policy's inputs; deploy.py builds them in this order
ACTOR_OBS = (("heading", 2), ("wheel_speed", 2), ("last_action", 2))

_FWD = torch.tensor(FORWARD) / math.hypot(*FORWARD)          # unit forward direction in the world at yaw 0
_SIDE = torch.stack([-_FWD[1], _FWD[0]])                     # unit direction 90 degrees to the left of it


def _get_spec() -> mujoco.MjSpec:
  if not ROBOT_XML.exists():
    raise FileNotFoundError(
      f"robot.xml not found (looked in {[str(p) for p in _CANDIDATES]}). Run onshape-to-robot in "
      "09_onshape_robot/, and pass --with 09_onshape_robot/onshape to runIt.py."
    )
  spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
  for actuator in list(spec.actuators):     # mjlab adds its own wheel motors (below)
    spec.delete(actuator)
  chassis = spec.worldbody.first_body()
  if not any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in spec.joints):
    chassis.add_freejoint()                 # let the robot move around (newer exports already have one)
  wheel_bodies = {spec.joint(LEFT).parent.name, spec.joint(RIGHT).parent.name}
  for body in spec.bodies:                  # only the wheels should grip; everything else slides (same as drive.py)
    if body.name not in wheel_bodies:
      for g in body.geoms:
        g.friction[0] = 0.01
  return spec


def _set_floor_friction(spec: mujoco.MjSpec) -> None:
  """Give the floor FLOOR_FRICTION.

  When two things touch, MuJoCo uses the larger of their two frictions. So this applies to everything that touches
  the floor: the wheels, and the chassis too (its 0.01 above is lower, so it does not count).
  """
  for geom in spec.geoms:
    if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
      geom.friction[0] = FLOOR_FRICTION


def _robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, START_HEIGHT), joint_vel={".*": 0.0}),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        # stiffness 0 = no position target: the joint is pushed toward the speed LegoWheelAction gives it
        BuiltinPdActuatorCfg(
          target_names_expr=(LEFT, RIGHT), stiffness=0.0, damping=WHEEL_KD, effort_limit=WHEEL_TORQUE_LIMIT
        ),
      ),
    ),
  )


# ---- the motor: whole percents, delay, and lag --------------------------------------------------------


@dataclass(kw_only=True)
class LegoWheelActionCfg(JointVelocityActionCfg):
  def build(self, env: ManagerBasedRlEnv) -> LegoWheelAction:
    return LegoWheelAction(self, env)


class LegoWheelAction(JointVelocityAction):
  """Policy output in [-1, 1] -> wheel speed, the way the real Double Motor responds.

  1. the output is clipped to [-1, 1] and scaled to rad/s, then rounded to whole speed percents (what the motor takes)
  2. it reaches the motor action_delay_s later (counted in physics steps, randomized each episode)
  3. the wheel speed follows it with a first-order lag (tau_up_s speeding up, tau_down_s slowing down)
  """

  def __init__(self, cfg: LegoWheelActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    n, dev = self.num_envs, self.device
    self._dt = env.physics_dt
    base = MOTOR["action_delay_s"] / self._dt
    jitter = max(MOTOR["action_delay_jitter_s"] / self._dt, 1.0)
    self._dmin = max(0, int(math.floor(base - jitter)))
    self._dmax = max(self._dmin, int(math.ceil(base + jitter)))
    self._buf = torch.zeros(n, self._dmax + 1, self.action_dim, device=dev)
    self._head = 0
    self._delay = torch.zeros(n, dtype=torch.long, device=dev)
    self._speed = torch.zeros(n, self.action_dim, device=dev)
    self._a_up = torch.zeros(n, 1, device=dev)
    self._a_down = torch.zeros(n, 1, device=dev)
    self._gain = torch.ones(n, 1, device=dev)
    self._arange = torch.arange(n, device=dev)
    self._resample(slice(None))

  def _resample(self, env_ids) -> None:
    ids = self._arange[env_ids]
    k, dev = len(ids), self.device

    def uniform(lo: float, hi: float) -> torch.Tensor:
      return lo + (hi - lo) * torch.rand(k, 1, device=dev)

    self._delay[ids] = torch.randint(self._dmin, self._dmax + 1, (k,), device=dev)
    scale = uniform(*TAU_SCALE)
    self._a_up[ids] = 1 - torch.exp(-self._dt / (MOTOR["tau_up_s"] * scale).clamp(min=1e-4))
    self._a_down[ids] = 1 - torch.exp(-self._dt / (MOTOR["tau_down_s"] * scale).clamp(min=1e-4))
    self._gain[ids] = uniform(*GAIN_SCALE)

  def process_actions(self, actions: torch.Tensor):
    self._raw_actions[:] = actions
    step = float(MOTOR["rad_s_per_percent"])
    rad_s = actions.clamp(-1.0, 1.0) * MAX_WHEEL_SPEED
    self._processed_actions = torch.round(rad_s / step) * step

  def apply_actions(self) -> None:
    # called every physics step: remember this step's command, use the one from `delay` physics steps ago
    self._head = (self._head + 1) % (self._dmax + 1)
    self._buf[:, self._head] = self._processed_actions
    target = self._buf[self._arange, (self._head - self._delay) % (self._dmax + 1)] * self._gain
    alpha = torch.where(target.abs() > self._speed.abs(), self._a_up, self._a_down)
    self._speed += alpha * (target - self._speed)
    self._entity.set_joint_velocity_target(self._speed, joint_ids=self._target_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    env_ids = slice(None) if env_ids is None else env_ids
    super().reset(env_ids)
    self._buf[env_ids] = 0.0
    self._speed[env_ids] = 0.0
    self._resample(env_ids)


# ---- what the robot is doing relative to "straight along FORWARD" --------------------------------------


def _yaw(env: ManagerBasedRlEnv) -> torch.Tensor:
  """How far the robot has turned from its exported orientation. 0 = facing FORWARD, counter-clockwise positive."""
  w, x, y, z = env.scene["robot"].data.root_link_quat_w.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _xy(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Robot position in the ground plane relative to its env's start point. [num_envs, 2]."""
  return (env.scene["robot"].data.root_link_pos_w - env.scene.env_origins)[:, :2]


def forward_distance(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _xy(env) @ _FWD.to(env.device)


def off_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Sideways distance from the start line (positive = to the robot's left), in metres."""
  return _xy(env) @ _SIDE.to(env.device)


def forward_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["robot"].data.root_link_lin_vel_w[:, :2] @ _FWD.to(env.device)


# ---- observations: the real robot's sensors (actor) --------------------------------------------------------


def gyro_heading(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[sin(yaw), cos(yaw)] as the Double Motor's gyro reports it: rounded to its resolution, plus a little noise."""
  res = float(MOTOR["imu_yaw_resolution_rad"])
  yaw = torch.round(_yaw(env) / res) * res + YAW_NOISE * torch.randn(env.num_envs, device=env.device)
  return torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)


class encoder_wheel_speed:
  """Each wheel's speed (rad/s) the way deploy.py gets it: change in whole encoder degrees over one control step."""

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    robot = env.scene["robot"]
    self._ids = [robot.joint_names.index(LEFT), robot.joint_names.index(RIGHT)]
    self._res = math.radians(float(MOTOR["encoder_resolution_deg"]))
    self._prev = torch.zeros(env.num_envs, 2, device=env.device)
    self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    self._out = torch.zeros(env.num_envs, 2, device=env.device)
    self._stamp = -1

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    if self._stamp == env.common_step_counter:      # computed once per step
      return self._out
    self._stamp = env.common_step_counter
    q = env.scene["robot"].data.joint_pos[:, self._ids]
    q = torch.round(q / self._res) * self._res
    speed = (q - self._prev) / env.step_dt
    self._out = torch.where(self._fresh[:, None], torch.zeros_like(speed), speed)
    self._prev = q
    self._fresh[:] = False
    return self._out

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._fresh[slice(None) if env_ids is None else env_ids] = True
    self._stamp = -1


def heading_and_offset(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[sin(yaw), cos(yaw), off-line distance], exact (critic only). Shape [num_envs, 3]."""
  yaw = _yaw(env)
  return torch.stack([torch.sin(yaw), torch.cos(yaw), off_line(env)], dim=-1)


def _obs_lag_steps() -> tuple[int, int]:
  """How many control steps old the newest reading is when the policy runs.

  At least the fastest Bluetooth trip (obs_delay_s); at most that, plus a Bluetooth hold-up (90 % of readings are held
  up less than reading_late_p90_s), plus up to one update interval since the reading was taken.
  """
  lo = MOTOR["obs_delay_s"] / CONTROL_DT
  hi = (MOTOR["obs_delay_s"] + MOTOR.get("reading_late_p90_s", 0.0) + MOTOR["notify_interval_s"]) / CONTROL_DT
  return int(round(lo)), max(int(round(lo)), int(round(hi)))


# ---- rewards and terminations ------------------------------------------------------------------------------


def forward_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward for going FORWARD at TARGET_SPEED: 0 when standing still, 1 at the target, less beyond, negative backwards.

  It slopes everywhere (a narrow bump around the target would give a robot that starts out still nothing to learn from).
  """
  v = forward_velocity(env)
  return torch.clamp(1.0 - torch.square((v - TARGET_SPEED) / TARGET_SPEED), min=-1.0)


def stay_on_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  return -torch.square(off_line(env) / 0.1)     # -1 at 10 cm off the line


def face_forward(env: ManagerBasedRlEnv) -> torch.Tensor:
  return -torch.square(_yaw(env))


def too_far_off_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  return off_line(env).abs() > MAX_OFF_LINE


# ---- environment ----------------------------------------------------------------------------------------


def drive_forward_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  lag_lo, lag_hi = _obs_lag_steps()
  late = {"delay_min_lag": lag_lo, "delay_max_lag": lag_hi}
  actor_terms = {                 # keep the same order as ACTOR_OBS
    "heading": ObservationTermCfg(func=gyro_heading, **late),
    "wheel_speed": ObservationTermCfg(func=encoder_wheel_speed, **late),
    "last_action": ObservationTermCfg(func=last_action),
  }
  critic_terms = {
    "lin_vel": ObservationTermCfg(func=base_lin_vel),
    "ang_vel": ObservationTermCfg(func=base_ang_vel),
    "heading_offset": ObservationTermCfg(func=heading_and_offset),
    "wheel_vel": ObservationTermCfg(func=joint_vel_rel),
    **actor_terms,
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
    "critic": ObservationGroupCfg(critic_terms),
  }

  actions: dict[str, ActionTermCfg] = {
    "wheels": LegoWheelActionCfg(
      entity_name="robot", actuator_names=(LEFT, RIGHT), scale=1.0, use_default_offset=False
    ),
  }

  events = {
    "reset_pose": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {"yaw": (-START_YAW, START_YAW)}, "velocity_range": {}},
    ),
  }

  rewards = {
    "forward_speed": RewardTermCfg(func=forward_speed, weight=1.0),
    "stay_on_line": RewardTermCfg(func=stay_on_line, weight=0.5),
    "face_forward": RewardTermCfg(func=face_forward, weight=0.5),
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.05),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "tipped_over": TerminationTermCfg(func=bad_orientation, params={"limit_angle": math.radians(60.0)}),
    "off_line": TerminationTermCfg(func=too_far_off_line),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": _robot_cfg()},
      num_envs=1,
      env_spacing=2.0,
      spec_fn=_set_floor_friction,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="chassis",
      distance=0.5,
      elevation=-25.0,
      azimuth=140.0,
    ),
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=PHYSICS_DT)),
    decimation=DECIMATION,
    episode_length_s=1e10 if play else EPISODE_S,
  )


def drive_forward_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(64, 64), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="drive_forward",
    logger="tensorboard",           # the default, wandb, needs an account and a login
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=500,
  )


print(f"[Drive-Forward] motor model: {MOTOR_SOURCE}; control every {CONTROL_DT * 1000:.0f} ms, "
      f"max wheel speed {MAX_WHEEL_SPEED:.2f} rad/s, actor observations {[n for n, _ in ACTOR_OBS]}")

register_mjlab_task(
  task_id="Drive-Forward",
  env_cfg=drive_forward_env_cfg(),
  play_env_cfg=drive_forward_env_cfg(play=True),
  rl_cfg=drive_forward_ppo_runner_cfg(),
)
