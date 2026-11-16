"""mjlab task: two paddles keep a ball going back and forth between them, like a ping-pong rally.

Same ball as 08_demo.py (radius 0.1 m, contact solref="-20000 -10"). Two paddles face each other GAP metres apart.
Each paddle can slide sideways (x), move up and down (z) and tilt (pitch about y). Tilting is what sends the ball
across: a flat paddle only bounces it straight up.

The rules:
  * The ball is served from the middle at the target height, towards a random paddle.
  * Only the paddle whose turn it is may hit the ball. After a good hit it is the other paddle's turn.
    Hitting it twice in a row (or out of turn) is a fault and ends the episode.
  * Dropping the ball ends the episode.

The rewards:
  * hit:          +1 for every hit made in turn.
  * flight:       every step the ball is flying, up to 1 per second: how close its peak will be to TARGET_APEX
                  times how close it will come down to the paddle whose turn it is. This is what makes the hits
                  "nice": a ball sent to the right height and right onto the other paddle earns the most.
  * fault:        -5 for a double hit or a dropped ball.
  * action_rate:  a small penalty for jerky paddle motion.

One policy controls both paddles. Importing this file registers the task "PingPong-Rally" (see train.py and
make_video.py next to it). Everything you will want to tune is in the constants block and the reward weights.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator import BuiltinPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import action_rate_l2, last_action, time_out
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.events import reset_scene_to_default, resolve_env_ids
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# ---- constants you may want to change ----------------------------------------------------------------
TARGET_APEX = 1.2           # m; height of the ball's centre at the top of each flight
GAP = 1.2                   # m; distance between the two paddles' centres (left at x = -GAP/2, right at +GAP/2)
PADDLE_Z = 0.5              # m; paddle height when the policy asks for 0
PADDLE_RADIUS = 0.2         # m
X_RANGE = 0.2               # m; each paddle can slide this far towards or away from the other one
Z_RANGE = 0.3               # m; and this far up or down
PITCH_RANGE = 0.6           # rad (34 deg); and tilt this far. A policy output of 1.0 means the full range.
BALL_RADIUS = 0.1           # m; same ball as 08_demo.py
BALL_MASS = 0.2             # kg
BALL_SOLREF = "-20000 -10"  # contact stiffness and damping, same as 08_demo.py. Less damping = bouncier.
PADDLE_MASS = 2.0           # kg
SLIDE_KP, SLIDE_KD, SLIDE_FORCE = 2000.0, 100.0, 300.0   # N/m, N*s/m, N   for the x and z motors
TILT_KP, TILT_KD, TILT_TORQUE = 40.0, 1.5, 20.0          # N*m/rad, N*m*s/rad, N*m   for the tilt motor
SERVE_Z_JITTER = 0.1        # m; the serve starts up to this far above or below TARGET_APEX
SERVE_SPEED_JITTER = 0.1    # the serve's sideways speed is off by up to this fraction (0.1 = 10 %)
SERVE_XY_JITTER = 0.03      # m; the serve starts up to this far off-centre
HIT_GAP_S = 0.1             # s; the ball must be off a paddle this long before touching it again counts as a new hit
APEX_SIGMA = 0.2            # m; the flight reward falls to 0.37 when the peak is this far from TARGET_APEX
LAND_SIGMA = 0.25           # m; ... or when the ball will come down this far from the next paddle's centre
LOST_Z = 0.25               # m; the ball is dropped once its centre is below this (under the lowest paddle reach)
EPISODE_S = 10.0
# -------------------------------------------------------------------------------------------------------

GRAVITY = 9.81
PADDLE_HALF_THICKNESS = 0.02
HIT_Z = PADDLE_Z + PADDLE_HALF_THICKNESS + BALL_RADIUS   # ball centre height when it touches a resting paddle
HOME_X = (-GAP / 2, GAP / 2)                             # paddle 0 = left, paddle 1 = right
PADDLES = ("left_paddle", "right_paddle")
SENSORS = ("ball_left", "ball_right")
JOINTS = ("slide", "lift", "tilt")

_BALL_XML = f"""
<mujoco>
  <worldbody>
    <body name="ball">
      <freejoint name="ball_free"/>
      <geom name="ball_geom" type="sphere" size="{BALL_RADIUS}" mass="{BALL_MASS}" rgba="1 .3 .2 1" solref="{BALL_SOLREF}"/>
    </body>
  </worldbody>
</mujoco>
"""


# mjlab places the paddle's base at its home position; the three joints move it from there.
def _paddle_xml(rgba: str) -> str:
  return f"""
<mujoco>
  <compiler angle="radian"/>  <!-- without this MuJoCo reads the tilt range in degrees -->
  <worldbody>
    <body name="paddle">
      <joint name="slide" type="slide" axis="1 0 0" range="{-X_RANGE} {X_RANGE}"/>
      <joint name="lift" type="slide" axis="0 0 1" range="{-Z_RANGE} {Z_RANGE}"/>
      <joint name="tilt" type="hinge" axis="0 1 0" range="{-PITCH_RANGE} {PITCH_RANGE}"/>
      <geom name="paddle_geom" type="cylinder" size="{PADDLE_RADIUS} {PADDLE_HALF_THICKNESS}" mass="{PADDLE_MASS}"
            rgba="{rgba}" solref="{BALL_SOLREF}"/>
    </body>
  </worldbody>
</mujoco>
"""


def _ball_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(_BALL_XML),
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, TARGET_APEX)),
  )


def _paddle_cfg(home_x: float, rgba: str) -> EntityCfg:
  return EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(_paddle_xml(rgba)),
    init_state=EntityCfg.InitialStateCfg(
      pos=(home_x, 0.0, PADDLE_Z), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}
    ),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPdActuatorCfg(
          target_names_expr=("slide", "lift"), stiffness=SLIDE_KP, damping=SLIDE_KD, effort_limit=SLIDE_FORCE
        ),
        BuiltinPdActuatorCfg(
          target_names_expr=("tilt",), stiffness=TILT_KP, damping=TILT_KD, effort_limit=TILT_TORQUE
        ),
      ),
    ),
  )


def _ball_contact_sensor(name: str, paddle: str, decimation: int) -> ContactSensorCfg:
  # history_length = decimation: remembers every physics substep of the last policy step, so a short touch
  # that starts and ends between two policy steps is still seen.
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(mode="geom", pattern="ball_geom", entity="ball"),
    secondary=ContactMatch(mode="geom", pattern="paddle_geom", entity=paddle),
    fields=("found", "force"),
    history_length=decimation,
  )


# ---- the rally: whose turn it is, and what happened this step ------------------------------------------


class _Rally:
  """Per-environment rally state, shared by the observations, rewards and terminations.

  expected[i]  which paddle must hit next in env i (0 = left, 1 = right)
  hits[i]      good hits so far this episode
  valid_hit[i] a paddle hit the ball in turn during the last step
  fault[i]     a paddle hit the ball out of turn (a double hit) during the last step
  """

  def __init__(self, env: ManagerBasedRlEnv):
    n, dev = env.num_envs, env.device
    self.expected = torch.ones(n, dtype=torch.long, device=dev)
    self.hits = torch.zeros(n, dtype=torch.long, device=dev)
    self.away_s = torch.full((n, 2), 1e3, device=dev)   # time since the ball last touched each paddle
    self.valid_hit = torch.zeros(n, dtype=torch.bool, device=dev)
    self.fault = torch.zeros(n, dtype=torch.bool, device=dev)
    self._stamp = -1

  def reset(self, env_ids: torch.Tensor, expected: torch.Tensor) -> None:
    self.expected[env_ids] = expected
    self.hits[env_ids] = 0
    self.away_s[env_ids] = 1e3
    self.valid_hit[env_ids] = False
    self.fault[env_ids] = False

  def update(self, env: ManagerBasedRlEnv) -> None:
    """Runs once per policy step (whichever term asks first), right after the physics."""
    if self._stamp == env.common_step_counter:
      return
    self._stamp = env.common_step_counter
    touching = torch.stack([_touching(env, s) for s in SENSORS], dim=-1)   # [n, 2]
    new_hit = touching & (self.away_s > HIT_GAP_S)
    self.away_s = torch.where(touching, torch.zeros_like(self.away_s), self.away_s + env.step_dt)
    any_hit = new_hit.any(dim=-1)
    hitter = new_hit[:, 1].long()                       # 1 if the right paddle hit, else 0
    in_turn = any_hit & (hitter == self.expected) & ~new_hit.all(dim=-1)
    self.valid_hit = in_turn
    self.fault = any_hit & ~in_turn
    self.expected = torch.where(in_turn, 1 - self.expected, self.expected)
    self.hits += in_turn.long()


def _touching(env: ManagerBasedRlEnv, sensor: str) -> torch.Tensor:
  """True if the ball touched that paddle at any physics substep of the last policy step. [num_envs]."""
  force = env.scene[sensor].data.force_history       # [n, 1, history, 3]
  return force.norm(dim=-1).amax(dim=(1, 2)) > 0.0


def _rally_state(env: ManagerBasedRlEnv) -> _Rally:
  rally = getattr(env, "_pingpong_rally", None)
  if rally is None:
    rally = _Rally(env)
    env._pingpong_rally = rally
  return rally


def rally(env: ManagerBasedRlEnv) -> _Rally:
  """The rally state, updated for this step."""
  r = _rally_state(env)
  r.update(env)
  return r


# ---- serve ----------------------------------------------------------------------------------------------


def serve(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Put the ball at the target height over the middle, moving towards a random paddle so it lands on it."""
  env_ids = resolve_env_ids(env, env_ids)
  n, dev = len(env_ids), env.device

  def uniform(half_width: float) -> torch.Tensor:
    return (torch.rand(n, device=dev) * 2 - 1) * half_width

  side = torch.randint(0, 2, (n,), device=dev)                      # 1 = towards the right paddle
  z0 = TARGET_APEX + uniform(SERVE_Z_JITTER)
  fall_time = torch.sqrt(2 * (z0 - HIT_Z) / GRAVITY)                 # from the top of the flight to paddle height
  vx = (2 * side - 1).float() * (GAP / 2) / fall_time * (1 + uniform(SERVE_SPEED_JITTER))

  pos = env.scene.env_origins[env_ids] + torch.stack([uniform(SERVE_XY_JITTER), uniform(SERVE_XY_JITTER), z0], -1)
  quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).expand(n, 4)
  vel = torch.zeros(n, 6, device=dev)
  vel[:, 0] = vx
  ball = env.scene["ball"]
  ball.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
  ball.write_root_link_velocity_to_sim(vel, env_ids=env_ids)
  _rally_state(env).reset(env_ids, side)


# ---- observations ---------------------------------------------------------------------------------------


def _ball_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball position relative to this env's origin, [num_envs, 3]."""
  return env.scene["ball"].data.root_link_pos_w - env.scene.env_origins


def _ball_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["ball"].data.root_link_lin_vel_w


def ball_state(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[ball x y z, ball vx vy vz]. Shape [num_envs, 6]."""
  return torch.cat([_ball_pos(env), _ball_vel(env)], dim=-1)


def paddle_state(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[slide, lift, tilt, and their speeds] for the left then the right paddle. Shape [num_envs, 12]."""
  parts = []
  for name in PADDLES:
    data = env.scene[name].data
    parts += [data.joint_pos, data.joint_vel]
  return torch.cat(parts, dim=-1)


def whose_turn(env: ManagerBasedRlEnv) -> torch.Tensor:
  """-1 if the left paddle must hit next, +1 if the right one. Shape [num_envs, 1]."""
  return (2 * rally(env).expected - 1).float()[:, None]


# ---- rewards --------------------------------------------------------------------------------------------


def apex_estimate(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Height the ball will peak at if nothing touches it: z + vz^2 / (2 g) (for a ball still going up)."""
  z, vz = _ball_pos(env)[:, 2], _ball_vel(env)[:, 2]
  return z + torch.square(torch.clamp(vz, min=0.0)) / (2 * GRAVITY)


def landing_x(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Where (x) the ball will come back down to paddle height if nothing touches it."""
  pos, vel = _ball_pos(env), _ball_vel(env)
  drop = torch.clamp(pos[:, 2] - HIT_Z, min=0.0)
  t = (vel[:, 2] + torch.sqrt(torch.square(vel[:, 2]) + 2 * GRAVITY * drop)) / GRAVITY
  return pos[:, 0] + vel[:, 0] * t


def flight_quality(env: ManagerBasedRlEnv) -> torch.Tensor:
  """In (0, 1]: 1 when the ball is on its way to peak at TARGET_APEX and come down on the paddle whose turn it is.

  It pays every step (not just once per hit), so there is something to learn from before the first good hit.
  Peak and landing point stay the same for the whole flight, so the policy is paid for the hit that set them.
  """
  target_x = torch.tensor(HOME_X, device=env.device)[rally(env).expected]
  apex_term = torch.exp(-torch.square((apex_estimate(env) - TARGET_APEX) / APEX_SIGMA))
  land_term = torch.exp(-torch.square((landing_x(env) - target_x) / LAND_SIGMA))
  return apex_term * land_term


def good_hit(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 per hit made in turn. (Divided by dt because mjlab multiplies every reward by dt.)"""
  return rally(env).valid_hit.float() / env.step_dt


def fault(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 per double hit or dropped ball, i.e. per episode that ends for any reason except running out of time."""
  return env.termination_manager.terminated.float() / env.step_dt


# ---- terminations ---------------------------------------------------------------------------------------


def double_hit(env: ManagerBasedRlEnv) -> torch.Tensor:
  """True when a paddle hits the ball out of turn (twice in a row)."""
  return rally(env).fault


def ball_dropped(env: ManagerBasedRlEnv) -> torch.Tensor:
  """True once the ball is too low to be saved, or has flown off to the side."""
  pos = _ball_pos(env)
  too_low = pos[:, 2] < LOST_Z
  too_far = (pos[:, 0].abs() > GAP / 2 + X_RANGE + PADDLE_RADIUS + BALL_RADIUS) | (pos[:, 1].abs() > PADDLE_RADIUS)
  return too_low | too_far


# ---- environment ----------------------------------------------------------------------------------------

DECIMATION = 5


def rally_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "ball": ObservationTermCfg(func=ball_state),
    "paddles": ObservationTermCfg(func=paddle_state),
    "whose_turn": ObservationTermCfg(func=whose_turn),
    "last_action": ObservationTermCfg(func=last_action),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
    "critic": ObservationGroupCfg({**actor_terms}),
  }

  scale = {"slide": X_RANGE, "lift": Z_RANGE, "tilt": PITCH_RANGE}
  actions: dict[str, ActionTermCfg] = {
    name: JointPositionActionCfg(entity_name=name, actuator_names=JOINTS, scale=scale, use_default_offset=True)
    for name in PADDLES
  }

  events = {
    # puts both paddles at home (fixed-base entities are only placed in the world by a reset) and the ball at rest
    "reset_scene": EventTermCfg(func=reset_scene_to_default, mode="reset"),
    "serve": EventTermCfg(func=serve, mode="reset"),
  }

  rewards = {
    "flight": RewardTermCfg(func=flight_quality, weight=1.0),
    "hit": RewardTermCfg(func=good_hit, weight=1.0),
    "fault": RewardTermCfg(func=fault, weight=-5.0),
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.01),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "double_hit": TerminationTermCfg(func=double_hit),
    "ball_dropped": TerminationTermCfg(func=ball_dropped),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={
        "ball": _ball_cfg(),
        "left_paddle": _paddle_cfg(HOME_X[0], ".2 .6 .3 1"),
        "right_paddle": _paddle_cfg(HOME_X[1], ".2 .3 .7 1"),
      },
      sensors=tuple(_ball_contact_sensor(s, p, DECIMATION) for s, p in zip(SENSORS, PADDLES, strict=True)),
      num_envs=1,
      env_spacing=3.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.WORLD,
      distance=3.5,
      elevation=-10.0,
      azimuth=90.0,
      lookat=(0.0, 0.0, 0.85),
    ),
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.002)),
    decimation=DECIMATION,
    episode_length_s=1e10 if play else EPISODE_S,
  )


def rally_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(128, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(128, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.995,                  # looks about 2 s ahead: longer than one flight from paddle to paddle
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="pingpong_rally",
    logger="tensorboard",           # the default, wandb, needs an account and a login
    save_interval=100,
    num_steps_per_env=48,
    max_iterations=1500,
  )


register_mjlab_task(
  task_id="PingPong-Rally",
  env_cfg=rally_env_cfg(),
  play_env_cfg=rally_env_cfg(play=True),
  rl_cfg=rally_ppo_runner_cfg(),
)

