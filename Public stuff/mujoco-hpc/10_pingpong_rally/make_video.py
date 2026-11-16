"""Record a video of a trained PingPong-Rally policy, and print every hit. Made for the cluster, which has no screen.

    python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 \
      --with 10_pingpong_rally/rally_task.py \
      --with runs/<your run>/logs/rsl_rl/pingpong_rally/<date folder>/model_1499.pt \
      10_pingpong_rally/make_video.py model_1499.pt

runIt.py brings the .mp4 home to ./runs/ and opens it. If the ball is dropped (or hit twice in a row) the rally
restarts with a new serve, and the video keeps going.
"""
import sys
from dataclasses import asdict
from pathlib import Path

import torch

import rally_task as task  # registers "PingPong-Rally"
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

TASK = "PingPong-Rally"

if len(sys.argv) not in (2, 3):
  sys.exit("usage: make_video.py model_1499.pt [seconds, default 15]")
checkpoint = Path(sys.argv[1])
SECONDS = float(sys.argv[2]) if len(sys.argv) == 3 else 15.0

configure_torch_backends()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = 1
env_cfg.viewer.width, env_cfg.viewer.height = 640, 480
agent_cfg = load_rl_cfg(TASK)
steps = round(SECONDS / (env_cfg.sim.mujoco.timestep * env_cfg.decimation))

env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
env = VideoRecorder(env, video_folder=".", step_trigger=lambda step: step == 0, video_length=steps, disable_logger=True)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

raw = env.unwrapped
names = ("left", "right")
rallies, longest, hits_now = 0, 0, 0
print(f"  time (s)   event")
obs = env.get_observations()
with torch.no_grad():
  for step in range(steps):
    t = (step + 1) * raw.step_dt
    obs, _, dones, _ = env.step(policy(obs))
    rally = task._rally_state(raw)
    if rally.valid_hit[0]:
      hitter = names[1 - int(rally.expected[0])]          # it has just become the other paddle's turn
      hits_now = int(rally.hits[0])
      print(f"  {t:7.2f}   hit {hits_now:3d} by the {hitter:5s} paddle")
    if dones[0]:
      reason = "double hit" if rally.fault[0] else ("time up" if raw.reset_time_outs[0] else "ball dropped")
      print(f"  {t:7.2f}   rally over after {hits_now} hits ({reason}); new serve")
      rallies, longest, hits_now = rallies + 1, max(longest, hits_now), 0
env.close()
longest = max(longest, hits_now)
print(f"longest rally: {longest} hits; rallies ended early: {rallies}")
print("wrote the video (an .mp4 next to this script)")
