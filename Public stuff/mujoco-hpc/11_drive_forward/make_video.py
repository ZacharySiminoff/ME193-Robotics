"""Record a video of a trained Drive-Forward policy, with the numbers drawn on it. Made for the cluster.

    python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 \
      --with 11_drive_forward/drive_forward_task.py --with 09_onshape_robot/onshape \
      --with runs/<your run>/logs/rsl_rl/drive_forward/<date folder>/model_499.pt \
      11_drive_forward/make_video.py model_499.pt

Writes drive_forward.mp4. Each frame shows the time, how far the robot has driven forward, how far it has drifted
sideways, its heading, its speed and both wheel speeds. The same numbers are printed once a second.
runIt.py brings the .mp4 home to ./runs/ and opens it.
"""
import math
import sys
from dataclasses import asdict
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import drive_forward_task as task  # registers "Drive-Forward"
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

TASK = "Drive-Forward"
OUT = "drive_forward.mp4"

if len(sys.argv) not in (2, 3):
  sys.exit("usage: make_video.py model_499.pt [seconds, default 30]")
checkpoint = Path(sys.argv[1])
SECONDS = float(sys.argv[2]) if len(sys.argv) == 3 else 30.0

configure_torch_backends()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = 1
env_cfg.viewer.width, env_cfg.viewer.height = 640, 480
agent_cfg = load_rl_cfg(TASK)
steps_per_s = round(1 / (env_cfg.sim.mujoco.timestep * env_cfg.decimation))
steps = round(SECONDS * steps_per_s)

env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)
raw = env.unwrapped
robot = raw.scene["robot"]

try:
  FONT = ImageFont.load_default(size=18)
except TypeError:                       # Pillow older than 10.1 has only a small fixed-size font
  FONT = ImageFont.load_default()


def readings() -> dict[str, float]:
  wheel = robot.data.joint_vel[0].tolist()
  return {
    "forward": task.forward_distance(raw)[0].item(),
    "sideways": task.off_line(raw)[0].item(),
    "heading": math.degrees(task._yaw(raw)[0].item()),
    "speed": task.forward_velocity(raw)[0].item(),
    "left": wheel[0],
    "right": wheel[1],
  }


def overlay(frame: np.ndarray, t: float, r: dict[str, float], note: str = "") -> np.ndarray:
  lines = [
    f"Time: {t:.1f} s",
    f"Forward: {r['forward']:+.3f} m",
    f"Sideways: {r['sideways']:+.3f} m",
    f"Heading: {r['heading']:+.1f} deg",
    f"Speed: {r['speed']:+.3f} m/s  (target {task.TARGET_SPEED:.3f})",
    f"Wheels: L {r['left']:+.1f}, R {r['right']:+.1f} rad/s  (commanded max {task.MAX_WHEEL_SPEED:.0f})",
  ]
  if note:
    lines.append(note)
  img = Image.fromarray(frame)
  draw = ImageDraw.Draw(img, "RGBA")
  line_h = FONT.getbbox("Ag")[3] + 6
  draw.rectangle([8, 8, 8 + 470, 8 + 10 + line_h * len(lines)], fill=(0, 0, 0, 150))
  for i, text in enumerate(lines):
    draw.text((16, 13 + i * line_h), text, font=FONT, fill=(255, 255, 255, 255))
  return np.asarray(img)


print("  second   forward (m)   sideways (m)   heading (deg)   speed (m/s)")
frames = []
note, note_until = "", -1.0
obs = env.get_observations()
frames.append(overlay(raw.render(), 0.0, readings()))
with torch.no_grad():
  for step in range(steps):
    obs, _, dones, _ = env.step(policy(obs))
    t = (step + 1) / steps_per_s
    r = readings()
    if dones[0]:
      print(f"  {t:5.1f}   episode ended (tipped over or too far off the line); restarting")
      note, note_until = "episode ended: restarted", t + 1.0
    elif step % steps_per_s == steps_per_s - 1:   # once a second
      print(f"  {t:5.0f}   {r['forward']:11.2f}   {r['sideways']:12.2f}   {r['heading']:13.1f}   {r['speed']:11.2f}")
    frames.append(overlay(raw.render(), t, r, note if t < note_until else ""))
env.close()
imageio.mimwrite(OUT, frames, fps=steps_per_s)
print(f"wrote {OUT}: {len(frames)} frames, {SECONDS:.0f} s")
