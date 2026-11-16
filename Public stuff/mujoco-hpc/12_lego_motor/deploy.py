"""Run a trained Drive-Forward policy on the real robot, over Bluetooth, from your Mac.

    python deploy.py                                   # newest policy.npz in ../runs/
    python deploy.py ../runs/train_and_video-XXXXXXXX-XXXXXX
    python deploy.py --check                           # just drive forward slowly for 1 s: are the directions right?
    python deploy.py --fake                            # no hardware

It connects ONCE, then you do as many runs as you like, from the terminal:
    Enter        start a run (the heading is zeroed: wherever the robot points now is "straight")
    Enter        stop the run early (runs also stop by themselves after --seconds)
    q + Enter    quit
Between runs the motors are stopped, so you can pick the robot up and put it back at the start.
Each run is saved to deploy_logs/run-<time>.csv (what the policy saw and what it commanded).

What it feeds the policy is built exactly like the simulation's actor observations (drive_forward_task.py):
  heading      sin, cos of the gyro yaw (counter-clockwise positive, radians)
  wheel_speed  change in each wheel's encoder (whole degrees) over one control step, in rad/s, forward positive
  last_action  the policy's previous output
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

import config
from lego_robot import FakeRobot, LegoRobot

HERE = Path(__file__).resolve().parent


# ---- the policy, in numpy -----------------------------------------------------------------------------------

class Policy:
  def __init__(self, folder: Path):
    self.arrays = dict(np.load(folder / "policy.npz"))
    self.meta = json.loads((folder / "policy_meta.json").read_text())
    self.eps = float(self.meta.get("obs_eps", 0.01))
    self.n = int(self.meta["num_layers"])

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    x = (obs - self.arrays["obs_mean"]) / (self.arrays["obs_std"] + self.eps)
    for i in range(self.n):
      x = self.arrays[f"w{i}"] @ x + self.arrays[f"b{i}"]
      if i < self.n - 1:
        x = np.where(x > 0, x, np.expm1(x))         # ELU
    return x


def find_policy(arg: str | None) -> Path:
  if arg:
    p = Path(arg)
    folder = p.parent if p.suffix == ".npz" else p
    if not (folder / "policy.npz").exists():
      found = sorted(folder.glob("**/policy.npz"), key=lambda q: q.stat().st_mtime)
      if not found:
        sys.exit(f"no policy.npz in {folder}")
      folder = found[-1].parent
    return folder
  found = sorted((HERE.parent / "runs").glob("*/policy.npz"), key=lambda q: q.stat().st_mtime)
  if not found:
    sys.exit("no policy.npz found in ../runs/. Train with 11_drive_forward/train_and_video.py first.")
  return found[-1].parent


# ---- keyboard: lines typed in the terminal, read in the background ------------------------------------------------

def start_keyboard() -> "queue.Queue[str]":
  q: "queue.Queue[str]" = queue.Queue()

  def reader():
    for line in sys.stdin:
      q.put(line.strip().lower())
    q.put("q")                    # the terminal closed: quit
  threading.Thread(target=reader, daemon=True).start()
  return q


# ---- the robot as the policy sees it ----------------------------------------------------------------------------------

class RobotIO:
  """Turns policy outputs into motor percents and motor readings into policy inputs, using motor_model.json."""

  def __init__(self, robot, hw: dict, max_wheel_speed: float, control_dt: float):
    self.robot, self.hw, self.max_speed, self.dt = robot, hw, max_wheel_speed, control_dt
    self.fwd = 1.0 if config.TANK_FORWARD_IS_ROBOT_FORWARD else -1.0
    self.tank_left_index = int(hw.get("tank_left_drives_motor_index", 0))
    # encoder index of the robot's left and right wheel
    self.enc = [self.tank_left_index, 1 - self.tank_left_index]
    if config.SWAP_LEFT_RIGHT:
      self.enc.reverse()
    self.enc_sign = [float(s) for s in hw.get("encoder_sign_tank_forward", [1.0, -1.0])]
    self.g = float(hw["rad_s_per_percent"])
    self.dead = float(hw.get("deadband_percent", 0.0))
    self.yaw_unit = float(hw["imu_yaw_rad_per_unit"])
    self.prev_pos = None

  def begin(self) -> None:
    self.robot.reset_heading()
    time.sleep(0.15)
    self.prev_pos = None

  def observe(self, last_action: np.ndarray) -> tuple[np.ndarray, dict]:
    r = self.robot.latest()
    yaw = r.yaw * self.yaw_unit
    if abs(yaw) > 1.05 * math.pi and not getattr(self, "_warned_yaw", False):
      self._warned_yaw = True
      print(f"  WARNING: yaw {yaw:+.1f} rad is more than half a turn from a raw {r.yaw:+.0f}: the yaw unit in "
            "motor_model.json (imu_yaw_rad_per_unit) is probably wrong. The raw yaw should stay within +-1800.")
    pos =[r.position[self.enc[0]], r.position[self.enc[1]]]
    if self.prev_pos is None:
      speed = [0.0, 0.0]
    else:
      speed = [self.fwd * self.enc_sign[self.enc[k]] * math.radians(round(pos[k]) - round(self.prev_pos[k])) / self.dt
               for k in range(2)]
    self.prev_pos = pos
    obs = np.array([math.sin(yaw), math.cos(yaw), speed[0], speed[1], last_action[0], last_action[1]], dtype=np.float32)
    return obs, {"yaw": yaw, "speed": speed, "reading_age": time.perf_counter() - r.t}

  def to_percent(self, rad_s: float) -> int:
    if abs(rad_s) < 0.5 * self.g:
      return 0
    table_p, table_v = self.hw.get("speed_table_percent"), self.hw.get("speed_table_rad_s")
    if table_p and table_v and np.all(np.diff(table_v) > 0):     # measured steady speed for each %: look it up
      return int(round(float(np.interp(rad_s, table_v, table_p))))
    return int(round(rad_s / self.g + math.copysign(self.dead, rad_s)))

  def command(self, action: np.ndarray) -> tuple[int, int]:
    """Policy output (robot left, robot right) -> Double Motor tank percents. Returns what was sent."""
    rad = np.clip(action, -1.0, 1.0) * self.max_speed
    pct = [self.fwd * self.to_percent(float(rad[0])), self.fwd * self.to_percent(float(rad[1]))]
    if self.enc[0] != self.tank_left_index:     # the robot's left wheel is on the motor's right output
      pct.reverse()
    self.robot.set_speed(pct[0], pct[1])
    return int(pct[0]), int(pct[1])


def load_hardware_model(meta: dict) -> dict:
  local = HERE / "motor_model.json"
  hw = dict(meta.get("motor_model", {}))
  if local.exists():
    hw.update(json.loads(local.read_text()))
  else:
    print("WARNING: no motor_model.json here (run characterize.py). Using the values the policy was trained with "
          "and guessing the encoder directions.")
  if "encoder_sign_tank_forward" not in hw:
    print("WARNING: encoder directions unknown; assuming the right encoder counts backwards. Check with --check.")
  return hw


# ---- runs ---------------------------------------------------------------------------------------------------------------

def run_episode(io: RobotIO, policy: Policy | None, seconds: float, keys: "queue.Queue[str]") -> None:
  log_dir = HERE / "deploy_logs"
  log_dir.mkdir(exist_ok=True)
  log_path = log_dir / f"run-{datetime.datetime.now():%Y%m%d-%H%M%S}.csv"
  rows = ["t,yaw_rad,speed_left,speed_right,action_left,action_right,tank_left,tank_right,reading_age_s"]
  io.begin()
  last_action = np.zeros(2, dtype=np.float32)
  steps = int(seconds / io.dt)
  t0 = time.perf_counter()
  t_next = t0
  print("  time    yaw (deg)   wheel L/R (rad/s)    command L/R (%)")
  try:
    for k in range(steps):
      if not keys.empty():
        if keys.get() in ("", "q", "s", "stop"):
          print("  stopped")
          break
      obs, info = io.observe(last_action)
      if policy is None:                       # --check: both wheels gently forward
        action = np.array([0.3, 0.3], dtype=np.float32)
      else:
        action = policy(obs).astype(np.float32)
      sent = io.command(action)
      last_action = action
      t = time.perf_counter() - t0
      rows.append(f"{t:.3f},{info['yaw']:.4f},{info['speed'][0]:.3f},{info['speed'][1]:.3f},"
                  f"{action[0]:.4f},{action[1]:.4f},{sent[0]},{sent[1]},{info['reading_age']:.4f}")
      if k % max(1, int(round(1 / io.dt))) == 0:
        print(f"  {t:5.1f}   {math.degrees(info['yaw']):+8.1f}    {info['speed'][0]:+6.2f} {info['speed'][1]:+6.2f}"
              f"        {sent[0]:+4d} {sent[1]:+4d}")
      t_next += io.dt
      late = time.perf_counter() - t_next
      if late < 0:
        time.sleep(-late)
      elif late > io.dt:                      # fell behind (Bluetooth hiccup): don't try to catch up
        t_next = time.perf_counter()
  finally:
    io.robot.stop()
    log_path.write_text("\n".join(rows) + "\n")
    print(f"  motors stopped. Log: {log_path.relative_to(HERE)}")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("policy", nargs="?", help="run folder (or policy.npz); default: the newest in ../runs/")
  ap.add_argument("--seconds", type=float, default=30.0, help="length of each run (default 30)")
  ap.add_argument("--check", action="store_true", help="drive forward slowly for 1 s to check directions, then quit")
  ap.add_argument("--fake", action="store_true", help="no hardware")
  a = ap.parse_args()

  if a.check:
    policy, meta = None, {"control_dt": config.CONTROL_DT, "max_wheel_speed_rad_s": 6.0}
    folder = None
  else:
    folder = find_policy(a.policy)
    policy = Policy(folder)
    meta = policy.meta
    print(f"policy: {folder}  ({meta['checkpoint']}, trained with motor model from {meta['motor_model_source']})")
    if abs(meta["control_dt"] - config.CONTROL_DT) > 1e-6:
      print(f"note: the policy runs every {meta['control_dt'] * 1000:.0f} ms (config.py says "
            f"{config.CONTROL_DT * 1000:.0f} ms); using the policy's")
  hw = load_hardware_model(meta)
  if "rad_s_per_percent" not in hw:
    hw["rad_s_per_percent"] = 0.14
  hw.setdefault("imu_yaw_rad_per_unit", math.pi / 1800)   # tenths of a degree, counter-clockwise +

  robot = FakeRobot() if a.fake else LegoRobot()
  robot.connect()
  io = RobotIO(robot, hw, float(meta["max_wheel_speed_rad_s"]), float(meta["control_dt"]))
  keys = start_keyboard()
  try:
    if a.check:
      print("\nCHECK: both wheels forward at 30 % of the policy's top speed for 1 s. Press Enter to go.")
      keys.get()
      run_episode(io, None, 1.0, keys)
      print("Did the robot go FORWARD (the way it should drive)? If it went backwards, set "
            "TANK_FORWARD_IS_ROBOT_FORWARD = False in config.py.\nBoth wheel speeds above should be POSITIVE; "
            "if one wheel turned the wrong way, set SWAP_LEFT_RIGHT or check the wiring.")
      return
    while True:
      print(f"\nPut the robot at the start, pointing the way it should go. Enter = start a {a.seconds:.0f} s run, "
            "q + Enter = quit.")
      line = keys.get()
      if line in ("q", "quit", "exit"):
        break
      while not keys.empty():
        keys.get()
      print("running (Enter = stop)")
      run_episode(io, policy, a.seconds, keys)
  except KeyboardInterrupt:
    print("\ninterrupted")
  finally:
    robot.stop()
    robot.disconnect()
    print("disconnected")


if __name__ == "__main__":
  main()
