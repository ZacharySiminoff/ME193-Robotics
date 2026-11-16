"""Characterize the LEGO Education Double Motor, and save a model the RL simulation uses.

    python characterize.py              # the motor tests, Bluetooth only (what the simulation needs)
    python characterize.py --camera     # also film the arm, to cross-check the timing (optional)
    python characterize.py --gyro       # also redo the hand-turn gyro test (otherwise the earlier result is kept)
    python characterize.py --fake       # no hardware: a simulated motor, to try the script
    python characterize.py --reanalyze results/20261116-144054    # redo the analysis of a saved run, no hardware
    python characterize.py --gyro-only  # only redo the hand-turn gyro test

Before running: both motor outputs must be free to spin, up to full speed, both ways, for about 1.5 minutes. Best with
the robot's own wheels on and the robot propped up so they don't touch anything (the motor then turns the same
wheels it will drive). With --camera: an arm on the ARM_MOTOR side and the camera square on to it.

Tests:
  1. update timing   how often encoder updates arrive, and how regular they are
  2. round trip      Bluetooth: send a command, wait for the motor's reply (x40)
  3. direction       which way each encoder counts when the motor drives "forward"
  4. speed sweep     steady speed at each speed % in both directions: speed per %, dead zone, top speed
  5. step responses  sudden speed changes
  6. random commands a new random speed every CONTROL_DT, like the policy will send
  7. camera moves    slow back-and-forth, so the camera sees the arm without blur (only with the camera)
  8. gyro (yaw)      you turn the motor a quarter turn by hand: yaw units and sign (optional)

The motor model (delay and time constants) is fitted to tests 5 and 6 together. The encoder positions are exact; their
arrival times on the computer are not (Bluetooth delivers them late by different amounts), so each reading's time is
rebuilt from the motor's own steady update clock first. The camera only cross-checks the timing.
Results go to results/<time>/ (raw data, summary.txt, characterization.png) and to motor_model.json, which
../11_drive_forward/drive_forward_task.py reads.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import config
from lego_robot import FakeRobot, LegoRobot

HERE = Path(__file__).resolve().parent
CAMERA_MAX_SPEED = 6.0       # rad/s; faster than this the arm blurs in a ~30 fps camera, so those frames are not used


# ======================================================================================================
# running the tests
# ======================================================================================================

class Session:
  def __init__(self, robot=None):
    self.robot = robot
    self.commands: list[tuple[float, int, int, str]] = []   # (send time, left %, right %, test)
    self.pings: list[float] = []
    self.marks: dict[str, tuple[float, float]] = {}          # test name -> (start, end) time

  def command(self, left: float, right: float, label: str) -> None:
    t = self.robot.set_speed(left, right)
    self.commands.append((t, int(round(left)), int(round(right)), label))

  def test(self, name: str):
    session = self

    class _Mark:
      def __enter__(self):
        print(f"\n--- {name}")
        self.t0 = time.perf_counter()

      def __exit__(self, *exc):
        session.marks[name] = (self.t0, time.perf_counter())
    return _Mark()


def gyro_test(robot) -> float | None:
  """You turn the motor a quarter turn; returns the change in raw yaw units (None if skipped)."""
  for attempt in range(3):
    print("    Lay the Double Motor flat, the way it sits in the robot, and keep it still. Press Enter (s + Enter = skip).")
    if input().strip().lower() == "s":
      return None
    y0 = robot.latest().yaw
    print("    Now turn it a QUARTER TURN COUNTER-CLOCKWISE (to the left, seen from above), keeping it flat on the")
    print("    table. Use a table edge or a mark to make it a real quarter turn. Put it down and press Enter.")
    input()
    time.sleep(0.3)
    change = robot.latest().yaw - y0
    print(f"    yaw changed by {change:+.0f} raw units")
    if abs(change) >= 100:          # tenths of a degree: at least a 10 degree turn, so the sign is clear
      return change
    print("    That is less than 10 degrees. Was it flat? If the motor sits on its side in the robot, set IMU_YAW_FACE")
    print("    in config.py. Let's try again.")
  return change


def run_tests(s: Session, skip_imu: bool, with_camera: bool) -> dict:
  robot = s.robot
  robot.recording = True
  extra = {}

  with s.test("update timing"):
    time.sleep(2.0)

  with s.test("round trip"):
    for _ in range(40):
      s.pings.append(robot.ping())
      time.sleep(0.02)
    print(f"    round trip median {np.median(s.pings) * 1000:.1f} ms")

  with s.test("direction"):
    s.command(30, 30, "direction")
    time.sleep(0.8)
    s.command(0, 0, "direction")
    time.sleep(0.6)
    s.command(30, 0, "direction-left-only")
    time.sleep(0.6)
    s.command(0, 0, "direction")
    time.sleep(0.6)

  with s.test("speed sweep"):
    for sign in (1, -1):
      for p in config.TEST_PERCENTS:
        s.command(sign * p, sign * p, "sweep")
        time.sleep(1.5)
      s.command(0, 0, "sweep-stop")
      time.sleep(0.8)
    print("    done")

  with s.test("step responses"):
    S = config.STEP_PERCENT
    for _ in range(config.STEP_REPEATS):
      for target, hold in ((S, 1.0), (0, 0.9), (-S, 1.0), (S, 1.0), (0, 0.9)):
        s.command(target, target, "step")
        time.sleep(hold)
    for target, hold in ((100, 1.0), (0, 1.0)):
      s.command(target, target, "step")
      time.sleep(hold)
    print("    done")

  with s.test("random commands"):
    rng = random.Random(1)
    t_next = time.perf_counter()
    value, hold_left = 0, 0
    for _ in range(int(config.RANDOM_SECONDS / config.CONTROL_DT)):
      if hold_left == 0:
        value, hold_left = rng.randint(-int(1.2 * config.STEP_PERCENT), int(1.2 * config.STEP_PERCENT)), rng.randint(1, 6)
      hold_left -= 1
      s.command(value, value, "random")
      t_next += config.CONTROL_DT
      time.sleep(max(0.0, t_next - time.perf_counter()))
    s.command(0, 0, "random")
    time.sleep(0.8)
    print("    done")

  if with_camera:
    with s.test("camera moves"):
      for k in range(10):
        s.command(20 if k % 2 == 0 else -20, 20 if k % 2 == 0 else -20, "camera")
        time.sleep(0.7)
      s.command(0, 0, "camera")
      time.sleep(0.6)
      print("    done")

  robot.stop()

  if not skip_imu:
    with s.test("gyro"):
      change = gyro_test(robot)
      if change is not None:
        extra["imu_raw_change_ccw_quarter"] = float(change)
  robot.recording = False
  return extra


# ======================================================================================================
# analysis helpers
# ======================================================================================================

def as_arrays(log) -> dict[str, np.ndarray]:
  return {
    "t": np.array([r.t for r in log]),
    "pos": np.array([r.position for r in log], dtype=float),
    "speed": np.array([r.speed for r in log], dtype=float),
    "power": np.array([r.power for r in log], dtype=float),
    "yaw": np.array([r.yaw for r in log], dtype=float),
    "gyro_z": np.array([r.gyro_z for r in log], dtype=float),
  }


def in_window(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
  return (t >= t0) & (t <= t1)


def slope(t: np.ndarray, y: np.ndarray) -> float:
  return float(np.polyfit(t - t[0], y, 1)[0]) if len(t) >= 3 else float("nan")


def command_windows(commands, label_prefix: str):
  """(send time, left %, right %, time of the next command) for commands whose label starts with label_prefix."""
  out = []
  for i, (t, left, right, label) in enumerate(commands):
    if label.startswith(label_prefix):
      t_next = commands[i + 1][0] if i + 1 < len(commands) else t + 1.5
      out.append((t, left, right, t_next))
  return out


def steady_clock_times(t: np.ndarray, half: int = 40) -> np.ndarray:
  """When each reading would have arrived with no Bluetooth hold-up.

  The motor sends a reading every update interval on its own clock; Bluetooth delivers some of them late (often by
  a whole interval) and then two at once. Assuming one reading per interval, the earliest-arriving readings nearby
  show where the steady clock is: t_k = min over j of (t_j - T * (j - k)), with T the local median spacing.
  """
  n = len(t)
  out = np.empty(n)
  for k in range(n):
    lo, hi = max(0, k - half), min(n, k + half + 1)
    tt = t[lo:hi]
    T = np.median(np.diff(tt))
    out[k] = np.min(tt - T * (np.arange(lo, hi) - k))
  return out


class MotorModel:
  """First-order speed response with a dead time, evaluated in closed form between command changes."""

  def __init__(self, cmd_t: np.ndarray, cmd_rad_s: np.ndarray):
    self.cmd_t, self.target = np.asarray(cmd_t), np.asarray(cmd_rad_s)

  def positions(self, t_eval: np.ndarray, delay: float, tau_up: float, tau_down: float) -> np.ndarray:
    ts, tgt = self.cmd_t + delay, self.target
    n = len(ts)
    v0, x0, taus = np.zeros(n), np.zeros(n), np.zeros(n)
    v = x = 0.0
    for k in range(n):
      if k > 0:
        dt, tau, T = ts[k] - ts[k - 1], taus[k - 1], tgt[k - 1]
        e = math.exp(-dt / tau)
        x, v = x + T * dt + (v - T) * tau * (1 - e), T + (v - T) * e
      v0[k], x0[k] = v, x
      taus[k] = tau_up if abs(tgt[k]) > abs(v) else tau_down
    idx = np.searchsorted(ts, t_eval, side="right") - 1
    before = idx < 0
    idx = np.clip(idx, 0, None)
    s = t_eval - ts[idx]
    T, tau = tgt[idx], taus[idx]
    out = x0[idx] + T * s + (v0[idx] - T) * tau * (1 - np.exp(-s / tau))
    return np.where(before, 0.0, out)


def windowed_error(model: MotorModel, blocks, delay, tau_up, tau_down, window=0.2) -> float:
  """RMS (degrees) of the error in how far each wheel turns over `window` seconds: timing, not slow drift."""
  total, count = 0.0, 0
  for (tb, angles) in blocks:          # angles: list of per-motor forward angles (rad)
    te = tb[tb < tb[-1] - window]
    p = model.positions(np.concatenate([te, te + window]), delay, tau_up, tau_down)
    dp = p[len(te):] - p[:len(te)]
    for ang in angles:
      dm = np.interp(te + window, tb, ang) - np.interp(te, tb, ang)
      total += float(np.sum((dm - dp) ** 2))
      count += len(te)
  return math.degrees(math.sqrt(total / max(count, 1)))


# ======================================================================================================
# analysis
# ======================================================================================================

def analyze(s: Session, data: dict, camera_samples, extra: dict, notify_s: float) -> tuple[dict, list[str]]:
  lines: list[str] = []

  def say(text: str) -> None:
    print(text)
    lines.append(text)

  t_arrive, pos = data["t"], data["pos"]
  m = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "card_serial": str(config.CARD_SERIAL),
       "control_dt": config.CONTROL_DT, "acceleration_setting": list(config.ACCELERATION),
       "encoder_resolution_deg": 1.0}

  # 1. update timing, and how late readings arrive compared with the motor's steady clock
  t = steady_clock_times(t_arrive)
  late = t_arrive - t
  t0, t1 = s.marks["update timing"]
  gaps = np.diff(t_arrive[in_window(t_arrive, t0, t1)])
  m["notify_interval_s"] = float(np.median(np.diff(t_arrive))) if len(t_arrive) > 2 else notify_s
  m["reading_late_p50_s"] = float(np.percentile(late, 50))
  m["reading_late_p90_s"] = float(np.percentile(late, 90))
  m["reading_late_max_s"] = float(late.max())
  say("UPDATE TIMING")
  say(f"  encoder updates every {m['notify_interval_s'] * 1000:.1f} ms (asked for {config.NOTIFY_MS}); "
      f"while idle 90 % of gaps within {np.percentile(gaps, 90) * 1000:.1f} ms" if len(gaps) else "  no idle data")
  say(f"  compared with the motor's steady clock, readings arrive late by: median {np.percentile(late, 50) * 1000:.1f} ms, "
      f"90 % within {np.percentile(late, 90) * 1000:.1f} ms, worst {late.max() * 1000:.0f} ms")

  # 2. round trip
  rtt = np.array(s.pings)
  m["ble_round_trip_s"] = float(np.median(rtt))
  m["ble_round_trip_min_s"] = float(rtt.min())
  one_way = float(rtt.min()) / 2           # the fastest crossing: what a reading on the steady clock took to arrive
  say("BLUETOOTH ROUND TRIP (send a command, wait for the reply)")
  say(f"  median {np.median(rtt) * 1000:.1f} ms, 90 % within {np.percentile(rtt, 90) * 1000:.1f} ms, "
      f"fastest {rtt.min() * 1000:.1f} ms  ->  one crossing takes at least {one_way * 1000:.1f} ms")

  # 3. direction
  wins = command_windows(s.commands, "direction")
  fwd = next(w for w in wins if w[1] == 30 and w[2] == 30)
  left_only = next(w for w in wins if w[1] == 30 and w[2] == 0)
  sel = in_window(t, fwd[0], fwd[3] + 0.3)
  change = pos[sel][-1] - pos[sel][0]
  enc_sign = [1.0 if c >= 0 else -1.0 for c in change]
  sel = in_window(t, left_only[0], left_only[3] + 0.3)
  lo_change = np.abs(pos[sel][-1] - pos[sel][0])
  m["encoder_sign_tank_forward"] = enc_sign
  m["tank_left_drives_motor_index"] = int(np.argmax(lo_change))
  say("DIRECTION")
  say(f"  driving forward (+30, +30): left encoder {change[0]:+.0f} deg, right encoder {change[1]:+.0f} deg "
      f"-> encoder signs {enc_sign}")
  say(f"  left-only command moved motor index {int(np.argmax(lo_change))} ({lo_change[0]:.0f} vs {lo_change[1]:.0f} deg)")
  fwd_angle = [np.radians(pos[:, i] * enc_sign[i]) for i in range(2)]

  # 4. speed sweep
  rows = []
  for (tc, left, _right, t_next) in command_windows(s.commands, "sweep"):
    if left == 0:
      continue
    w = in_window(t, tc + 0.8, t_next - 0.05)
    if w.sum() < 5:
      continue
    v = [slope(t[w], fwd_angle[i][w]) for i in range(2)]
    rows.append((left, *v, *[float(np.mean(data["speed"][w, i])) for i in range(2)]))
  sweep = np.array(rows)
  p = sweep[:, 0]
  A = np.stack([p, np.sign(p)], axis=1)
  gains, deadbands = [], []
  for i in range(2):
    (g_i, c), *_ = np.linalg.lstsq(A[np.abs(p) >= 15], sweep[np.abs(p) >= 15, 1 + i], rcond=None)
    gains.append(float(g_i))
    deadbands.append(float(max(0.0, -c / g_i)))
  g = float(np.mean(gains))
  m["rad_s_per_percent"] = g
  m["rad_s_per_percent_each"] = gains
  m["deadband_percent"] = float(np.mean(deadbands))
  m["max_rad_s"] = float(np.mean(np.abs(sweep[np.abs(p) == np.abs(p).max(), 1:3])))
  order = np.argsort(p)
  m["speed_table_percent"] = [0.0] + [float(x) for x in p[order]]
  m["speed_table_rad_s"] = [0.0] + [float(x) for x in sweep[order, 1:3].mean(axis=1)]
  o = np.argsort(m["speed_table_percent"])
  m["speed_table_percent"] = [m["speed_table_percent"][k] for k in o]
  m["speed_table_rad_s"] = [m["speed_table_rad_s"][k] for k in o]
  linear_err = float(np.sqrt(np.mean((sweep[:, 1:3].mean(axis=1) - A @ np.array([g, -g * m["deadband_percent"]])) ** 2)))
  say("SPEED SWEEP (steady speed, rad/s, forward positive)")
  say("    cmd %    left    right")
  for row in rows:
    say(f"  {row[0]:+6.0f}  {row[1]:+7.2f}  {row[2]:+7.2f}")
  say(f"  {g:.4f} rad/s per % (left {gains[0]:.4f}, right {gains[1]:.4f}), dead zone {m['deadband_percent']:.1f} %, "
      f"top speed {m['max_rad_s']:.2f} rad/s; a straight line is off by {linear_err:.2f} rad/s rms "
      f"(deploy.py uses the measured table instead)")

  def to_rad_s(percent):
    return np.interp(percent, m["speed_table_percent"], m["speed_table_rad_s"])

  # 5 + 6. one motor model fitted to the step responses and the random commands together
  blocks = []
  for label in ("step", "random"):
    wins = command_windows(s.commands, label)
    if wins:
      w = in_window(t, wins[0][0] - 0.2, wins[-1][3] + 0.3)
      blocks.append((label, wins, t[w], [a[w] for a in fwd_angle]))
  all_wins = [w for b in blocks for w in b[1]]
  model = MotorModel(np.array([w[0] for w in all_wins]), to_rad_s(np.array([w[1] for w in all_wins], dtype=float)))
  fit_blocks = [(tb, angles) for (_, _, tb, angles) in blocks]
  taus_up = (0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.065, 0.08, 0.1)
  taus_down = (0.003, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08)
  delays = np.arange(0.0, 0.1001, 0.005)
  curve = []
  best = None
  for L in delays:
    best_L = None
    for tu in taus_up:
      for td in taus_down:
        e = windowed_error(model, fit_blocks, L, tu, td)
        if best_L is None or e < best_L[0]:
          best_L = (e, L, tu, td)
    curve.append(best_L)
    if best is None or best_L[0] < best[0]:
      best = best_L
  # A change can't show in a reading sooner than one trip there (the command) plus one trip back (the reading), so
  # delays shorter than the fastest round trip only fit by trading delay for a slower time constant: not physical.
  floor = float(rtt.min()) - 0.0025
  physical = [c for c in curve if c[1] >= floor]
  unconstrained = best
  best = min(physical, key=lambda c: c[0]) if physical else best
  err, L, tu, td = best
  ok = [c[1] for c in physical if c[0] <= 1.05 * err] or [L]
  no_model = windowed_error(model, fit_blocks, 0.0, 1e-3, 1e-3)
  m["loop_delay_s"] = float(L)
  m["loop_delay_range_s"] = [float(min(ok)), float(max(ok))]
  m["tau_up_s"], m["tau_down_s"] = float(tu), float(td)
  m["tau_s"] = float((tu + td) / 2)
  m["fit_rms_deg_per_0.2s"] = float(err)
  m["obs_delay_s"] = one_way
  m["action_delay_s"] = max(0.0, float(L) - one_way)
  m["action_delay_jitter_s"] = max(0.005, (max(ok) - min(ok)) / 2)
  say("MOTOR MODEL (fitted to the step responses and random commands together)")
  say("  error in how far a wheel turns in 0.2 s, for each loop delay (with the best time constants):")
  for c in curve:
    if c[1] <= max(ok) + 0.03:
      say(f"    delay {c[1] * 1000:3.0f} ms: {c[0]:5.2f} deg   (tau up {c[2] * 1000:3.0f} ms, down {c[3] * 1000:3.0f} ms)"
          + ("   <- best" if c is best else "") + ("   (shorter than a round trip)" if c[1] < floor else ""))
  if unconstrained is not best:
    say(f"  (the very best fit, {unconstrained[1] * 1000:.0f} ms, is shorter than the fastest round trip "
        f"{rtt.min() * 1000:.0f} ms, so the best delay at least that long is used)")
  say(f"  model: command sent -> wheel responds -> reading sent {L * 1000:.0f} ms (anything from {min(ok) * 1000:.0f} to "
      f"{max(ok) * 1000:.0f} ms fits within 5 %), time constant {tu * 1000:.0f} ms speeding up, {td * 1000:.0f} ms slowing down")
  say(f"  fit error {err:.2f} deg per 0.2 s (with no delay and no lag: {no_model:.2f})")
  say(f"  -> action delay (command -> wheel) about {m['action_delay_s'] * 1000:.0f} ms; reading delay (wheel -> computer) "
      f"at least {one_way * 1000:.0f} ms, plus {m['reading_late_p50_s'] * 1000:.0f} ms typical / "
      f"{m['reading_late_p90_s'] * 1000:.0f} ms (90 %) Bluetooth hold-up")
  if blocks and blocks[-1][0] == "random":
    _, wins, tb, angles = blocks[-1]
    sends = np.diff([w_[0] for w_ in wins])
    say(f"  random commands went out every {np.mean(sends) * 1000:.1f} ms (asked {config.CONTROL_DT * 1000:.0f}), "
        f"longest gap {sends.max() * 1000:.0f} ms")

  # gyro
  raw = extra.get("imu_raw_change_ccw_quarter")
  say("GYRO (YAW)")
  if raw is None or abs(raw) < 5:
    m["imu_yaw_rad_per_unit"] = extra.get("imu_yaw_rad_per_unit_previous", math.pi / 1800)
    m["imu_yaw_calibrated"] = bool(extra.get("imu_yaw_calibrated_previous", False))
    say("  not measured this time" + (" (kept the earlier result)" if "imu_yaw_rad_per_unit_previous" in extra else
        ": assuming tenths of a degree, counter-clockwise positive (as measured on this motor)"))
  else:
    unit = math.pi / 1800          # the Double Motor reports yaw in tenths of a degree (it wraps at +-1800)
    m["imu_yaw_rad_per_unit"] = unit if raw > 0 else -unit
    m["imu_yaw_calibrated"] = True
    say(f"  a turn counter-clockwise changed yaw by {raw:+.0f} tenths of a degree -> "
        f"{'counter-clockwise' if raw > 0 else 'clockwise'} positive")
  m["imu_yaw_resolution_rad"] = abs(m["imu_yaw_rad_per_unit"])

  # camera
  if len(camera_samples):
    motor_tests = (s.commands[0][0] - 0.5, s.commands[-1][0] + 1.0)     # not the gyro test: the whole motor turns then
    m.update(analyze_camera(camera_samples, t, pos[:, arm_index()] * enc_sign[arm_index()], say, one_way, motor_tests))

  m["notes"] = ("loop_delay_s: command sent -> reading on the motor's steady clock shows the change. action_delay_s: "
                "command -> wheel changes speed. obs_delay_s: fastest one-way Bluetooth trip; reading_late_*: extra "
                "hold-ups. tau_*: first-order time constants. speed_table_*: measured steady speed per command %.")
  return m, lines


def arm_index() -> int:
  import legoeducation as le
  return 0 if config.ARM_MOTOR == le.MOTOR_LEFT else 1


def analyze_camera(samples, t_enc, enc_deg, say, one_way: float, t_range=(-np.inf, np.inf)) -> dict:
  """Compare the camera's arm angle with the encoder, on the circle (so lost turns don't matter), slow frames only."""
  cam = np.asarray(samples)
  found = cam[:, 2] > 0.5
  out = {}
  say("CAMERA")
  fps = (len(cam) - 1) / (cam[-1, 0] - cam[0, 0]) if len(cam) > 1 else 0
  say(f"  {len(cam)} frames at {fps:.1f} fps, arm found in {100 * found.mean():.0f} %")
  tc, ac = cam[found, 0], np.radians(cam[found, 1])
  inside = (tc > max(t_enc[0] + 0.4, t_range[0])) & (tc < min(t_enc[-1] - 0.4, t_range[1]))
  tc, ac = tc[inside], ac[inside]
  enc = np.radians(enc_deg)
  speed = np.abs(np.gradient(np.interp(tc, t_enc, enc), tc)) if len(tc) > 2 else np.zeros(0)
  slow = speed < CAMERA_MAX_SPEED
  tc, ac = tc[slow], ac[slow]
  say(f"  {len(tc)} frames with the arm turning slower than {CAMERA_MAX_SPEED:.0f} rad/s are used")
  if len(tc) < 100:
    say("  too few usable frames; skipping the camera")
    return out

  def fit(direction: int, shift: float):
    z = np.exp(1j * (ac - direction * np.interp(tc + shift, t_enc, enc)))
    mean = z.mean()
    r = np.angle(z * np.exp(-1j * np.angle(mean)))
    return math.degrees(math.sqrt(np.mean(r ** 2))), abs(mean)

  best = None
  for direction in (1, -1):
    for shift in np.arange(-0.25, 0.25, 0.0025):
      rms, agree = fit(direction, shift)
      if best is None or rms < best[0]:
        best = (rms, direction, shift, agree)
  rms, direction, shift, agree = best
  side = min(fit(direction, shift - 0.06)[0], fit(direction, shift + 0.06)[0])
  sharp = side / rms - 1
  latency = one_way - shift
  out.update({"camera_rms_deg": rms, "camera_agreement": agree, "camera_direction": direction,
              "camera_shift_s": float(shift), "camera_shift_sharpness": float(sharp)})
  say(f"  camera vs encoder: {rms:.1f} deg rms apart, agreement {agree:.3f} (1 = perfect), direction {direction:+d}")
  if agree < 0.9 or rms > 15:
    say("  the camera tracking is too noisy to check the encoder or the timing. See README: camera tips.")
  elif sharp < 0.1:
    say(f"  angles agree, but the timing is not clear (moving the camera by 60 ms changes the error by only "
        f"{100 * sharp:.0f} %). More slow, well-lit frames would help.")
  else:
    out["camera_latency_s"] = float(latency)
    say(f"  the encoder reports a moment {abs(shift) * 1000:.0f} ms {'before' if shift < 0 else 'after'} the camera "
        f"shows it -> camera latency about {latency * 1000:.0f} ms. A webcam is usually 50 to 150 ms; much outside "
        f"that would mean the reading delay above is off.")
  return out


def save_plot(path: Path, s: Session, data: dict, m: dict, camera_samples) -> None:
  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    return
  t = steady_clock_times(data["t"])
  pos = data["pos"]
  t_ref = t[0]
  enc_sign = m["encoder_sign_tank_forward"]
  has_cam = len(camera_samples) > 0
  fig, ax = plt.subplots(4 if has_cam else 3, 1, figsize=(11, 13 if has_cam else 10))
  # sweep
  for (tc, left, _r, t_next) in command_windows(s.commands, "sweep"):
    w = in_window(t, tc + 0.8, t_next - 0.05)
    if left != 0 and w.sum() > 3:
      for i, c in ((0, "tab:blue"), (1, "tab:orange")):
        ax[0].plot(left, slope(t[w], np.radians(pos[w, i] * enc_sign[i])), "o", color=c)
  ax[0].plot(m["speed_table_percent"], m["speed_table_rad_s"], "k--", lw=1, label="table used by deploy.py")
  ax[0].set(xlabel="command (%)", ylabel="steady speed (rad/s)", title="speed sweep (blue left, orange right)")
  ax[0].legend()
  # steps and random: measured vs model, speed over each update (steady-clock times)
  model_t = [c for c in s.commands if c[3] in ("step", "random")]
  if model_t:
    mm = MotorModel(np.array([c[0] for c in model_t]),
                    np.interp([c[1] for c in model_t], m["speed_table_percent"], m["speed_table_rad_s"]))
    for k, label in ((1, "step"), (2, "random")):
      wins = command_windows(s.commands, label)
      if not wins:
        continue
      w = in_window(t, wins[0][0] - 0.2, wins[-1][3] + 0.3)
      tw = t[w]
      for i, c in ((0, "tab:blue"), (1, "tab:orange")):
        ang = np.radians(pos[w, i] * enc_sign[i])
        ax[k].plot(tw[1:] - t_ref, np.diff(ang) / np.diff(tw), color=c, lw=0.8)
      tg = np.arange(tw[0], tw[-1], 0.005)
      pm = mm.positions(tg, m["loop_delay_s"], m["tau_up_s"], m["tau_down_s"])
      ax[k].plot(tg[1:] - t_ref, np.diff(pm) / np.diff(tg), "k", lw=1.2, label="model")
      ax[k].set(xlabel="time (s)", ylabel="speed (rad/s)", title=f"{label} test: encoders (blue/orange) vs model (black)")
      ax[k].legend()
  if has_cam:
    cam = np.asarray(camera_samples)
    ok = cam[:, 2] > 0.5
    i = arm_index()
    enc = pos[:, i] * enc_sign[i] * m.get("camera_direction", 1)
    shift = m.get("camera_shift_s", 0.0)
    e_at = np.interp(cam[ok, 0] + shift, t, enc)
    resid = (cam[ok, 1] - e_at + 180) % 360 - 180
    resid -= np.median(resid)
    resid = (resid + 180) % 360 - 180
    ax[3].plot(cam[ok, 0] - t_ref, resid, ".", ms=2)
    ax[3].set(xlabel="time (s)", ylabel="camera - encoder (deg)", ylim=(-180, 180),
              title="camera minus encoder angle (flat near 0 = good tracking)")
  fig.tight_layout()
  fig.savefig(path, dpi=110)


# ======================================================================================================

def save_raw(out_dir: Path, data: dict, s: Session, cam, extra: dict) -> None:
  np.savez_compressed(out_dir / "raw_data.npz", **data,
                      commands=np.array([(c[0], c[1], c[2]) for c in s.commands]),
                      command_labels=np.array([c[3] for c in s.commands]),
                      pings=np.array(s.pings), camera=np.array(cam) if len(cam) else np.zeros((0, 3)),
                      marks=json.dumps(s.marks), extra=json.dumps(extra))


def previous_gyro() -> dict:
  """The gyro's units and sign from the last motor_model.json, so a run without --gyro keeps them."""
  path = HERE / "motor_model.json"
  if not path.exists():
    return {}
  prev = json.loads(path.read_text())
  if "imu_yaw_rad_per_unit" not in prev:
    return {}
  return {"imu_yaw_rad_per_unit_previous": prev["imu_yaw_rad_per_unit"],
          "imu_yaw_calibrated_previous": prev.get("imu_yaw_calibrated", False)}


def load_raw(folder: Path) -> tuple[dict, Session, np.ndarray, dict]:
  d = np.load(folder / "raw_data.npz")
  data = {k: d[k] for k in ("t", "pos", "speed", "power", "yaw", "gyro_z")}
  s = Session()
  s.commands = [(float(c[0]), int(c[1]), int(c[2]), str(lab)) for c, lab in zip(d["commands"], d["command_labels"])]
  s.pings = list(d["pings"])
  if "marks" in d:
    s.marks = {k: tuple(v) for k, v in json.loads(str(d["marks"])).items()}
  else:                                    # runs saved before marks were stored
    s.marks = {"update timing": (float(data["t"][0]), float(data["t"][0]) + 2.0)}
  extra = json.loads(str(d["extra"])) if "extra" in d else {}
  old = folder / "motor_model.json"
  if "imu_raw_change_ccw_quarter" not in extra and old.exists():
    prev = json.loads(old.read_text())
    if "imu_yaw_rad_per_unit" in prev:
      extra["imu_yaw_rad_per_unit_previous"] = prev["imu_yaw_rad_per_unit"]
      extra["imu_yaw_calibrated_previous"] = prev.get("imu_yaw_calibrated", False)
  return data, s, d["camera"], extra


def finish(out_dir: Path, s: Session, data: dict, cam, extra: dict, fake: bool) -> None:
  print("\n================ RESULTS ================")
  model, lines = analyze(s, data, cam, extra, config.NOTIFY_MS / 1000)
  if fake:
    model["fake"] = True
  (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
  (out_dir / "motor_model.json").write_text(json.dumps(model, indent=2))
  save_plot(out_dir / "characterization.png", s, data, model, cam)
  if fake:
    print(f"\nFAKE run: results in {out_dir} (motor_model.json NOT replaced)")
  else:
    shutil.copy(out_dir / "motor_model.json", HERE / "motor_model.json")
    print(f"\nsaved {HERE / 'motor_model.json'} (the simulation reads this) and everything in {out_dir}")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--fake", action="store_true", help="no hardware: use a simulated motor")
  ap.add_argument("--camera", action="store_true", help="also film the arm to cross-check the timing (optional)")
  ap.add_argument("--gyro", action="store_true", help="also redo the hand-turn gyro test")
  ap.add_argument("--reanalyze", metavar="RESULTS_FOLDER", help="redo the analysis of a saved run (no hardware)")
  ap.add_argument("--gyro-only", action="store_true", help="only redo the gyro test; updates motor_model.json")
  a = ap.parse_args()

  if a.gyro_only:
    path = HERE / "motor_model.json"
    if not path.exists():
      sys.exit("no motor_model.json yet: run the full characterization first")
    robot = LegoRobot()
    robot.connect()
    try:
      change = gyro_test(robot)
    finally:
      robot.disconnect()
    if change is None:
      return
    model = json.loads(path.read_text())
    unit = math.pi / 1800          # tenths of a degree; the test only settles the sign
    model["imu_yaw_rad_per_unit"] = unit if change > 0 else -unit
    model["imu_yaw_resolution_rad"] = unit
    model["imu_yaw_calibrated"] = True
    model["imu_raw_change_ccw_quarter"] = float(change)
    path.write_text(json.dumps(model, indent=2))
    print(f"updated {path}: yaw in tenths of a degree, {'counter-clockwise' if change > 0 else 'clockwise'} positive")
    return

  if a.reanalyze:
    folder = Path(a.reanalyze).resolve()
    data, s, cam, extra = load_raw(folder)
    finish(folder, s, data, cam, extra, fake=False)
    return

  tracker = None
  if a.camera:
    from arm_tracker import ArmTracker
    tracker = ArmTracker(config.CAMERA_NAME)
    print(f"using {tracker.camera_label()} (cameras: {tracker.camera_names})")
    print("Camera setup: a window opens. Click the axle centre, then the arm, check it, press Enter (Esc = no camera).")
    if not tracker.setup():
      print("no camera: continuing without it")
      tracker = None

  robot = FakeRobot() if a.fake else LegoRobot()
  robot.connect()
  s = Session(robot)
  out_dir = HERE / "results" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  out_dir.mkdir(parents=True, exist_ok=True)
  if tracker is not None:
    tracker.start()
    import cv2
    cv2.imwrite(str(out_dir / "camera_setup.jpg"), tracker.snapshot)
  print("\nThe motor will now spin, up to full speed, both ways, for about 1.5 minutes. Keep hands clear.")
  extra = {}
  try:
    extra = run_tests(s, skip_imu=not a.gyro or a.fake, with_camera=tracker is not None)
    extra.update(previous_gyro())
  finally:
    robot.stop()
    if tracker is not None:
      tracker.stop()
    robot.disconnect()
    data = as_arrays(robot.log)
    cam = tracker.samples if tracker is not None else []
    save_raw(out_dir, data, s, cam, extra)
  finish(out_dir, s, data, cam, extra, fake=a.fake)


if __name__ == "__main__":
  sys.exit(main())
