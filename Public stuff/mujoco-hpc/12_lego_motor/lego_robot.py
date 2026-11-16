"""The Double Motor over Bluetooth, with timestamps on everything. Used by characterize.py and deploy.py.

    robot = LegoRobot()
    robot.connect()                       # card number and color from config.py
    robot.set_speed(30, 30)               # percent, -100..100, left and right (the motor's own "forward" is +)
    robot.latest()                        # newest encoder/IMU reading, with the time it arrived
    robot.stop(); robot.disconnect()

FakeRobot has the same methods and simulates a motor, so the scripts can be tried without hardware (--fake).
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass

import config

LEFT, RIGHT = 0, 1


@dataclass
class Reading:
  """One update from the motor, stamped with the computer's clock when it arrived (time.perf_counter())."""
  t: float
  position: tuple[float, float]    # degrees, left and right, as counted by each motor
  speed: tuple[float, float]       # the motor's own speed field, left and right (percent units)
  power: tuple[float, float]
  yaw: float                       # IMU yaw, raw units (see motor_model.json imu_yaw_rad_per_unit)
  gyro_z: float


class LegoRobot:
  """Wraps legoeducation.DoubleMotor. Keeps every update (with arrival time) in self.log while recording."""

  def __init__(self):
    import legoeducation as le
    self.le = le
    self.dev = le.DoubleMotor()
    self._lock = threading.Lock()
    self._latest: Reading | None = None
    self.log: deque[Reading] = deque(maxlen=200_000)
    self.recording = False
    self.last_command = (0, 0)

  # ---- connection ---------------------------------------------------------------------------------------

  def connect(self, card_serial=None, card_color=None) -> None:
    serial = config.CARD_SERIAL if card_serial is None else card_serial
    color = config.CARD_COLOR if card_color is None else card_color
    print(f"connecting to the Double Motor with card {serial} ({self.le.LEGO_COLOR_NAME_MAP.get(color, color)}) ...")
    for attempt in range(5):
      try:
        self.dev.connect(card_serial=serial, card_color=color, device_notification_delay=config.NOTIFY_MS)
        break
      except Exception as e:     # the library says "not ready" if Bluetooth is still starting
        if "not ready" in str(e).lower() and attempt < 4:
          time.sleep(1)
        else:
          raise
    if not self.dev.connected:
      raise ConnectionError("could not connect to the Double Motor (check the card number and color in config.py)")
    self.dev.set_notification_callback(self._on_notification)
    a, d = config.ACCELERATION
    self.dev.motor_set_acceleration(a, d, motor=self.le.MOTOR_LEFT)
    self.dev.motor_set_acceleration(a, d, motor=self.le.MOTOR_RIGHT)
    self.dev.movement_set_acceleration(a, d)
    if getattr(config, "IMU_YAW_FACE", None) is not None:
      self.dev.imu_set_yaw_face(config.IMU_YAW_FACE)
    self.wait_for_reading()
    print(f"connected (firmware {getattr(self.dev, 'version_firmware', '?')}), updates every {config.NOTIFY_MS} ms")

  def disconnect(self) -> None:
    try:
      self.stop()
    finally:
      try:
        self.dev.disconnect()
      finally:
        # The library cleans up at exit in the wrong order and prints "can't create new thread at interpreter
        # shutdown". Clean up now, while that still works, and cancel its exit-time cleanup.
        import atexit
        try:
          self.dev._shutdown()
        except Exception:
          pass
        atexit.unregister(self.dev._shutdown)

  # ---- sensing ------------------------------------------------------------------------------------------

  def _on_notification(self, _notification) -> None:
    # Runs on the Bluetooth thread right after the library has stored the new values: only copy them here.
    t = time.perf_counter()
    m, imu = self.dev.motor, self.dev.imu_device
    r = Reading(
      t=t,
      position=(float(m[0].position), float(m[1].position)),
      speed=(float(m[0].speed), float(m[1].speed)),
      power=(float(m[0].power), float(m[1].power)),
      yaw=float(imu.yaw),
      gyro_z=float(imu.gyroscopeZ),
    )
    with self._lock:
      self._latest = r
      if self.recording:
        self.log.append(r)

  def latest(self) -> Reading | None:
    with self._lock:
      return self._latest

  def wait_for_reading(self, timeout: float = 3.0) -> Reading:
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
      r = self.latest()
      if r is not None and not math.isnan(r.position[0]):
        return r
      time.sleep(0.01)
    raise TimeoutError("no encoder updates from the motor")

  # ---- commands -----------------------------------------------------------------------------------------

  def set_speed(self, left_percent: float, right_percent: float, blocking: bool = False) -> float:
    """Run both motors at these speeds (percent, -100..100; the motor's own forward is +). Returns the send time."""
    left, right = int(round(max(-100, min(100, left_percent)))), int(round(max(-100, min(100, right_percent))))
    t = time.perf_counter()
    self.dev.movement_move_tank(left, right, blocking=blocking)
    self.last_command = (left, right)
    return t

  def stop(self) -> None:
    self.set_speed(0, 0, blocking=True)
    self.dev.movement_stop()

  def ping(self) -> float:
    """Round trip of one command that changes nothing: send, wait for the motor's reply. Seconds."""
    t0 = time.perf_counter()
    self.dev.movement_move_tank(*self.last_command, blocking=True)
    return time.perf_counter() - t0

  def reset_heading(self) -> None:
    self.dev.imu_reset_yaw_axis(0)


class FakeRobot:
  """A pretend Double Motor for trying the scripts without hardware: 30 ms each way, 60 ms time constant."""

  def __init__(self, one_way: float = 0.03, tau: float = 0.06, rad_s_per_percent: float = 0.14):
    self.one_way, self.tau, self.k = one_way, tau, rad_s_per_percent
    self._lock = threading.Lock()
    self._latest: Reading | None = None
    self.log: deque[Reading] = deque(maxlen=200_000)
    self.recording = False
    self.last_command = (0, 0)
    self._cmd_queue: deque = deque()
    self._state = {"cmd": [0.0, 0.0], "vel": [0.0, 0.0], "pos": [0.0, 0.0], "yaw": 0.0}
    self._stop = threading.Event()
    self._thread = threading.Thread(target=self._run, daemon=True)

  def connect(self, *_args, **_kw) -> None:
    print("FAKE robot: no Bluetooth, simulated motor")
    self._thread.start()
    time.sleep(0.1)

  def disconnect(self) -> None:
    self._stop.set()

  def _run(self) -> None:
    dt, next_notify, s = 0.002, 0.0, self._state
    t_prev = time.perf_counter()
    while not self._stop.is_set():
      time.sleep(dt)
      now = time.perf_counter()
      h, t_prev = now - t_prev, now
      while self._cmd_queue and self._cmd_queue[0][0] <= now:
        s["cmd"] = list(self._cmd_queue.popleft()[1])
      for i in range(2):
        target = s["cmd"][i] * self.k if abs(s["cmd"][i]) >= 3 else 0.0   # small dead zone
        s["vel"][i] += (target - s["vel"][i]) * (1 - math.exp(-h / self.tau))
        s["pos"][i] += math.degrees(s["vel"][i]) * h * (1 if i == 0 else -1)   # right side counts the other way
      if now >= next_notify:
        next_notify = now + config.NOTIFY_MS / 1000
        snap = Reading(t=now + self.one_way, position=(round(s["pos"][0]), round(s["pos"][1])),
                       speed=(round(math.degrees(s["vel"][0]) / 6.6), -round(math.degrees(s["vel"][1]) / 6.6)),
                       power=(0.0, 0.0), yaw=s["yaw"], gyro_z=0.0)
        threading.Timer(self.one_way, self._deliver, args=(snap,)).start()

  def _deliver(self, r: Reading) -> None:
    r.t = time.perf_counter()
    with self._lock:
      self._latest = r
      if self.recording:
        self.log.append(r)

  def latest(self) -> Reading | None:
    with self._lock:
      return self._latest

  def wait_for_reading(self, timeout: float = 3.0) -> Reading:
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
      if self.latest() is not None:
        return self.latest()
      time.sleep(0.01)
    raise TimeoutError("no fake readings")

  def set_speed(self, left_percent: float, right_percent: float, blocking: bool = False) -> float:
    left, right = int(round(left_percent)), int(round(right_percent))
    t = time.perf_counter()
    self._cmd_queue.append((t + self.one_way, (left, right)))
    self.last_command = (left, right)
    if blocking:
      time.sleep(2 * self.one_way)
    return t

  def stop(self) -> None:
    self.set_speed(0, 0, blocking=True)

  def ping(self) -> float:
    t0 = time.perf_counter()
    time.sleep(2 * self.one_way)
    return time.perf_counter() - t0

  def reset_heading(self) -> None:
    self._state["yaw"] = 0.0
