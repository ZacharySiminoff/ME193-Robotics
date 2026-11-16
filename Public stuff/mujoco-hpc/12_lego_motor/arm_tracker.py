"""Measure the arm's angle with the computer's camera.

setup() opens a window (on the main thread, which macOS requires):
  1. click the centre of the motor axle,
  2. click on the arm NEAR ITS TIP (its colour is what gets tracked, in a ring at that distance),
  3. turn the arm by hand and check the green line follows it, then press Enter.
  Keys: N = next camera, R = click again, + / - = colour tolerance, Esc = no camera.
start() then records (time, angle) in a background thread until stop(). Times use time.perf_counter(), the same
clock as the motor readings. The angle is counter-clockwise as seen by the camera, in degrees, unwrapped.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

try:
  import cv2
except ImportError:      # characterize.py without --camera works without OpenCV
  cv2 = None

WINDOW = "arm tracker  (click axle centre, then the arm; Enter = OK, N = next camera, Esc = skip)"


def list_cameras() -> list[str]:
  """Camera names in the order OpenCV numbers them on macOS (AVFoundation devices sorted by uniqueID).

  Mirrors orderedVideoCaptureDevices() in OpenCV's cap_avfoundation_mac.mm. Returns [] if AVFoundation's Python
  bindings (pyobjc-framework-AVFoundation) are missing; then cameras are only known by number.
  """
  try:
    import AVFoundation as AV
  except ImportError:
    return []
  devices = list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo))
  devices += list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeMuxed))
  devices.sort(key=lambda d: str(d.uniqueID()))
  return [str(d.localizedName()) for d in devices]


def find_camera(name: str | int) -> int:
  """OpenCV camera number for a camera whose name contains `name` (or `name` itself if it is a number)."""
  if isinstance(name, int):
    return name
  names = list_cameras()
  for i, n in enumerate(names):
    if name.lower() in n.lower():
      return i
  print(f"no camera named like '{name}' (found {names or 'no names'}); using camera 0")
  return 0


class ArmTracker:
  def __init__(self, camera: str | int = "FaceTime", width: int = 640, height: int = 480):
    if cv2 is None:
      raise ImportError("OpenCV is missing. Switch the environment on: source .venv/bin/activate")
    self.camera_names = list_cameras()
    self.camera_index, self.width, self.height = find_camera(camera), width, height
    self.cap = None
    self.center: tuple[int, int] | None = None
    self.ref_lab: np.ndarray | None = None
    self.radius = 0.0
    self.tolerance = 28.0          # Lab colour distance counted as "the arm"
    self.samples: list[tuple[float, float, bool]] = []   # (time, angle in degrees, found)
    self._thread: threading.Thread | None = None
    self._stop = threading.Event()
    self._unwrap_prev: float | None = None
    self._turns = 0.0
    self._last_angle: float | None = None
    self._rate = 0.0               # degrees per frame, smoothed
    self.snapshot = None

  # ---- camera ---------------------------------------------------------------------------------------------

  def camera_label(self) -> str:
    name = self.camera_names[self.camera_index] if self.camera_index < len(self.camera_names) else "?"
    return f"camera {self.camera_index}: {name}"

  def _open(self) -> bool:
    if self.cap is not None:
      self.cap.release()
    self.cap = cv2.VideoCapture(self.camera_index)
    if not self.cap.isOpened():
      return False
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, _ = self.cap.read()
    return ok

  # ---- detection --------------------------------------------------------------------------------------------

  def measure(self, frame: np.ndarray) -> tuple[float | None, np.ndarray | None]:
    """Angle of the arm in this frame (degrees, counter-clockwise in the image, -180..180), and the arm mask."""
    if self.center is None or self.ref_lab is None:
      return None, None
    cx, cy = self.center
    r_out = int(self.radius * 1.6) + 4
    x0, x1 = max(cx - r_out, 0), min(cx + r_out, frame.shape[1])
    y0, y1 = max(cy - r_out, 0), min(cy + r_out, frame.shape[0])
    roi = frame[y0:y1, x0:x1]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    dist = np.linalg.norm(lab - self.ref_lab, axis=-1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - cx, yy - cy)
    # only a ring around where the arm's tip was clicked: leaves out the hub, the motor body and most background
    ring = (rr > self.radius * 0.45) & (rr < self.radius * 1.35)
    mask = ((dist < self.tolerance) & ring).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))      # drop single stray pixels
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
      return None, mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.max() < 20:
      return None, mask.astype(bool)
    # candidates: patches at least a third the size of the biggest. With no history, take the biggest; otherwise the
    # one closest to where the arm was last frame (it moves only a few degrees per frame), so a look-alike patch
    # in the background doesn't steal the track.
    candidates = [1 + k for k in np.flatnonzero(areas >= areas.max() / 3)]
    angles = {}
    for k in candidates:
      b = labels == k
      dx, dy = (xx[b] - cx).mean(), (yy[b] - cy).mean()
      if math.hypot(dx, dy) >= self.radius * 0.3:          # a patch around the axle has no clear direction
        angles[k] = math.degrees(math.atan2(-dy, dx))
    if not angles:
      return None, mask.astype(bool)
    if self._last_angle is None or len(angles) == 1:
      pick = max(angles, key=lambda k: stats[k, cv2.CC_STAT_AREA])
    else:                       # nearest to where the arm should be now, from its recent speed
      predicted = self._last_angle + self._rate
      pick = min(angles, key=lambda k: abs((angles[k] - predicted + 180) % 360 - 180))
    if self._last_angle is not None:
      step = (angles[pick] - self._last_angle + 180) % 360 - 180
      self._rate = 0.6 * self._rate + 0.4 * step
    self._last_angle = angles[pick]
    return angles[pick], labels == pick

  # ---- setup window ----------------------------------------------------------------------------------------------

  def setup(self) -> bool:
    """Interactive setup. Returns False if the user skipped the camera (Esc)."""
    if not self._open():
      print(f"{self.camera_label()} did not open")
    clicks: list[tuple[int, int]] = []
    frame_holder = {}

    def on_mouse(event, x, y, _flags, _param):
      if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
        clicks.append((x, y))
        if len(clicks) == 2 and "frame" in frame_holder:
          self._set_reference(frame_holder["frame"], clicks)

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    accepted = False
    while True:
      ok, frame = self.cap.read() if self.cap is not None else (False, None)
      if not ok:
        frame = np.zeros((self.height, self.width, 3), np.uint8)
        cv2.putText(frame, f"{self.camera_label()} gives no picture: press N", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
      frame_holder["frame"] = frame
      view = frame.copy()
      if len(clicks) == 0:
        msg = "1) click the centre of the motor axle"
      elif len(clicks) == 1:
        msg = "2) click on the arm, near its tip"
        cv2.circle(view, clicks[0], 5, (0, 0, 255), -1)
      else:
        angle, mask = self.measure(frame)
        if mask is not None:
          cx, cy = self.center
          r = int(self.radius * 1.6) + 4
          y0, x0 = max(cy - r, 0), max(cx - r, 0)
          sub = view[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
          sub[mask] = (0.5 * sub[mask] + [0, 0, 127]).astype(np.uint8)
        cv2.circle(view, self.center, int(self.radius * 0.45), (255, 255, 0), 1)
        cv2.circle(view, self.center, int(self.radius * 1.35), (255, 255, 0), 1)
        if angle is None:
          msg = "arm NOT found: click R and try again, or +/- tolerance"
        else:
          end = (int(self.center[0] + self.radius * math.cos(math.radians(angle))),
                 int(self.center[1] - self.radius * math.sin(math.radians(angle))))
          cv2.line(view, self.center, end, (0, 255, 0), 3)
          msg = f"angle {angle:+6.1f} deg. Turn the arm by hand: does the green line follow? Enter = OK"
      cv2.putText(view, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
      cv2.putText(view, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
      cv2.putText(view, f"{self.camera_label()}   tolerance {self.tolerance:.0f}", (10, self.height - 12),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
      cv2.imshow(WINDOW, view)
      key = cv2.waitKey(15) & 0xFF
      if key in (13, 10, 32) and len(clicks) == 2:          # Enter / space
        accepted = True
        self.snapshot = view
        break
      if key == 27:                                       # Esc
        break
      if key in (ord("n"), ord("N")):
        self.camera_index = (self.camera_index + 1) % max(len(self.camera_names), 2)
        clicks.clear()
        self.center = None
        self._open()
      if key in (ord("r"), ord("R")):
        clicks.clear()
        self.center = None
      if key in (ord("+"), ord("=")):
        self.tolerance += 3
      if key in (ord("-"), ord("_")):
        self.tolerance = max(5, self.tolerance - 3)
    cv2.destroyWindow(WINDOW)
    cv2.waitKey(1)
    return accepted

  def _set_reference(self, frame: np.ndarray, clicks: list[tuple[int, int]]) -> None:
    self._last_angle, self._rate = None, 0.0
    self.center = clicks[0]
    ax, ay = clicks[1]
    self.radius = max(15.0, math.hypot(ax - self.center[0], ay - self.center[1]))
    patch = frame[max(ay - 4, 0):ay + 5, max(ax - 4, 0):ax + 5]
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    self.ref_lab = np.median(lab, axis=0)

  # ---- recording ------------------------------------------------------------------------------------------------

  def start(self) -> None:
    self._stop.clear()
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()

  def stop(self) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=2)
    if self.cap is not None:
      self.cap.release()

  def _run(self) -> None:
    while not self._stop.is_set():
      ok, frame = self.cap.read()
      t = time.perf_counter()
      if not ok:
        time.sleep(0.005)
        continue
      angle, _ = self.measure(frame)
      if angle is None:
        self.samples.append((t, float("nan"), False))
        continue
      if self._unwrap_prev is not None:           # keep counting whole turns
        jump = angle - self._unwrap_prev
        if jump > 180:
          self._turns -= 360
        elif jump < -180:
          self._turns += 360
      self._unwrap_prev = angle
      self.samples.append((t, angle + self._turns, True))

  def fps(self) -> float:
    if len(self.samples) < 2:
      return 0.0
    t = np.array([s[0] for s in self.samples])
    return (len(t) - 1) / (t[-1] - t[0])
