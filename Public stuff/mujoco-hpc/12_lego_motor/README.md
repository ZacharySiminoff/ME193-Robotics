# The real motor: measure it, train with it, drive with it

Three steps, all with the LEGO Education **Double Motor** over Bluetooth:

1. **`characterize.py`** (your Mac) measures the motor and saves `motor_model.json`.
2. **Train on the cluster** with `11_drive_forward/train_and_video.py`. The simulation reads `motor_model.json`, so the
   simulated motor has the real delays, speed and lag. The run brings back `policy.npz`.
3. **`deploy.py`** (your Mac) runs that policy on the real robot, over Bluetooth.

| File | What it is |
|---|---|
| `config.py` | **Edit first:** the Connection Card number and color, which side has the arm, the camera, the control rate |
| `lego_robot.py` | The Double Motor with timestamps: `connect()`, `set_speed(left %, right %)`, `latest()`, `stop()`. Also `FakeRobot` for trying things without hardware |
| `arm_tracker.py` | Finds the arm's angle in the camera picture |
| `characterize.py` | The tests (below). Writes `motor_model.json` and `results/<time>/` |
| `deploy.py` | Runs a trained policy on the robot. Enter starts or stops a run; no reconnecting between runs |

## Setup (once)

Everything here runs in its own Python environment, `.venv` in this folder (Python 3.12, with `legoeducation`,
OpenCV for the camera, numpy and matplotlib). It is already made. To make it again, e.g. on another computer:
```
cd "/Users/zacharysiminoff/Documents/Code/ME193 Chris Code/Public stuff/mujoco-hpc/12_lego_motor"
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```
Each time you open a new terminal, go to this folder and switch the environment on:
```
cd "/Users/zacharysiminoff/Documents/Code/ME193 Chris Code/Public stuff/mujoco-hpc/12_lego_motor"
source .venv/bin/activate
```
Your prompt then starts with `(.venv)`, and `python` means this environment's Python, so the commands below are
`python characterize.py` and so on. (`deactivate` switches it off.) In VS Code you can also pick it once:
Cmd+Shift+P > "Python: Select Interpreter" > `12_lego_motor/.venv`.

Then put your card's number and color in `config.py` (`CARD_SERIAL`, `CARD_COLOR`).

macOS asks once for Bluetooth and camera permission for your terminal (or VS Code). Say yes. If you said no:
System Settings > Privacy & Security > Bluetooth / Camera.

## 1. Characterize the motor

Both motor outputs must be free to spin, up to full speed, both ways. Best: the robot's own wheels on, and the robot
propped up so the wheels touch nothing. The motor then spins the same wheels it will drive. No arm and no camera needed.
```
python characterize.py
```
1. It connects and runs the tests (about 1.5 minutes; keep hands clear). Nothing to click.
2. Optional extras: `--camera` also films an arm to cross-check the timing (see "The camera" below), and `--gyro`
   redoes the hand-turn gyro test. Without `--gyro`, the gyro's units and sign from the last run are kept.

| Test | What it measures |
|---|---|
| update timing | how often encoder updates arrive, and how late Bluetooth delivers them |
| round trip | Bluetooth: send a command, wait for the motor's reply, 40 times |
| direction | which way each encoder counts when the motor drives "forward"; which output is left |
| speed sweep | steady speed at 10 to 100 % both ways: rad/s per %, dead zone, top speed, and a table for deploy.py |
| step responses + random commands | together: the delay from command to motion, and the time constants |
| camera moves | only with `--camera`: slow back-and-forth, so the camera sees the arm without blur |
| gyro | only with `--gyro`: you turn the motor a quarter turn left by hand: the yaw's units and sign |

It prints the results and saves them in `results/<time>/` (`summary.txt`, `characterization.png`, the raw data).
   It also saves `motor_model.json` here, which the simulation uses.

To redo the analysis of a saved run without the motor (for example after an update to this script):
```
python characterize.py --reanalyze results/20261116-144054
```

### How the timing is worked out (no camera needed)

The encoder positions are exact; only their arrival times are uncertain. The motor sends a reading on its own steady
clock (every 30 ms on this Mac, though 20 ms is asked for), and Bluetooth delivers most of them promptly but holds
some back by a whole interval or more, then delivers two at once. So:
1. **Reading times:** each reading's time is rebuilt from that steady clock (the earliest arrivals around it show
   where the clock is). How late readings actually arrive is saved too, and the simulation uses it.
2. **One model for all the data:** a delay, then a first-order speed change (time constant `tau`), fitted to the step
   responses and the random commands together. It compares how far each wheel turns in every 0.2 s window, so a small
   error in speed doesn't swamp the timing.
3. **Delay vs. time constant:** with a reading only every 30 ms, a slightly longer delay plus a slightly faster
   response looks almost the same. The summary lists the fit for every delay. A change can't show up in a reading
   sooner than one trip to the motor plus one trip back, so only delays of at least the fastest round trip count.
4. **The split:** the reading delay is the fastest one-way trip (half the fastest round trip) plus the hold-ups.
   The action delay is the rest of the loop.

### The camera (optional)

The camera only cross-checks the timing: it sees the arm without Bluetooth in the way, so it could tell how the delay
splits between the command trip and the reading trip (the script otherwise assumes both take the same time). The motor
model doesn't use it. To try it: `python characterize.py --camera`. A window opens: click the centre of the axle, then
the arm near its tip, turn the arm a full turn by hand (the green line must follow), and press Enter. The summary says whether the camera tracking was good enough to say anything. The plot's
last panel shows camera minus encoder angle over time: a flat line near 0 means good tracking.

Camera tips. Most of the difference comes from the setup, not the code:
- **Background:** plain, one colour, filling the whole ring around the arm, and different from the arm. A sheet of
  paper or card behind the motor works. Keep yourself (and dark clothes) out of the ring.
- **Arm:** one bright, saturated colour that appears nowhere else in view: yellow or green on a red board, or a
  strip of coloured tape or a sticker on the tip. Matte, not shiny. A long arm (5 to 10 cm) gives a cleaner angle.
- **Light:** bright and even, from the front, no shadow of the arm on the background. More light also means a
  shorter exposure, so less motion blur.
- **Camera:** square-on to the motor, close enough that the arm fills a good part of the picture, held still (on a
  stand, not handheld).
- **Clicking:** click the axle centre precisely, and the arm near its tip. Turn the arm a full turn by hand before
  pressing Enter: the green line must follow it all the way round.
- **Speed:** the camera runs at about 30 frames per second. Only frames where the arm turns slower than 6 rad/s are
  used. The "camera moves" test is slow on purpose.

### Why speed control, not position control

The robot's wheels keep turning, so each step the policy asks for wheel **speeds** (`movement_move_tank(left %, right %)`).
The motor's own position controller, with its easing, is made for "go to this angle and stop". As a driving
command, the target would have to keep moving ahead of the wheel, and the easing would fight the policy. The
motor's ramp (`ACCELERATION` in `config.py`) is set the same way in the tests and on the robot, so whatever it
does is part of what gets measured.

## 2. Train

From the `mujoco-hpc` folder (one level up):
```
caffeinate -i python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 \
  --with 11_drive_forward/drive_forward_task.py --with 11_drive_forward/train.py \
  --with 11_drive_forward/make_video.py --with 11_drive_forward/export_policy.py \
  --with 09_onshape_robot/onshape --with 12_lego_motor/motor_model.json \
  11_drive_forward/train_and_video.py
```
The job output starts with `[Drive-Forward] motor model: .../motor_model.json`. That's how you know it used your
measurements. `runs/train_and_video-<time>/` then has the video, and `policy.npz` + `policy_meta.json`.

## 3. Drive the real robot

Build the robot back (no arm), then first check the directions:
```
python deploy.py --check
```
It drives both wheels gently forward for 1 s. The robot should go forward, and both wheel speeds printed should be positive.
If it goes backwards, set `TANK_FORWARD_IS_ROBOT_FORWARD = False` in `config.py`. If the left/right wheels are mixed
up, set `SWAP_LEFT_RIGHT = True`.

Then run the policy (the newest `policy.npz` in `../runs/`, or give a run folder):
```
python deploy.py
python deploy.py ../runs/train_and_video-XXXXXXXX-XXXXXX --seconds 15
```
- **Enter** starts a run. Whichever way the robot points then counts as straight ahead (the gyro is zeroed).
- **Enter** again stops it early. Runs also stop by themselves after `--seconds` (30).
- Pick the robot up, put it back at the start, press **Enter** for the next run. No reconnecting.
- **q** + Enter quits (Ctrl-C also stops the motors and disconnects).

Each run is saved as `deploy_logs/run-<time>.csv`: time, yaw, wheel speeds, the policy's outputs, the percents
sent, and how old each reading was. Compare it with the simulation's video numbers to see where the real robot differs.

## If something goes wrong

| What you see | What to do |
|---|---|
| `could not connect` | Check `CARD_SERIAL` / `CARD_COLOR`; the motor must be on and not connected to another app or script. |
| `No module named 'legoeducation'` or `cv2` | The environment is not on: `source .venv/bin/activate` (the prompt shows `(.venv)`). |
| Camera window shows the wrong camera | Press N (the window shows each camera's name), or set `CAMERA_NAME` in `config.py`. Use the Mac's own camera, not the iPhone: Continuity Camera adds wireless delay. |
| `arm NOT found` | Click R and click the arm again on a well-lit part; +/- changes the colour tolerance. A plain background helps. |
| Robot drives backwards, or turns in place | `deploy.py --check`, then `TANK_FORWARD_IS_ROBOT_FORWARD` / `SWAP_LEFT_RIGHT` in `config.py`. |
| Gyro test: yaw changed by much less than 90 | The motor wasn't flat or didn't turn a full quarter turn, or it sits on its side (set `IMU_YAW_FACE` in `config.py`). Redo just that test: `python characterize.py --gyro-only`. |
| Robot drives straight but slowly, or wobbles | Compare `deploy_logs/` with the simulation; re-measure the motor on the floor-loaded robot, or retrain longer. |
