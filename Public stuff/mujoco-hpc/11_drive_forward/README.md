# Drive forward: train the Onshape robot to drive in a straight line

`drive_forward_task.py` is an [mjlab](https://github.com/mujocolab/mjlab) task for the robot you exported in
`09_onshape_robot/onshape/`. 4096 copies of the robot start pointing up to 0.4 rad off course and learn to drive
forward at `TARGET_SPEED` while staying on their start line. The policy outputs one speed per wheel.

## What this folder needs, and what it borrows

| From | What | Copied here? |
|---|---|---|
| `09_onshape_robot/onshape/` | the robot: `robot.xml`, `scene.xml`, `assets/` (the shapes) | **No.** The task reads it from there, so after you re-export in Onshape every task gets the new robot. `runIt.py` sends it along with `--with 09_onshape_robot/onshape`. |
| `09_onshape_robot/mjlab_task/drive_task.py` | the starting point for `drive_forward_task.py` | Copied and changed (see below) |
| `10_pingpong_rally/` | `train.py`, `make_video.py`, and the idea for `train_and_video.py` | Copied, with the task name changed |

How `drive_forward_task.py` differs from `drive_task.py`:
- **"Forward" is `FORWARD`, in the robot's own frame.** Your export's wheel axles point along x, so the robot drives
  along y: both wheels forward moves it towards **-y**. `drive_task.py` assumed +x, which for this robot would mean sideways.
- It only adds a free joint if the export doesn't already have one. The newer onshape-to-robot exports do, and adding a second one breaks the model.
- It finds the main body and the wheel bodies by itself, so the part names in Onshape don't matter (only the joint names `left_wheel` / `right_wheel` do).

| File | What it is |
|---|---|
| `drive_forward_task.py` | The task: robot, rewards, resets. **Tune this one.** |
| `train.py` | mjlab's train command, with the task registered first |
| `make_video.py` | Records `drive_forward.mp4` of a trained policy, with time, forward distance, sideways drift, heading, speed and wheel speeds drawn on every frame |
| `train_and_video.py` | **The one to run.** One job: trains, exports the policy for the real robot, then makes the video. |
| `export_policy.py` | Checkpoint -> `policy.npz` + `policy_meta.json` for `12_lego_motor/deploy.py` (run by `train_and_video.py`) |

To start another task with new goals, copy this folder (for example to `12_my_task/`), rename the task file, and
change in it `task_id`, `experiment_name`, the rewards and the terminations. Change the task name in `train.py`,
`make_video.py` and `train_and_video.py` to match.

## Commands (run from the `mujoco-hpc` folder)

**Train and get a video** (one job, about 4 minutes on a Pax GPU node; stopped at 10 minutes at the most):
```
caffeinate -i python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 \
  --with 11_drive_forward/drive_forward_task.py --with 11_drive_forward/train.py \
  --with 11_drive_forward/make_video.py --with 11_drive_forward/export_policy.py \
  --with 09_onshape_robot/onshape --with 12_lego_motor/motor_model.json \
  11_drive_forward/train_and_video.py
```
Leave out `--with 12_lego_motor/motor_model.json` until you have measured your motor with `12_lego_motor/characterize.py`
(the task then uses typical values and says so in the job output). The run also brings back `policy.npz` and
`policy_meta.json`: that is what `12_lego_motor/deploy.py` runs on the real robot.
When it ends, `drive_forward.mp4` opens on your laptop. Everything is in `runs/train_and_video-<time>/`: the video, the
job output (`slurm-<id>.out`: training progress, then the numbers once a second), and the checkpoints in
`logs/rsl_rl/drive_forward/<date>/`.

Change the amount of training by adding numbers at the end: iterations, envs, video seconds (default 500, 4096, 30).
- Quick test that everything works (about 2 minutes): `11_drive_forward/train_and_video.py 30 512 5`
- Longer training: `11_drive_forward/train_and_video.py 2000` (with `--time 00:30:00`)

It is learning if, in the job output, `Episode_Reward/forward_speed` and `Mean episode length` go up and
`Episode_Termination/tipped_over` goes down.

**Video of an older checkpoint only** (no training):
```
python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 \
  --with 11_drive_forward/drive_forward_task.py --with 09_onshape_robot/onshape \
  --with runs/<run folder>/logs/rsl_rl/drive_forward/<date folder>/model_499.pt \
  11_drive_forward/make_video.py model_499.pt
```

## Made for the real robot

The simulation copies the real LEGO Double Motor, as measured in `12_lego_motor/motor_model.json`:
- **Commands:** the policy runs every `control_dt` (50 ms). Its wheel speeds are rounded to whole motor percents,
  reach the motor after the measured delay, and the wheel follows with the measured time constant. Delay, time constant
  and gain are varied a little every episode, so the policy copes with a motor that isn't exactly like the measurement.
- **What the policy sees** is only what the real robot can measure, measured the same way, and late by the measured amount:
  the gyro's yaw (sin, cos), each wheel's encoder speed (whole degrees per control step), and its own last command.
  The critic, used only in training, also sees the true velocities and the sideways offset.

Checkpoints from before this change expect different inputs and cannot be used with it.

## Tuning (all in `drive_forward_task.py`)

- **It drives backwards or sideways:** check `FORWARD`. Spin both wheels forward in `../09_onshape_robot/drive.py` and see
  which way the robot goes; put that direction (in the robot's own frame) in `FORWARD`.
- **It tips over a lot:** lower `WHEEL_TORQUE_LIMIT` or `MAX_WHEEL_SPEED`, or make the `action_rate` weight more negative.
  When this was set up, driving both wheels backwards tipped the robot over within a second.
- **Speed:** `MAX_WHEEL_SPEED` is 6 rad/s (40 % of the first version's 15) and `TARGET_SPEED` 0.04 m/s to match. With both
  wheels at 7.5 rad/s the robot moves about 0.045 m/s. Change both together.
- **Floor grip:** `FLOOR_FRICTION` (1.5; MuJoCo's default is 1.0). MuJoCo uses the larger of two touching surfaces'
  frictions, so this applies to the chassis as well as the wheels.
- **It wiggles off the line:** raise the `stay_on_line` and `face_forward` weights, or lower `START_YAW`.
- **More training:** add `--agent.max-iterations 2000` and a longer `--time`.
