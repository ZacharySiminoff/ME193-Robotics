"""Train Drive-Forward, then record a video of the result, in one cluster job.

    python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 \
      --with 11_drive_forward/drive_forward_task.py --with 11_drive_forward/train.py \
      --with 11_drive_forward/make_video.py --with 11_drive_forward/export_policy.py \
      --with 09_onshape_robot/onshape --with 12_lego_motor/motor_model.json \
      11_drive_forward/train_and_video.py

Optional arguments, in this order: train_and_video.py [iterations, default 500] [number of envs, default 4096]
[video seconds, default 30]. A quick test that everything works: train_and_video.py 30 512 5

runIt.py brings back the checkpoints and training log (in logs/rsl_rl/drive_forward/<date>/), drive_forward.mp4
(opens on your laptop), and policy.npz + policy_meta.json, which ../12_lego_motor/deploy.py runs on the real robot.
Leave out --with 12_lego_motor/motor_model.json until you have measured your motor (typical values are used).
"""
import re
import subprocess
import sys
from pathlib import Path

import torch

iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 500
num_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
video_seconds = sys.argv[3] if len(sys.argv) > 3 else "30"


def run(cmd: list[str]) -> None:
  print("\n>>> " + " ".join(cmd), flush=True)
  subprocess.run([sys.executable, *cmd], check=True)


no_gpu = [] if torch.cuda.is_available() else ["--gpu-ids", "None"]   # without a GPU, train on the CPU (slow)
run(["train.py", "Drive-Forward", "--env.scene.num-envs", str(num_envs),
     "--agent.max-iterations", str(iterations), *no_gpu])

checkpoints = list(Path("logs/rsl_rl/drive_forward").glob("*/model_*.pt"))
if not checkpoints:
  sys.exit("training finished but saved no checkpoint (look for an error above)")
last = max(checkpoints, key=lambda p: (p.parent.name, int(re.findall(r"\d+", p.stem)[-1])))
print(f"\nlast checkpoint: {last}", flush=True)

run(["export_policy.py", str(last)])        # policy.npz + policy_meta.json for ../12_lego_motor/deploy.py
run(["make_video.py", str(last), video_seconds])
print(f"\nDONE: trained {iterations} iterations and recorded a video of {last.name}.")
