"""Quick end-to-end test on the cluster: train PingPong-Rally for a minute or two, then record a short video.

    python3 runIt.py --gpu --time 00:15:00 --mem 16G --cpus 4 \
      --with 10_pingpong_rally/rally_task.py --with 10_pingpong_rally/train.py --with 10_pingpong_rally/make_video.py \
      10_pingpong_rally/quick_test.py

It is far too short to learn a rally: the paddles will mostly flail. The point is to check that training runs on a
GPU node and that the checkpoints, the training log and a video all come back to runs/ on your laptop.
Optional arguments: quick_test.py [iterations, default 30] [number of envs, default 512]
"""
import re
import subprocess
import sys
from pathlib import Path

iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 30
num_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 512


def run(cmd: list[str]) -> None:
  print("\n>>> " + " ".join(cmd), flush=True)
  subprocess.run([sys.executable, *cmd], check=True)


import torch  # noqa: E402

no_gpu = [] if torch.cuda.is_available() else ["--gpu-ids", "None"]   # without a GPU, train on the CPU (slow)
run(["train.py", "PingPong-Rally", "--env.scene.num-envs", str(num_envs),
     "--agent.max-iterations", str(iterations), "--agent.save-interval", "10", *no_gpu])

checkpoints = list(Path("logs/rsl_rl/pingpong_rally").glob("*/model_*.pt"))
if not checkpoints:
  sys.exit("training finished but saved no checkpoint (look for an error above)")
last = max(checkpoints, key=lambda p: (p.parent.name, int(re.findall(r"\d+", p.stem)[-1])))
print(f"\nlast checkpoint: {last}", flush=True)

run(["make_video.py", str(last), "5"])
print("\nQUICK TEST OK: training ran, a checkpoint was saved, and a video was recorded.")
