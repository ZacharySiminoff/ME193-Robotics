"""Train the PingPong-Rally task. Same as mjlab's own `train`, but it registers our task first.

    python train.py PingPong-Rally --env.scene.num-envs 4096
"""
import sys

import rally_task  # noqa: F401  (registers "PingPong-Rally")
from mjlab.scripts.train import main

if __name__ == "__main__":
  sys.exit(main())
