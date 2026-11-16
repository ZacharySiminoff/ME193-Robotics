"""Train the Drive-Forward task. Same as mjlab's own `train`, but it registers our task first.

    python train.py Drive-Forward --env.scene.num-envs 4096
"""
import sys

import drive_forward_task  # noqa: F401  (registers "Drive-Forward")
from mjlab.scripts.train import main

if __name__ == "__main__":
  sys.exit(main())
