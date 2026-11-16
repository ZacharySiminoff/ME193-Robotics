"""Turn a trained checkpoint into policy.npz + policy_meta.json for ../12_lego_motor/deploy.py.

    python export_policy.py logs/rsl_rl/drive_forward/<date>/model_499.pt

train_and_video.py runs this for you on the cluster. The .npz holds only numpy arrays (the observation normalizer
and the three layers of the actor network), so deploy.py on your Mac needs neither PyTorch nor mjlab.
policy_meta.json records what the policy expects: the order of its inputs, the control rate, the wheel speed a
policy output of 1.0 means, and the motor model it was trained with.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import drive_forward_task as task

if len(sys.argv) != 2:
  sys.exit("usage: export_policy.py model_499.pt")
checkpoint = Path(sys.argv[1])
state = torch.load(checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]

arrays = {
  "obs_mean": state["obs_normalizer._mean"].numpy().reshape(-1),
  "obs_std": state["obs_normalizer._std"].numpy().reshape(-1),
}
layers = sorted({int(k.split(".")[1]) for k in state if k.startswith("mlp.") and k.endswith(".weight")})
for n, i in enumerate(layers):
  arrays[f"w{n}"] = state[f"mlp.{i}.weight"].numpy()
  arrays[f"b{n}"] = state[f"mlp.{i}.bias"].numpy()
np.savez("policy.npz", **arrays)

obs_dim = sum(d for _, d in task.ACTOR_OBS)
assert arrays["obs_mean"].shape[0] == obs_dim, f"checkpoint expects {arrays['obs_mean'].shape[0]} inputs, task has {obs_dim}"
meta = {
  "checkpoint": checkpoint.name,
  "task": "Drive-Forward",
  "actor_obs": task.ACTOR_OBS,
  "obs_eps": 0.01,                 # rsl_rl EmpiricalNormalization: (x - mean) / (std + 0.01)
  "activation": "elu",
  "num_layers": len(layers),
  "control_dt": task.CONTROL_DT,
  "max_wheel_speed_rad_s": task.MAX_WHEEL_SPEED,
  "motor_model": task.MOTOR,
  "motor_model_source": task.MOTOR_SOURCE,
}
Path("policy_meta.json").write_text(json.dumps(meta, indent=2))
print(f"wrote policy.npz ({len(layers)} layers, {obs_dim} inputs) and policy_meta.json from {checkpoint}")
