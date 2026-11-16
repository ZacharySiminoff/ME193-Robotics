# Ping-pong rally: two paddles keep a ball going between them

`rally_task.py` is an [mjlab](https://github.com/mujocolab/mjlab) reinforcement-learning task. Two paddles stand 1.2 m apart
(green on the left, blue on the right). The ball (the one from `08_demo.py`) is served from the middle at the target
height towards a random paddle. Each paddle can slide sideways, move up and down, and tilt; tilting is what sends the
ball across. One policy controls both paddles (6 outputs).

**Rules.** Only the paddle whose turn it is may hit the ball; then it is the other paddle's turn. A double hit (the same
paddle twice in a row) is a fault and ends the episode, and so does dropping the ball.

**Rewards** (weights in `rally_env_cfg`):

| Term | Weight | What it pays for |
|---|---|---|
| `hit` | +1 per hit | a hit made in turn |
| `flight` | up to +1 per second | while the ball flies: its peak is close to `TARGET_APEX` (1.2 m) **and** it will come down on the paddle whose turn it is. This is what makes the hits "nice". |
| `fault` | -5 per fault | a double hit or a dropped ball |
| `action_rate` | -0.01 | jerky paddle motion |

mjlab multiplies every reward by the step time (0.01 s). `hit` and `fault` divide by it, so their weights are per event.

| File | What it is |
|---|---|
| `rally_task.py` | The task: paddles, ball, rules, rewards. **Tune this one.** |
| `train.py` | mjlab's own train command, with our task registered first |
| `make_video.py` | Records a 15 s movie of a trained policy on the cluster and prints every hit |
| `quick_test.py` | One short job: trains briefly, then makes a 5 s video. Use it to check everything works. |

Everything runs from your laptop with `runIt.py`, in the `mujoco-hpc` folder (VPN on if off campus).

## 1. Quick test: does it run, and do results come back? (a few minutes)

```
python3 runIt.py --gpu --time 00:15:00 --mem 16G --cpus 4 \
  --with 10_pingpong_rally/rally_task.py --with 10_pingpong_rally/train.py --with 10_pingpong_rally/make_video.py \
  10_pingpong_rally/quick_test.py
```

`quick_test.py` trains 512 envs for 30 iterations, then records a 5 s video with the last checkpoint. The output ends
with `QUICK TEST OK`. The video opens on your laptop, and the checkpoints and training log are in `runs/quick_test-<time>/`.
30 iterations is far too short to learn a rally, so the paddles will mostly flail. For a longer test, add
iterations and envs at the end: `10_pingpong_rally/quick_test.py 100 1024`.

## 2. Train

```
python3 runIt.py --gpu --time 01:00:00 --mem 16G --cpus 4 \
  --with 10_pingpong_rally/rally_task.py \
  10_pingpong_rally/train.py PingPong-Rally --env.scene.num-envs 4096
```

The default is 1500 iterations, with a checkpoint every 100. Checkpoints come back to
`runs/train-<time>/logs/rsl_rl/pingpong_rally/<date>/model_<n>.pt`; the last one is `model_1499.pt`.

How to tell it is learning: in the job output, `Episode_Reward/hit` and `Mean episode length` should go up, and
`Episode_Termination/ball_dropped` should go down. A perfect 10 s episode is about 14 hits.

## 3. Watch it

```
python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 \
  --with 10_pingpong_rally/rally_task.py \
  --with runs/train-XXXXXXXX-XXXXXX/logs/rsl_rl/pingpong_rally/XXXX-XX-XX_XX-XX-XX/model_1499.pt \
  10_pingpong_rally/make_video.py model_1499.pt
```

Replace the `X`s with your own folder names (type `runs/train-` and press Tab).

## Tuning (all in `rally_task.py`)

- **Rallies are short because the ball dies down:** every bounce loses energy, so the paddle has to swing up into the
  ball. Give the paddles more punch with `SLIDE_KP` and `SLIDE_FORCE`, or make the ball bouncier: lower the second
  number of `BALL_SOLREF`.
- **It gets the first hit (the serve) but never sends the ball across:** widen `LAND_SIGMA` and `APEX_SIGMA` so poor
  early hits still earn some `flight` reward. Or raise the `hit` weight.
- **Hits reach the other side but at the wrong height:** narrow `APEX_SIGMA`, or raise the `flight` weight.
- **Harder or easier:** change `GAP`, `TARGET_APEX`, or the serve randomness (`SERVE_*_JITTER`).
- **Not enough training:** add `--agent.max-iterations 3000` (and a longer `--time`).
