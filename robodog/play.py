"""播放訓練好的策略。

開 3D 視窗（需要螢幕，WSL2 在 Windows 11 可以直接開視窗）：
  python -m robodog.play runs/walk/latest

  walk 的鍵盤操作（滑鼠先點一下視窗）：
    ↑ / ↓   前進 / 後退速度 ±0.25 m/s
    ← / →   左轉 / 右轉速度 ±0.3 rad/s
    , / .   向左 / 向右平移 ±0.2 m/s
    /       全部歸零（停下來）

錄影（不用螢幕）：
  python -m robodog.play runs/walk/latest --video walk.mp4 --seconds 12
"""

from __future__ import annotations

import argparse
import time

import numpy as np

# GLFW 按鍵代碼
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265
KEY_COMMA, KEY_PERIOD, KEY_SLASH = 44, 46, 47

CAMERA = {"walk": "track", "getup": "track", "navigate": "overview"}


class Commander:
  """walk 任務用：把鍵盤變成速度指令。"""

  def __init__(self, limits):
    self.cmd = np.zeros(3, dtype=np.float32)
    self.limits = np.array(limits, dtype=np.float32)

  def key(self, keycode: int) -> None:
    step = {KEY_UP: (0, 0.25), KEY_DOWN: (0, -0.25), KEY_LEFT: (2, 0.3),
            KEY_RIGHT: (2, -0.3), KEY_COMMA: (1, 0.2), KEY_PERIOD: (1, -0.2)}
    if keycode == KEY_SLASH:
      self.cmd[:] = 0
    elif keycode in step:
      i, d = step[keycode]
      self.cmd[i] = np.clip(self.cmd[i] + d, -self.limits[i], self.limits[i])
    else:
      return
    print(f"\r指令 vx={self.cmd[0]:+.2f} vy={self.cmd[1]:+.2f} wz={self.cmd[2]:+.2f}   ", end="", flush=True)


def scripted_command(t: float) -> np.ndarray:
  """錄影時 walk 用的固定劇本：前進 → 轉彎 → 平移 → 後退 → 停。"""
  plan = [(3, [1.0, 0, 0]), (3, [0.6, 0, 0.8]), (2, [0, 0.5, 0]), (2, [-0.6, 0, 0]), (99, [0, 0, 0])]
  acc = 0.0
  for dur, cmd in plan:
    acc += dur
    if t < acc:
      return np.array(cmd, dtype=np.float32)
  return np.zeros(3, dtype=np.float32)


def make_stepper(env, policy, task):
  import jax
  import jax.numpy as jp

  @jax.jit
  def step(state, cmd, rng):
    if task == "walk":
      state.info["command"] = cmd
      obs = dict(state.obs)
      obs["state"] = obs["state"].at[-3:].set(cmd)
      state = state.replace(obs=obs)
    action, _ = policy(state.obs, rng)
    return env.step(state, action)

  return step, jax.jit(env.reset)


def record_video(env, policy, task, path, seconds, seed, width, height):
  import jax
  import imageio

  step, reset = make_stepper(env, policy, task)
  rng = jax.random.PRNGKey(seed)
  rng, k = jax.random.split(rng)
  state = reset(k)
  traj = [state]
  n = int(seconds / env.dt)
  for i in range(n):
    rng, k = jax.random.split(rng)
    state = step(state, scripted_command(i * env.dt), k)
    traj.append(state)
    if float(state.done) > 0:
      rng, k = jax.random.split(rng)
      state = reset(k)
  render_every = max(1, int(round((1 / 30) / env.dt)))
  frames = env.render(traj[::render_every], height=height, width=width, camera=CAMERA[task])
  fps = 1.0 / (env.dt * render_every)
  imageio.mimsave(path, frames, fps=fps)
  print(f"已存影片：{path}（{len(frames)} 格，{fps:.0f} fps）")


def interactive(env, policy, task, seed):
  import jax
  import mujoco
  import mujoco.viewer

  step, reset = make_stepper(env, policy, task)
  commander = Commander(env._config.command_config.a if task == "walk" else [0, 0, 0])
  model = env.mj_model
  data = mujoco.MjData(model)

  rng = jax.random.PRNGKey(seed)
  rng, k = jax.random.split(rng)
  state = reset(k)

  def sync(state):
    data.qpos[:] = np.asarray(state.data.qpos)
    data.qvel[:] = np.asarray(state.data.qvel)
    if model.nmocap:
      data.mocap_pos[:] = np.asarray(state.data.mocap_pos)
    mujoco.mj_forward(model, data)

  print("關掉視窗即結束。" + (" 用方向鍵控制，詳見 play.py 說明。" if task == "walk" else ""))
  with mujoco.viewer.launch_passive(model, data, key_callback=commander.key) as viewer:
    if task != "navigate":
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
      viewer.cam.trackbodyid = model.body("trunk").id
      viewer.cam.distance = 1.8
    while viewer.is_running():
      t0 = time.time()
      rng, k = jax.random.split(rng)
      state = step(state, commander.cmd, k)
      if float(state.done) > 0:
        rng, k = jax.random.split(rng)
        state = reset(k)
      sync(state)
      viewer.sync()
      time.sleep(max(0.0, env.dt - (time.time() - t0)))


def main(argv=None) -> None:
  p = argparse.ArgumentParser(description="播放訓練好的策略")
  p.add_argument("run", help="run 資料夾或 checkpoint，例如 runs/walk/latest")
  p.add_argument("--video", default=None, help="存成 mp4 而不是開視窗")
  p.add_argument("--seconds", type=float, default=12.0)
  p.add_argument("--impl", default=None, help="覆寫物理引擎實作：warp 或 jax（播放單一環境用 jax 也很快）")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--width", type=int, default=960)
  p.add_argument("--height", type=int, default=540)
  args = p.parse_args(argv)

  from robodog.policy_io import load_run

  cfg, env, policy, _ = load_run(args.run, impl=args.impl)
  task = cfg["task"]
  if args.video:
    record_video(env, policy, task, args.video, args.seconds, args.seed, args.width, args.height)
  else:
    interactive(env, policy, task, args.seed)


if __name__ == "__main__":
  main()
