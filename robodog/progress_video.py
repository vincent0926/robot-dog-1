"""把一次訓練的每個 checkpoint 各錄一段影片，用來看「訓練過程中機器狗學到了什麼」。

用法：
  python -m robodog.progress_video runs/walk/latest
  python -m robodog.progress_video runs/walk/20261105-223530-cloud-cpu --seconds 8

輸出（放在該次訓練資料夾的 videos/ 底下）：
  step_000000000000.mp4 …   每個 checkpoint 一段，畫面左上角標出步數和評估 reward
  progress.mp4              全部接在一起，從頭看到尾就是整個學習過程
  progress.json             每段影片對應的步數、reward、跌倒次數

沒有螢幕的機器（例如雲端）請先設定：export MUJOCO_GL=osmesa
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robodog.play import CAMERA, scripted_command


def _eval_rewards(run_dir: Path) -> dict[int, float]:
  """從 TensorBoard 紀錄讀出每個步數的評估 reward（讀不到就回傳空的）。"""
  try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  except ImportError:
    return {}
  ea = EventAccumulator((run_dir / "tb").as_posix())
  ea.Reload()
  if "eval/episode_reward" not in ea.Tags().get("scalars", []):
    return {}
  return {int(e.step): float(e.value) for e in ea.Scalars("eval/episode_reward")}


def _label(frames, text: str):
  from PIL import Image, ImageDraw, ImageFont

  try:
    font = ImageFont.load_default(size=22)
  except TypeError:  # 舊版 Pillow
    font = ImageFont.load_default()
  out = []
  for f in frames:
    img = Image.fromarray(f)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 34], fill=(0, 0, 0))
    draw.text((10, 6), text, fill=(255, 255, 255), font=font)
    out.append(np.asarray(img))
  return out


def _fmt_steps(n: int) -> str:
  return f"{n/1e6:.1f}M" if n >= 1e6 else f"{n/1e3:.0f}k"


def main(argv=None) -> None:
  p = argparse.ArgumentParser(description="每個 checkpoint 錄一段影片")
  p.add_argument("run", help="run 資料夾，例如 runs/walk/latest")
  p.add_argument("--seconds", type=float, default=8.0)
  p.add_argument("--width", type=int, default=480)
  p.add_argument("--height", type=int, default=320)
  p.add_argument("--fps", type=int, default=25)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--only-new", action="store_true", help="已經錄過的 checkpoint 跳過")
  p.add_argument("--impl", default=None, help="覆寫物理引擎實作：warp 或 jax（訓練中錄影建議用 jax + CPU，不搶顯示卡）")
  args = p.parse_args(argv)

  import imageio
  import jax
  import yaml
  from brax.training import checkpoint as brax_ckpt
  from brax.training.agents.ppo import checkpoint as ppo_checkpoint
  from brax.training.agents.ppo import networks as ppo_networks

  from robodog import envs
  from robodog.policy_io import _is_ckpt, _load_network_config, resolve_checkpoint

  run_dir = resolve_checkpoint(args.run).parent.parent
  ckpts = sorted(d for d in (run_dir / "checkpoints").iterdir() if _is_ckpt(d))
  cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
  task = cfg["task"]
  out_dir = run_dir / "videos"
  out_dir.mkdir(exist_ok=True)
  rewards = _eval_rewards(run_dir)
  previous = {}
  if (out_dir / "progress.json").exists():
    previous = {c["step"]: c for c in json.loads((out_dir / "progress.json").read_text())["clips"]}

  env_cfg = dict(cfg.get("env") or {})
  if args.impl:
    env_cfg["impl"] = args.impl
  env = envs.make_env(task, env_cfg)
  network = brax_ckpt.get_network(_load_network_config(ckpts[-1]), ppo_networks.make_ppo_networks)
  make_policy = ppo_networks.make_inference_fn(network)

  # 策略參數當作 jit 的輸入，換 checkpoint 不用重新編譯
  @jax.jit
  def step(state, cmd, params, rng):
    if task in ("walk", "rough"):
      state.info["command"] = cmd
      obs = dict(state.obs)
      obs["state"] = obs["state"].at[-3:].set(cmd)
      state = state.replace(obs=obs)
    action, _ = make_policy(params, deterministic=True)(state.obs, rng)
    return env.step(state, action)

  reset = jax.jit(env.reset)
  every = max(1, int(round(1 / (args.fps * env.dt))))
  fps = 1.0 / (env.dt * every)
  n_steps = int(args.seconds / env.dt)

  summary, clips = [], []
  for ckpt in ckpts:
    n = int(ckpt.name)
    path = out_dir / f"step_{ckpt.name}.mp4"
    r = rewards.get(n)
    text = f"step {_fmt_steps(n)}" + (f"   eval reward {r:.1f}" if r is not None else "")
    if args.only_new and path.exists():
      clips.append(imageio.mimread(path, memtest=False))
      summary.append(previous.get(n, {"step": n, "reward": r, "video": path.name}))
      continue

    params = ppo_checkpoint.load(ckpt)
    rng = jax.random.PRNGKey(args.seed)
    rng, k = jax.random.split(rng)
    state = reset(k)
    traj, falls, fwd = [state], 0, []
    for i in range(n_steps):
      rng, k = jax.random.split(rng)
      state = step(state, scripted_command(i * env.dt), params, k)
      traj.append(state)
      if task in ("walk", "rough") and 1.0 <= i * env.dt < 3.0:  # 劇本前 3 秒是「前進 1 m/s」
        fwd.append(float(env.get_local_linvel(state.data)[0]))
      if float(state.done) > 0:
        falls += 1
        rng, k = jax.random.split(rng)
        state = reset(k)
    frames = env.render(traj[::every], height=args.height, width=args.width, camera=CAMERA[task])
    fwd_speed = float(np.mean(fwd)) if fwd else None
    extra = f"   fwd {fwd_speed:.2f} m/s (cmd 1.0)   falls {falls}" if task in ("walk", "rough") else ""
    frames = _label(frames, text + extra)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=8)
    clips.append(frames)
    summary.append({"step": n, "reward": r, "falls": falls, "fwd_speed": fwd_speed, "video": path.name})
    print(f"[video] {path.name}  {text}{extra}", flush=True)

  all_frames = [f for clip in clips for f in clip]
  imageio.mimsave(out_dir / "progress.mp4", all_frames, fps=fps, macro_block_size=8)
  (out_dir / "progress.json").write_text(
      json.dumps({"task": task, "run": run_dir.name, "seconds": args.seconds, "clips": summary},
                 ensure_ascii=False, indent=2),
      encoding="utf-8",
  )
  print(f"[video] 完成：{out_dir / 'progress.mp4'}")


if __name__ == "__main__":
  main()
