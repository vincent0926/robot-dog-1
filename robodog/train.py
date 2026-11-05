"""訓練入口。

用法：
  python -m robodog.train configs/walk.yaml
  python -m robodog.train configs/walk.yaml --set ppo.num_envs=2048 --name lowmem
  python -m robodog.train configs/walk.yaml --resume runs/walk/latest
  python -m robodog.train configs/walk.yaml --smoke      # CPU 小規模測試，確認程式能跑
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import json
import math
import os
import time
from pathlib import Path

import yaml

from robodog.config import CONFIG_DIR, REPO_ROOT, load_config


def _build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description="訓練機器狗策略（PPO）")
  p.add_argument("config", help="設定檔，例如 configs/walk.yaml")
  p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                 help="覆寫設定，可重複使用，例如 --set ppo.num_envs=2048")
  p.add_argument("--name", default="", help="這次實驗的名稱（會加在資料夾名後面）")
  p.add_argument("--resume", default=None, help="從某個 checkpoint 接續訓練")
  p.add_argument("--smoke", action="store_true", help="疊上 configs/smoke.yaml，用 CPU 快速測試")
  p.add_argument("--out", default=None, help="輸出根目錄（預設 runs；--smoke 時為 runs_smoke）")
  return p


def _ppo_kwargs(ppo_cfg: dict, task: str):
  """把 YAML 的 ppo: 區塊轉成 brax ppo.train 的參數。"""
  from brax.training.agents.ppo import networks as ppo_networks

  kwargs = dict(ppo_cfg)
  net = kwargs.pop("network", {}) or {}
  for key in ("policy_hidden_layer_sizes", "value_hidden_layer_sizes"):
    if key in net:
      net[key] = tuple(net[key])
  kwargs["network_factory"] = functools.partial(ppo_networks.make_ppo_networks, **net)
  return kwargs


def _check_batch(ppo_cfg: dict) -> None:
  """brax 規定 batch_size * num_minibatches 必須能被 num_envs 整除。"""
  n, b, m = ppo_cfg["num_envs"], ppo_cfg["batch_size"], ppo_cfg["num_minibatches"]
  if (b * m) % n != 0:
    raise ValueError(
        f"batch_size({b}) × num_minibatches({m}) = {b*m} 必須是 num_envs({n}) 的倍數。"
        "通常 num_envs 用 2 的次方（1024/2048/4096）就沒問題。"
    )


def _gpu_summary() -> str:
  import jax
  dev = jax.devices()[0]
  info = f"{dev.platform.upper()} {getattr(dev, 'device_kind', '')}"
  try:
    stats = dev.memory_stats() or {}
    if "bytes_limit" in stats:
      info += f"，JAX 可用記憶體 {stats['bytes_limit'] / 2**30:.1f} GB"
  except Exception:  # CPU 沒有 memory_stats
    pass
  return info


def _scalar(v):
  try:
    return float(v)
  except (TypeError, ValueError):
    return None


def main(argv=None) -> None:
  args = _build_parser().parse_args(argv)
  extra = [CONFIG_DIR / "smoke.yaml"] if args.smoke else []
  cfg = load_config(args.config, extra_files=extra, sets=args.set)
  task = cfg["task"]
  out_root = args.out or ("runs_smoke" if args.smoke else "runs")
  if args.smoke and task == "navigate":
    # 冒煙測試和正式訓練分開放，避免 runs/walk/latest 被測試用的爛策略蓋過
    llp = (cfg.get("env") or {}).get("low_level_policy", "runs/walk/latest")
    if llp.startswith("runs/"):
      cfg.setdefault("env", {})["low_level_policy"] = "runs_smoke/" + llp[len("runs/"):]

  import jax
  from brax.training.agents.ppo import train as ppo
  from mujoco_playground import wrapper
  from tensorboardX import SummaryWriter

  from robodog import envs
  from robodog.policy_io import resolve_checkpoint

  # ---- 輸出資料夾 ----
  stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
  run_dir = Path(out_root) / task / (stamp + (f"-{args.name}" if args.name else ""))
  if not run_dir.is_absolute():
    run_dir = Path.cwd() / run_dir
  ckpt_dir = run_dir / "checkpoints"
  run_dir.mkdir(parents=True, exist_ok=True)
  with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

  print(f"[train] 任務：{task}")
  print(f"[train] 裝置：{_gpu_summary()}")
  print(f"[train] 輸出：{run_dir}")

  # ---- 環境 ----
  env_overrides = cfg.get("env", {}) or {}
  env = envs.make_env(task, env_overrides)
  eval_env = envs.make_env(task, env_overrides)
  rand_fn = envs.randomization_fn(task) if cfg.get("domain_randomization", False) else None

  ppo_cfg = dict(cfg["ppo"])
  ppo_cfg.setdefault("episode_length", env._config.episode_length)
  _check_batch(ppo_cfg)
  ppo_kwargs = _ppo_kwargs(ppo_cfg, task)

  # 導航任務：每次回合結束都重新隨機擺場地
  wrap_fn = functools.partial(
      wrapper.wrap_for_brax_training, full_reset=(task == "navigate")
  )

  restore = None
  if args.resume:
    restore = resolve_checkpoint(args.resume).as_posix()
    print(f"[train] 從 {restore} 接續訓練")

  # ---- 紀錄 ----
  writer = SummaryWriter((run_dir / "tb").as_posix())
  t0 = time.time()
  times = [t0]

  def progress(step: int, metrics: dict) -> None:
    times.append(time.time())
    for k, v in metrics.items():
      s = _scalar(v)
      if s is not None:
        writer.add_scalar(k, s, step)
    writer.flush()
    r = _scalar(metrics.get("eval/episode_reward"))
    if r is not None and math.isnan(r):
      raise FloatingPointError("訓練發散（reward 變成 NaN），已停止。上一個 checkpoint 仍可使用。")
    elapsed = times[-1] - t0
    msg = f"[train] step {step:>12,d} | {elapsed/60:6.1f} 分鐘"
    if r is not None:
      msg += f" | eval reward {r:8.2f}"
    sps = _scalar(metrics.get("training/sps"))
    if sps:
      msg += f" | {sps:,.0f} steps/s"
    print(msg, flush=True)

  def check_params(step, make_policy, params) -> None:
    # 在存 checkpoint 之前檢查參數有沒有 NaN，有就停下來，避免存到壞掉的權重
    leaves = jax.tree_util.tree_leaves(params[1])
    if any(bool(jax.numpy.isnan(x).any()) for x in leaves):
      raise FloatingPointError(f"step {step} 的策略參數出現 NaN，已停止（不會存這份 checkpoint）。")

  try:
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrap_fn,
        randomization_fn=rand_fn,
        progress_fn=progress,
        policy_params_fn=check_params,
        save_checkpoint_path=ckpt_dir.as_posix(),
        restore_checkpoint_path=restore,
        seed=cfg.get("seed", 0),
        **ppo_kwargs,
    )
  except Exception as e:  # noqa: BLE001
    text = str(e)
    if "RESOURCE_EXHAUSTED" in text or "out of memory" in text.lower():
      half = max(ppo_cfg["num_envs"] // 2, 1)
      print("\n[train] 顯示卡記憶體不足！")
      print(f"[train] 建議改用：--set ppo.num_envs={half}")
      print("[train] 或先關掉其他吃顯卡的程式（瀏覽器、遊戲）。")
    raise
  finally:
    writer.close()

  total = time.time() - t0
  summary = {"task": task, "minutes": round(total / 60, 2), "run_dir": run_dir.as_posix()}
  with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
  print(f"[train] 完成，共 {total/60:.1f} 分鐘。checkpoint 在 {ckpt_dir}")
  print(f"[train] 播放：python -m robodog.play {run_dir.relative_to(Path.cwd()) if run_dir.is_relative_to(Path.cwd()) else run_dir}")


if __name__ == "__main__":
  os.environ.setdefault("XLA_FLAGS", "--xla_gpu_triton_gemm_any=True")
  main()
