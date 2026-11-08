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
import sys
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


def _scale_contact_buffers(cfg: dict, task: str) -> None:
  """Playground 預設的碰撞緩衝區（naconmax）是照 8192 個環境配置的，環境變少時跟著縮小，省顯示卡記憶體。

  YAML 裡自己寫了 env.naconmax 就不動。
  """
  from robodog import envs

  env_cfg = cfg["env"]
  if "naconmax" in env_cfg:
    return
  default = envs.default_env_config(task)
  if "naconmax" not in default:
    return
  per_env = default.naconmax / 8192
  n = max(int(cfg["ppo"]["num_envs"]), int(cfg["ppo"].get("num_eval_envs", 128)))
  env_cfg["naconmax"] = int(per_env * n * 1.25) + 1024  # 多留 25% 餘裕
  print(f"[train] 碰撞緩衝區 naconmax：{default.naconmax:,} → {env_cfg['naconmax']:,}（依 {n} 個環境縮小）")


def _quiet_solver_warnings(env) -> bool:
  """關掉 MuJoCo Warp 的「solver / linesearch iterations limit reached」警告。

  Playground 的 Go1 為了速度故意把求解器迭代次數設得很少（iterations=1），
  所以這個警告幾乎每一步都會出現，log 會變成好幾 GB，但它不代表模擬出錯。
  其他溢位警告（碰撞緩衝區 naconmax、njmax 不夠等）代表真的有問題，保留不動。
  回傳是否有修改（impl 不是 warp 時不用處理）。
  """
  from mujoco.mjx.third_party.mujoco_warp._src.types import OverflowType

  target = getattr(env, "_ll_env", env)  # 導航環境的物理模型在低階環境裡
  model = target._mjx_model
  impl = model.opt._impl
  if not hasattr(impl, "warn_overflow"):
    return False
  mask = int(impl.warn_overflow) & ~int(OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS)
  target._mjx_model = model.replace(opt=model.opt.replace(_impl=impl.replace(warn_overflow=mask)))
  return True


class _Tee:
  """把 print 同時寫到螢幕和 run 資料夾的 train.log，監看網站可以即時顯示。"""

  def __init__(self, stream, path: Path):
    self.stream = stream
    self.file = open(path, "a", encoding="utf-8", buffering=1)

  def write(self, text):
    self.stream.write(text)
    try:
      self.file.write(text)
    except ValueError:  # 檔案已關閉
      pass
    return len(text)

  def flush(self):
    self.stream.flush()
    try:
      self.file.flush()
    except ValueError:
      pass

  def __getattr__(self, name):
    return getattr(self.stream, name)


def _live_every(ppo_cfg: dict) -> int:
  """多久回報一次「即時」訓練數據（環境步數）。約整個訓練的 1/200，跟評估無關，網站因此能每十幾秒更新。"""
  return max(int(ppo_cfg["num_timesteps"]) // 200, 1)


class StatusFile:
  """把訓練狀態寫到 run 資料夾的 status.json，給監看網站（robodog.monitor）讀。"""

  def __init__(self, path: Path, **fields):
    self.path = path
    self.data = {"state": "starting", "pid": os.getpid(), "started": time.time(),
                 "updated": time.time(), "step": 0, **fields}
    self.write()

  def update(self, **fields) -> None:
    self.data.update(fields, updated=time.time())
    self.write()

  def write(self) -> None:
    tmp = self.path.with_suffix(".tmp")
    tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, self.path)  # 先寫暫存檔再換名，網站不會讀到寫一半的檔案


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

  env_cfg = cfg.setdefault("env", {}) or {}
  cfg["env"] = env_cfg
  if env_cfg.get("impl", "warp") == "warp":
    # MuJoCo Warp 的碰撞緩衝區是 JAX 記憶體池以外另外配置的。
    # JAX 預設一開始就佔走 75% 顯示卡記憶體，8GB 的卡會讓 Warp 不夠用，所以先讓出一半。
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
    _scale_contact_buffers(cfg, task)

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
  sys.stdout = _Tee(sys.stdout, run_dir / "train.log")
  sys.stderr = _Tee(sys.stderr, run_dir / "train.log")

  device = _gpu_summary()
  status = StatusFile(
      run_dir / "status.json", task=task, name=args.name, device=device,
      total_steps=int(cfg["ppo"]["num_timesteps"]), num_envs=int(cfg["ppo"]["num_envs"]),
      resumed_from=None,
  )
  print(f"[train] 任務：{task}")
  print(f"[train] 裝置：{device}")
  print(f"[train] 輸出：{run_dir}")

  # ---- 環境 ----
  env_overrides = cfg.get("env", {}) or {}
  env = envs.make_env(task, env_overrides)
  eval_env = envs.make_env(task, env_overrides)
  if not cfg.get("solver_warnings", False):
    if _quiet_solver_warnings(env) and _quiet_solver_warnings(eval_env):
      print("[train] 已關閉「iterations limit reached」警告（要看請加 --set solver_warnings=true）")
  rand_fn = envs.randomization_fn(task) if cfg.get("domain_randomization", False) else None

  ppo_cfg = dict(cfg["ppo"])
  ppo_cfg.setdefault("episode_length", env._config.episode_length)
  _check_batch(ppo_cfg)
  ppo_kwargs = _ppo_kwargs(ppo_cfg, task)
  # 兩次評估之間也回報訓練中的回合數據（brax 內建），監看網站才看得到即時進度
  ppo_kwargs.setdefault("log_training_metrics", True)
  ppo_kwargs.setdefault("training_metrics_steps", _live_every(ppo_cfg))

  # 導航任務：每次回合結束都重新隨機擺場地
  wrap_fn = functools.partial(
      wrapper.wrap_for_brax_training, full_reset=(task == "navigate")
  )

  restore = None
  if args.resume:
    restore = resolve_checkpoint(args.resume).as_posix()
    print(f"[train] 從 {restore} 接續訓練")
    status.update(resumed_from=restore)

  # ---- 紀錄 ----
  writer = SummaryWriter((run_dir / "tb").as_posix())
  t0 = time.time()
  times = [t0]

  last_print = [0.0]

  def live(step: int, metrics: dict) -> None:
    """訓練中的回合數據（不是評估）：寫進 TensorBoard 和 status.json，每分鐘印一行。"""
    for k, v in metrics.items():
      s = _scalar(v)
      if s is not None and math.isfinite(s):
        writer.add_scalar(k, s, step)
    writer.flush()
    r = _scalar(metrics.get("episode/sum_reward"))
    sps = _scalar(metrics.get("episode/sps"))
    elapsed = time.time() - t0
    status.update(state="running", step=max(int(step), status.data.get("step") or 0),
                  live_step=int(step), live_reward=r if r is not None and math.isfinite(r) else None,
                  live_length=_scalar(metrics.get("episode/length")),
                  sps=sps or status.data.get("sps"), elapsed_min=round(elapsed / 60, 2))
    if time.time() - last_print[0] > 60:
      last_print[0] = time.time()
      msg = f"[live]  step {step:>12,d} | {elapsed/60:6.1f} 分鐘"
      if r is not None:
        msg += f" | 訓練回合 reward {r:8.2f}"
      if sps:
        msg += f" | {sps:,.0f} steps/s"
      print(msg, flush=True)

  def progress(step: int, metrics: dict) -> None:
    if any(k.startswith("episode/") for k in metrics):
      return live(step, metrics)
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
    status.update(state="running", step=max(int(step), status.data.get("step") or 0), reward=r,
                  sps=sps or status.data.get("sps"), elapsed_min=round(elapsed / 60, 2))

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
  except BaseException as e:  # noqa: BLE001  （含 Ctrl+C）
    text = str(e)
    stopped = isinstance(e, KeyboardInterrupt)
    status.update(state="stopped" if stopped else "failed",
                  error=None if stopped else f"{type(e).__name__}: {text[:500]}")
    if "RESOURCE_EXHAUSTED" in text or "out of memory" in text.lower():
      half = max(ppo_cfg["num_envs"] // 2, 1)
      print("\n[train] 顯示卡記憶體不足！")
      print(f"[train] 建議改用：--set ppo.num_envs={half}")
      print("[train] 或先關掉其他吃顯卡的程式（瀏覽器、遊戲）。")
    raise
  finally:
    writer.close()

  total = time.time() - t0
  status.update(state="done", elapsed_min=round(total / 60, 2))
  summary = {"task": task, "minutes": round(total / 60, 2), "run_dir": run_dir.as_posix()}
  with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
  print(f"[train] 完成，共 {total/60:.1f} 分鐘。checkpoint 在 {ckpt_dir}")
  print(f"[train] 播放：python -m robodog.play {run_dir.relative_to(Path.cwd()) if run_dir.is_relative_to(Path.cwd()) else run_dir}")


if __name__ == "__main__":
  os.environ.setdefault("XLA_FLAGS", "--xla_gpu_triton_gemm_any=True")
  main()
