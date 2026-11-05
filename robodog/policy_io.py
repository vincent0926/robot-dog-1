"""找 checkpoint、載入訓練好的策略。

訓練輸出的資料夾結構：
  runs/<task>/<時間>-<名稱>/
      config.yaml          這次訓練用的完整設定
      tb/                  TensorBoard 紀錄
      checkpoints/<步數>/  每次評估存一份（brax/orbax 格式）

指定策略時可以給：
  runs/walk/latest                       → 最新一次 walk 訓練的最後一個 checkpoint
  runs/walk/20261105-120000-test         → 這次訓練的最後一個 checkpoint
  runs/walk/.../checkpoints/000050003968 → 指定某個 checkpoint
"""

from __future__ import annotations

import json
from pathlib import Path

from robodog.config import REPO_ROOT

_CKPT_CONFIG = "ppo_network_config.json"


def _is_ckpt(p: Path) -> bool:
  return p.is_dir() and (p / _CKPT_CONFIG).exists()


def latest_run(task_dir: Path) -> Path:
  runs = sorted(d for d in task_dir.iterdir() if (d / "checkpoints").is_dir())
  if not runs:
    raise FileNotFoundError(f"{task_dir} 底下找不到任何訓練紀錄，請先訓練。")
  return runs[-1]


def resolve_checkpoint(path: str | Path) -> Path:
  p = Path(path)
  if not p.is_absolute() and not p.exists() and (REPO_ROOT / p).exists():
    p = REPO_ROOT / p
  if p.name == "latest" and not p.exists():
    p = latest_run(p.parent)
  if _is_ckpt(p):
    return p.resolve()
  if (p / "checkpoints").is_dir():
    p = p / "checkpoints"
  if p.is_dir():
    ckpts = sorted(d for d in p.iterdir() if _is_ckpt(d))
    if ckpts:
      return ckpts[-1].resolve()
  raise FileNotFoundError(f"在 {path} 找不到 checkpoint（要有 {_CKPT_CONFIG}）。")


def _load_network_config(ckpt: Path):
  """讀 ppo_network_config.json。

  brax 0.14.2 的 load_config 遇到值為 null 的 kernel init 欄位會 KeyError，
  所以這裡自己讀，把 null 的欄位拿掉（交給 make_ppo_networks 用預設值）。
  """
  from brax.training import checkpoint as brax_ckpt
  from brax.training import networks
  from ml_collections import config_dict

  d = json.loads((ckpt / _CKPT_CONFIG).read_text())
  kw = d["network_factory_kwargs"]
  if "activation" in kw:
    kw["activation"] = networks.ACTIVATION[kw["activation"]]
  for key in brax_ckpt._KERNEL_INIT_FN_KEYWORDS:
    if key in kw:
      if kw[key] is None:
        kw.pop(key)
      else:
        kw[key] = networks.KERNEL_INITIALIZER[kw[key]]
  return config_dict.create(**d)


def load_inference_fn(path: str | Path, deterministic: bool = True):
  """回傳 policy(obs, rng) -> (action, extras)。"""
  from brax.training import checkpoint as brax_ckpt
  from brax.training.agents.ppo import checkpoint as ppo_checkpoint
  from brax.training.agents.ppo import networks as ppo_networks

  ckpt = resolve_checkpoint(path)
  print(f"[policy] 載入 {ckpt}")
  config = _load_network_config(ckpt)
  network = brax_ckpt.get_network(config, ppo_networks.make_ppo_networks)
  params = ppo_checkpoint.load(ckpt)
  return ppo_networks.make_inference_fn(network)(params, deterministic=deterministic)


def load_run(path: str | Path, impl: str | None = None, env_sets: dict | None = None):
  """載入一次訓練的設定、環境和策略。回傳 (cfg, env, policy, ckpt_path)。"""
  import yaml

  from robodog import envs
  from robodog.config import deep_merge

  ckpt = resolve_checkpoint(path)
  run_dir = ckpt.parent.parent
  with open(run_dir / "config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
  env_overrides = deep_merge(cfg.get("env", {}) or {}, env_sets or {})
  if impl:
    env_overrides["impl"] = impl
  env = envs.make_env(cfg["task"], env_overrides)
  policy = load_inference_fn(ckpt)
  return cfg, env, policy, ckpt
