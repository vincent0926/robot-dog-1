"""把「盲走」的樓梯斜坡策略擴充成「看得到地形」的版本（加上 terrain.height_scan 的 85 個輸入）。

用法：
  python -m robodog.expand_ckpt runs/terrain/20261115-204812-stage2 --out runs_init/scan-from-stage2
  python -m robodog.train configs/terrain.yaml --name scan --resume runs_init/scan-from-stage2 \\
      --set env.terrain.height_scan=true ...

做的事（類似 Net2Net）：
  - 策略網路和價值網路的第一層，對應新輸入的權重全部設成 0
    → 擴充後的策略一開始的動作和原本「完全一樣」，會走路、會爬低台階的能力全部保留，
      之後訓練時再慢慢學會利用地形資訊。
  - 觀測正規化（normalizer）裡新輸入的平均值和標準差，用隨機擺在各種地形上量到的掃描值估計。
    （原本的統計已經累積了幾億筆，之後幾乎不會再變，所以一開始就要給合理的值。）
輸出放在 runs/ 以外（預設 runs_init/），監看網站不會把它當成一次訓練。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from robodog.policy_io import _CKPT_CONFIG, resolve_checkpoint

STATE_DIM_BLIND = 48   # Go1 Joystick 的 state 長度（最後 3 格是速度指令）


def scan_stats(env, n: int = 16384, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
  """在所有區塊（每種地形、每個難度）隨機位置、隨機朝向，量掃描值的平均和標準差。"""
  import jax
  import jax.numpy as jp

  from robodog.envs import terrain as T

  rng = np.random.default_rng(seed)
  lv = rng.integers(0, T.N_LEVELS, n)
  ty = rng.integers(0, T.N_TYPES, n)
  centers = np.asarray(env._centers_np)[lv, ty]
  xy = centers[:, :2] + rng.uniform(-T.TILE / 2, T.TILE / 2, (n, 2))
  yaw = rng.uniform(-np.pi, np.pi, n)
  ground = np.asarray(env.terrain_height(jp.asarray(xy, dtype=jp.float32)))
  z = ground + T.SCAN_NOMINAL + rng.normal(0, 0.03, n)
  base = jp.asarray(np.c_[xy, z], dtype=jp.float32)
  scans = np.asarray(jax.jit(jax.vmap(env.height_scan_at))(base, jp.asarray(yaw, dtype=jp.float32)))
  return scans.mean(0), np.maximum(scans.std(0), 0.02)


def _insert(a: np.ndarray, at: int, block: np.ndarray) -> np.ndarray:
  a = np.asarray(a)
  return np.concatenate([a[:at], block.astype(a.dtype), a[at:]], axis=0)


def expand_params(params, k: int, mean: np.ndarray, std: np.ndarray, at: int = STATE_DIM_BLIND - 3):
  """回傳擴充後的 [normalizer, policy, value]。新輸入插在 state 和 privileged_state 的第 at 個位置。"""
  norm, policy, value = params[0], params[1], params[2]
  c = norm.count
  count = float(c.hi) * 2.0 ** 32 + float(c.lo) if hasattr(c, "hi") else float(np.asarray(c))
  new = {}
  for field, block in (("mean", mean), ("std", std), ("summed_variance", std ** 2 * count)):
    new[field] = {key: _insert(v, at, np.asarray(block)) for key, v in getattr(norm, field).items()}
  norm = norm.replace(**new)

  def widen(net):
    net = _to_np(net)   # 複製一份 numpy，不改到原本的參數
    first = net["params"]["hidden_0"]
    first["kernel"] = _insert(first["kernel"], at, np.zeros((k, first["kernel"].shape[1]), np.float32))
    return net

  return [norm, widen(policy), widen(value)]


def _to_np(tree):
  if isinstance(tree, dict):
    return {k: _to_np(v) for k, v in tree.items()}
  return np.array(tree, dtype=np.float32, copy=True)


def expand_checkpoint(src: Path, out: Path, k: int, mean: np.ndarray, std: np.ndarray) -> Path:
  from brax.training import checkpoint as brax_ckpt
  from flax.training import orbax_utils
  from orbax import checkpoint as ocp

  cfg = json.loads((src / _CKPT_CONFIG).read_text())
  sizes = cfg["observation_size"]
  shape = lambda key: sizes[key]["shape"] if isinstance(sizes[key], dict) else sizes[key]   # 兩種存法都可能
  dim = shape("state")[0]
  if dim != STATE_DIM_BLIND:
    raise ValueError(f"這個 checkpoint 的 state 長度是 {dim}，不是盲走版的 {STATE_DIM_BLIND}（可能已經擴充過了）。")
  params = expand_params(brax_ckpt.load(src), k, mean, std)
  dst = out / "checkpoints" / src.name
  if dst.exists():
    shutil.rmtree(dst)
  dst.parent.mkdir(parents=True, exist_ok=True)
  ocp.PyTreeCheckpointer().save(dst, params, force=True, save_args=orbax_utils.save_args_from_target(params))
  for key in ("state", "privileged_state"):
    shape(key)[0] += k
  (dst / _CKPT_CONFIG).write_text(json.dumps(cfg, ensure_ascii=False))
  (out / "expanded_from.json").write_text(json.dumps({"source": str(src), "added_inputs": k}, ensure_ascii=False))
  return dst


def main(argv=None) -> None:
  p = argparse.ArgumentParser(description="把盲走的樓梯斜坡策略擴充成有地形掃描的版本")
  p.add_argument("run", help="來源 run 資料夾或 checkpoint，例如 runs/terrain/20261115-204812-stage2")
  p.add_argument("--out", required=True, help="輸出資料夾（不要放在 runs/ 底下），例如 runs_init/scan-from-stage2")
  args = p.parse_args(argv)

  from robodog import envs
  from robodog.envs import terrain as T

  src = resolve_checkpoint(args.run)
  print(f"[expand] 來源：{src}")
  env = envs.make_env("terrain", {"impl": "jax", "terrain": {"height_scan": True}}).unwrapped
  mean, std = scan_stats(env)
  print(f"[expand] 掃描值統計：平均 {mean.mean():+.3f} m，標準差 {std.mean():.3f} m（{T.N_SCAN} 個點）")
  dst = expand_checkpoint(src, Path(args.out), T.N_SCAN, mean, std)
  print(f"[expand] 完成：{dst}")
  print(f"[expand] 接著練：python -m robodog.train configs/terrain.yaml --resume {args.out} --set env.terrain.height_scan=true …")


if __name__ == "__main__":
  main()
