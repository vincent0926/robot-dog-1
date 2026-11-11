"""診斷跌倒：跌倒率依指令大小分組，並分別在「有推擠 / 沒推擠」下量。

用法：
  python -m robodog.falls runs/rough/latest
  JAX_PLATFORMS=cpu python -m robodog.falls runs/rough/latest --episodes 128   # 訓練還在跑時用 CPU

看輸出可以回答：
  - 是被推倒的，還是自己走倒的？（有推擠 vs 沒推擠）
  - 哪種指令最容易跌倒？（前進速度大、轉向小…）
  - 跌倒大多發生在回合的什麼時候？（一開始就倒 = 起步問題；很晚才倒 = 累積的問題）
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def bin_rates(fell: np.ndarray, x: np.ndarray, edges: list[float]) -> list[dict]:
  """把 x 依 edges 分組，每組算跌倒率。回傳 [{'range': (a, b), 'n': 個數, 'rate': 跌倒率}]。"""
  out = []
  for a, b in zip(edges, edges[1:]):
    m = (x >= a) & (x < b) if b != edges[-1] else (x >= a) & (x <= b)
    out.append({"range": (a, b), "n": int(m.sum()), "rate": float(fell[m].mean()) if m.any() else float("nan")})
  return out


def measure(run: str, episodes: int, push: bool, impl: str | None, seed: int, seconds: float = 20.0) -> dict:
  import jax
  import jax.numpy as jp

  from robodog import evaluate as ev
  from robodog.policy_io import load_run

  env_sets = None if push else {"pert_config": {"enable": False}}
  _, env, policy, ckpt = load_run(run, impl=impl, env_sets=env_sets)
  n = episodes
  cmd_max = jp.array(env._config.command_config.a)
  cmd = jax.random.uniform(jax.random.PRNGKey(seed), (n, 3), minval=-cmd_max, maxval=cmd_max)

  def prepare(state):
    state.info["command"] = cmd
    state.info["steps_until_next_cmd"] = jp.full((n,), 10**8, dtype=jp.int32)
    obs = dict(state.obs)
    obs["state"] = obs["state"].at[:, -3:].set(cmd)
    return state.replace(obs=obs)

  steps = int(seconds / env.dt)
  _, alive, _, hist = ev._rollout(env, policy, n, steps, jax.random.PRNGKey(seed + 1), prepare,
                                  lambda s, ns, t: jp.zeros(n))
  fell = 1.0 - np.asarray(alive, dtype=float)
  hist = np.asarray(hist)
  c = np.asarray(cmd)
  t_fall = np.where(fell == 1, np.argmin(hist, axis=0) * env.dt, np.nan)
  speed = np.linalg.norm(c[:, :2], axis=1)
  return {
      "checkpoint": str(ckpt), "push": push, "episodes": n, "fall_rate": float(fell.mean()),
      "fall_time_mean_s": float(np.nanmean(t_fall)) if fell.any() else None,
      "fall_time_hist": np.histogram(t_fall[fell == 1], bins=[0, 2, 5, 10, 15, 20])[0].tolist() if fell.any() else [0] * 5,
      "by_vx": bin_rates(fell, np.abs(c[:, 0]), [0, 0.5, 1.0, 1.5]),
      "by_vy": bin_rates(fell, np.abs(c[:, 1]), [0, 0.4, 0.8]),
      "by_wz": bin_rates(fell, np.abs(c[:, 2]), [0, 0.6, 1.2]),
      "by_speed": bin_rates(fell, speed, [0, 0.5, 1.0, 1.5, 1.7]),
  }


def show(r: dict) -> None:
  print(f"\n=== {'有推擠' if r['push'] else '沒推擠'}：跌倒率 {r['fall_rate'] * 100:.1f}%（{r['episodes']} 回合）===")
  if r["fall_time_mean_s"] is not None:
    h = r["fall_time_hist"]
    print(f"跌倒平均發生在第 {r['fall_time_mean_s']:.1f} 秒；時間分布 0-2s:{h[0]} 2-5s:{h[1]} 5-10s:{h[2]} 10-15s:{h[3]} 15-20s:{h[4]}")
  for key, label in [("by_vx", "|前後速度|"), ("by_vy", "|側移速度|"), ("by_wz", "|轉向速度|"), ("by_speed", "平面總速度")]:
    cells = "   ".join(f"[{b['range'][0]:.1f},{b['range'][1]:.1f}) {b['rate'] * 100:4.1f}% (n={b['n']})" for b in r[key])
    print(f"{label:8s} {cells}")


def main(argv=None) -> None:
  p = argparse.ArgumentParser(description="診斷跌倒")
  p.add_argument("run", help="run 資料夾，例如 runs/rough/latest")
  p.add_argument("--episodes", type=int, default=512)
  p.add_argument("--impl", default=None, help="物理引擎：預設用訓練時的設定（GPU 訓練的是 warp）；CPU 請用 jax")
  p.add_argument("--seed", type=int, default=7)
  p.add_argument("--only", choices=["push", "nopush"], default=None, help="只量其中一種（預設兩種都量）")
  args = p.parse_args(argv)

  results = []
  for push in ([True, False] if args.only is None else [args.only == "push"]):
    r = measure(args.run, args.episodes, push, args.impl, args.seed)
    show(r)
    results.append(r)
  from pathlib import Path

  out = Path(results[0]["checkpoint"]).parent.parent / "falls.json"
  out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
  print(f"\n結果已存到 {out}")


if __name__ == "__main__":
  main()
