"""量走路姿勢（步態），驗收「像不像小跑步」。

用法：
  python -m robodog.gait runs/walk/latest                 # 指令 1 m/s
  python -m robodog.gait runs/walk/latest --vx 0.5

注意：訓練用的 GPU 還在跑的時候，請用 CPU 量，不要搶顯示卡：
  JAX_PLATFORMS=cpu python -m robodog.gait runs/walk/latest

四隻腳的順序是 FR（右前）、FL（左前）、RR（右後）、RL（左後）。
小跑步（trot）＝對角的兩隻腳（FR＋RL、FL＋RR）同時著地／離地，兩組輪流。
"""

from __future__ import annotations

import argparse
import json

import numpy as np

NAMES = ["FR", "FL", "RR", "RL"]


def analyze(contact: np.ndarray, foot_z: np.ndarray, dt: float) -> dict:
  """contact: (步數, 4) 每隻腳有沒有著地（0/1）；foot_z: (步數, 4) 腳的高度（公尺）。"""
  c = np.asarray(contact, dtype=float)
  z = np.asarray(foot_z, dtype=float)

  def same(i, j):
    return float((c[:, i] == c[:, j]).mean())

  touchdowns = [int(((c[1:, i] == 1) & (c[:-1, i] == 0)).sum()) for i in range(4)]
  duty = [float(c[:, i].mean()) for i in range(4)]
  # 兩組對角腳各自的著地比例，差越多越「跛」：0 代表左右對稱
  duty_a, duty_b = (duty[0] + duty[3]) / 2, (duty[1] + duty[2]) / 2
  return {
      "duty": dict(zip(NAMES, duty)),
      "duty_gap": abs(duty_a - duty_b),
      "foot_lift_cm": {n: float((z[:, i].max() - z[:, i].min()) * 100) for i, n in enumerate(NAMES)},
      "diag_sync": {"FR-RL": same(0, 3), "FL-RR": same(1, 2)},
      "pair_opposite": {"FR-FL": 1 - same(0, 1), "RR-RL": 1 - same(2, 3)},
      "step_hz": {n: touchdowns[i] / (len(c) * dt) for i, n in enumerate(NAMES)},
      "all_air": float((c.sum(1) == 0).mean()),
      "all_ground": float((c.sum(1) == 4).mean()),
  }


def verdict(g: dict) -> dict:
  """小跑步的判斷標準（寬鬆）：對角同步、兩組輪流、不要用跳的、左右大致對稱。"""
  return {
      "對角腳同步 > 0.8": min(g["diag_sync"].values()) > 0.8,
      "兩組輪流 > 0.8": min(g["pair_opposite"].values()) > 0.8,
      "四腳同時騰空 < 5%": g["all_air"] < 0.05,
      "每秒步數在 1.5～3.5": all(1.5 <= v <= 3.5 for v in g["step_hz"].values()),
      "兩組著地比例差 < 15%": g["duty_gap"] < 0.15,
  }


def measure(run: str, vx: float = 1.0, seconds: float = 8.0, impl: str = "jax", skip: float = 2.0, seed: int = 0):
  import jax
  import jax.numpy as jp

  from robodog.policy_io import load_run

  cfg, env, policy, ckpt = load_run(run, impl=impl, env_sets={"pert_config": {"enable": False}})
  u = env.unwrapped
  cmd = jp.array([vx, 0.0, 0.0])
  step, reset, pol = jax.jit(env.step), jax.jit(env.reset), jax.jit(policy)
  st = reset(jax.random.PRNGKey(seed))
  rng = jax.random.PRNGKey(seed + 1)
  contact, z, height, speed = [], [], [], []
  n, n_skip = int(seconds / env.dt), int(skip / env.dt)
  for i in range(n):
    st.info["command"] = cmd
    st.info["steps_until_next_cmd"] = jp.int32(10**8)
    obs = dict(st.obs)
    obs["state"] = obs["state"].at[-3:].set(cmd)
    st = st.replace(obs=obs)
    rng, k = jax.random.split(rng)
    action, _ = pol(st.obs, k)
    st = step(st, action)
    if i >= n_skip:  # 前幾秒在起步，不算
      contact.append(np.asarray(st.info["last_contact"]))
      z.append(np.asarray(st.data.site_xpos[u._feet_site_id][:, 2]))
      height.append(float(st.data.qpos[2]))
      speed.append(float(u.get_local_linvel(st.data)[0]))
  return ckpt, analyze(np.array(contact), np.array(z), env.dt), {
      "cmd_vx": vx, "speed": float(np.mean(speed)), "height_m": float(np.mean(height)),
      "contact": np.array(contact),
  }


def main(argv=None) -> None:
  p = argparse.ArgumentParser(description="量走路姿勢（步態）")
  p.add_argument("run", help="run 資料夾，例如 runs/walk/latest")
  p.add_argument("--vx", type=float, default=1.0, help="前進速度指令（m/s）")
  p.add_argument("--seconds", type=float, default=8.0)
  p.add_argument("--impl", default="jax", help="物理引擎：jax（預設，CPU 也能跑）或 warp")
  args = p.parse_args(argv)

  ckpt, g, extra = measure(args.run, args.vx, args.seconds, args.impl)
  c = extra.pop("contact")
  print(f"\n=== 步態（{ckpt.parent.parent.name}，指令 {args.vx} m/s）===")
  print(f"實際速度 {extra['speed']:.2f} m/s   機身高度 {extra['height_m']:.3f} m")
  print("著地比例   " + "  ".join(f"{n} {v:.2f}" for n, v in g["duty"].items()) + f"   兩組差 {g['duty_gap']:.2f}")
  print("抬腳高度cm " + "  ".join(f"{n} {v:.1f}" for n, v in g["foot_lift_cm"].items()))
  print("對角同步   " + "  ".join(f"{k} {v:.2f}" for k, v in g["diag_sync"].items()) + "   （接近 1 是小跑）")
  print("兩組輪流   " + "  ".join(f"{k} {v:.2f}" for k, v in g["pair_opposite"].items()) + "   （接近 1 是小跑）")
  print("步頻 Hz    " + "  ".join(f"{n} {v:.1f}" for n, v in g["step_hz"].items()))
  print(f"四腳同時騰空 {g['all_air']:.2f}   四腳同時著地 {g['all_ground']:.2f}")
  print("\n時序（每格 0.02 秒，# 是著地）：")
  for i, n in enumerate(["FR", "FL", "RR", "RL"]):
    print(n, "".join("#" if x else "." for x in c[:100, i]))
  v = verdict(g)
  print("\n驗收：")
  for k, ok in v.items():
    print(f"  {'✅' if ok else '❌'} {k}")
  out = {"run": str(ckpt), "vx": args.vx, **extra, **g, "verdict": v, "passed": all(v.values())}
  path = ckpt.parent.parent / "gait.json"
  path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
  print(f"結果已存到 {path}")


if __name__ == "__main__":
  main()
