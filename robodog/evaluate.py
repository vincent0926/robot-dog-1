"""評估訓練好的策略，對照 PRD 的驗收標準。

用法：
  python -m robodog.evaluate runs/walk/latest
  python -m robodog.evaluate runs/getup/latest --episodes 512
  python -m robodog.evaluate runs/navigate/latest

各任務的指標：
  walk     固定一個隨機速度指令跑 20 秒：平均速度誤差（目標 < 0.15 m/s）、跌倒率（目標 < 5%）
  getup    從跌倒狀態開始 5 秒後：站穩的比例（目標 ≥ 85%）
  navigate 一回合內：抵達率（目標 ≥ 80%）、碰撞率、跌倒率、超時率
"""

from __future__ import annotations

import argparse
import json
import time

TARGETS = {
    "walk": {"vel_error_mps": ("<", 0.15), "fall_rate": ("<", 0.05)},
    "rough": {"vel_error_mps": ("<", 0.20), "fall_rate": ("<", 0.05)},  # 地形不平，速度誤差放寬一點
    "getup": {"success_rate": (">=", 0.85)},
    "navigate": {"reach_rate": (">=", 0.80)},
}


def _rollout(env, policy, n_envs, n_steps, rng, prepare_fn, record_fn):
  """同時跑 n_envs 個環境 n_steps 步（不自動重置），每步呼叫 record_fn 記錄資料。"""
  import jax
  import jax.numpy as jp

  def body(carry, t):
    state, alive, rng = carry
    rng, key = jax.random.split(rng)
    action, _ = policy(state.obs, key)
    nstate = jax.vmap(env.step)(state, action)
    rec = record_fn(state, nstate, t)
    new_alive = alive & (nstate.done == 0)
    return (nstate, new_alive, rng), (rec, alive)

  @jax.jit
  def run(rng):
    rng, reset_rng = jax.random.split(rng)
    state = jax.vmap(env.reset)(jax.random.split(reset_rng, n_envs))
    state = prepare_fn(state)
    alive0 = jp.ones(n_envs, dtype=bool)
    (state, alive, _), (recs, alive_hist) = jax.lax.scan(
        body, (state, alive0, rng), jp.arange(n_steps)
    )
    return state, alive, recs, alive_hist

  return run(rng)


def eval_walk(env, policy, n_envs, rng, seconds=20.0, warmup=1.0):
  import jax
  import jax.numpy as jp

  n_steps = int(seconds / env.dt)
  cmd_max = jp.array(env._config.command_config.a)

  def prepare(state):
    # 每個環境整回合固定一個指令（避免換指令瞬間的誤差干擾統計）
    key = jax.random.PRNGKey(123)
    cmd = jax.random.uniform(key, (n_envs, 3), minval=-cmd_max, maxval=cmd_max)
    state.info["command"] = cmd
    state.info["steps_until_next_cmd"] = jp.full((n_envs,), 10**8, dtype=jp.int32)
    obs = dict(state.obs)
    obs["state"] = obs["state"].at[:, -3:].set(cmd)
    return state.replace(obs=obs)

  def record(state, nstate, t):
    v = jax.vmap(env.get_local_linvel)(nstate.data)[:, :2]
    err = jp.linalg.norm(v - state.info["command"][:, :2], axis=-1)
    return err

  _, alive, errs, alive_hist = _rollout(env, policy, n_envs, n_steps, rng, prepare, record)
  mask = alive_hist & (jp.arange(n_steps)[:, None] * env.dt >= warmup)
  vel_error = float(jp.sum(errs * mask) / jp.maximum(jp.sum(mask), 1))
  return {"vel_error_mps": vel_error, "fall_rate": float(1.0 - jp.mean(alive))}


def eval_getup(env, policy, n_envs, rng, seconds=5.0):
  import jax
  import jax.numpy as jp

  n_steps = int(seconds / env.dt)

  def record(state, nstate, t):
    return jp.zeros(n_envs)

  state, _, _, _ = _rollout(env, policy, n_envs, n_steps, rng, lambda s: s, record)
  gravity = jax.vmap(env.get_gravity)(state.data)
  height = state.data.site_xpos[:, env._imu_site_id, 2]
  standing = (gravity[:, 2] < -0.95) & (height > 0.25)
  return {"success_rate": float(jp.mean(standing))}


def eval_navigate(env, policy, n_envs, rng):
  import jax.numpy as jp

  n_steps = env._config.episode_length

  def record(state, nstate, t):
    m = nstate.metrics
    return jp.stack([m["reached"], m["collided"], m["fell"]], axis=-1)

  _, alive, recs, alive_hist = _rollout(env, policy, n_envs, n_steps, rng, lambda s: s, record)
  # 只算每個環境「第一次結束」那一步的原因
  ended_now = alive_hist[..., None] & (recs > 0)
  outcome = jp.any(ended_now, axis=0).astype(jp.float32)
  return {
      "reach_rate": float(jp.mean(outcome[:, 0])),
      "collision_rate": float(jp.mean(outcome[:, 1])),
      "fall_rate": float(jp.mean(outcome[:, 2])),
      "timeout_rate": float(jp.mean(alive)),
  }


def check_targets(task: str, result: dict) -> dict:
  verdict = {}
  for key, (op, target) in TARGETS.get(task, {}).items():
    v = result[key]
    verdict[key] = (v < target) if op == "<" else (v >= target)
  return verdict


def main(argv=None) -> None:
  p = argparse.ArgumentParser(description="評估訓練好的策略")
  p.add_argument("run", help="run 資料夾或 checkpoint，例如 runs/walk/latest")
  p.add_argument("--episodes", type=int, default=256, help="同時評估幾個回合")
  p.add_argument("--impl", default=None, help="覆寫物理引擎實作：warp 或 jax")
  p.add_argument("--seed", type=int, default=1)
  p.add_argument("--json", default=None, help="結果另存的 JSON 路徑（預設存到該次訓練資料夾的 eval.json）")
  args = p.parse_args(argv)

  import jax
  from robodog.policy_io import load_run

  cfg, env, policy, ckpt = load_run(args.run, impl=args.impl)
  task = cfg["task"]
  rng = jax.random.PRNGKey(args.seed)
  fn = {"walk": eval_walk, "rough": eval_walk, "getup": eval_getup, "navigate": eval_navigate}[task]
  result = fn(env, policy, args.episodes, rng)
  verdict = check_targets(task, result)

  print(f"\n=== 評估結果：{task}（{args.episodes} 回合）===")
  print(f"checkpoint：{ckpt}")
  for k, v in result.items():
    mark = ""
    if k in verdict:
      op, target = TARGETS[task][k]
      mark = f"   目標 {op} {target}  {'✅ 通過' if verdict[k] else '❌ 未達標'}"
    print(f"  {k:16s} {v:8.4f}{mark}")

  out = args.json or (ckpt.parent.parent / "eval.json")
  with open(out, "w", encoding="utf-8") as f:
    json.dump({"task": task, "checkpoint": str(ckpt), "step": int(ckpt.name),
               "episodes": args.episodes, **result,
               "targets": {k: list(v) for k, v in TARGETS.get(task, {}).items()},
               "verdict": verdict, "passed": all(verdict.values()), "time": time.time()},
              f, ensure_ascii=False, indent=2)
  print(f"結果已存到 {out}")


if __name__ == "__main__":
  main()
