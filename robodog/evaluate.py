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
    "terrain": {"success_rate": (">=", 0.80)},  # 樓梯、斜坡、方塊（難度 0~5）穿過區塊的比例
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


def terrain_half_size(env):
  """有高度場（崎嶇地形）的環境回傳地形的半邊長（公尺），沒有就回傳 None。"""
  m = env.mj_model
  return float(m.hfield_size[0][0]) if m.nhfield else None


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
    return jp.concatenate([err[:, None], nstate.data.qpos[:, :2]], axis=1)  # 速度誤差、機身 x、y

  _, alive, recs, alive_hist = _rollout(env, policy, n_envs, n_steps, rng, prepare, record)
  errs, xy = recs[..., 0], recs[..., 1:]
  mask = alive_hist & (jp.arange(n_steps)[:, None] * env.dt >= warmup)
  vel_error = float(jp.sum(errs * mask) / jp.maximum(jp.sum(mask), 1))
  fell = ~alive
  out = {"vel_error_mps": vel_error, "fall_rate": float(jp.mean(fell))}
  bound = terrain_half_size(env)
  if bound is not None:
    # 崎嶇地形只有 20m x 20m。整回合固定一個指令，直走 20 秒會走出地圖邊緣然後掉下去，那不是走路不穩。
    # 所以倒在邊緣附近（離邊界 0.5m 內）的另外算，不計入 fall_rate。
    last = jp.clip(jp.sum(alive_hist, axis=0) - 1, 0, n_steps - 1)
    pos = xy[last, jp.arange(n_envs)]
    at_edge = fell & (jp.max(jp.abs(pos), axis=-1) >= bound - 0.5)
    out["fall_rate"] = float(jp.mean(fell & ~at_edge))
    out["edge_exit_rate"] = float(jp.mean(at_edge))
  return out


def eval_terrain(env, policy, n_envs, rng, seconds=8.0, vx=0.6, headline_levels=6):
  """樓梯與斜坡：每種地形 x 每個難度各放一批機器狗，用 vx 往前走，看有多少能穿過這個 4m 區塊。

  成功 ＝ 在倒下之前離開了這個區塊（離區塊中心超過 2m，Chebyshev 距離）；失敗 ＝ 還沒離開就倒了。
  離開之後發生的事不算：角落的區塊往外走會走出地圖，旁邊可能是更難的區塊，那些跟這個區塊的難度無關。
  success_rate ＝ 斜坡（上/下）、樓梯（上/下）、方塊這五種裡「最弱的那一種」，在難度 0 ~ headline_levels-1 的平均通過率。
  取最弱而不是平均，因為平均會被下坡、下樓梯這類簡單的項目墊高，掩蓋上樓梯過不去（基準線就是這樣）。
  success_mean 是五種的平均；各難度的細節放在 success_by。
  """
  import jax
  import jax.numpy as jp
  import numpy as np

  from robodog.envs import terrain as T

  env = env.unwrapped  # 評估不需要獎勵的包裝（步態、跌倒懲罰）；而且 reset_at 在最裡面的環境
  grid = [(lv, ty) for lv in range(T.N_LEVELS) for ty in range(T.N_TYPES)]
  n_per = max(n_envs // len(grid), 2)
  total = len(grid) * n_per
  lv = jp.repeat(jp.array([g[0] for g in grid]), n_per)
  ty = jp.repeat(jp.array([g[1] for g in grid]), n_per)
  cmd = jp.tile(jp.array([[vx, 0.0, 0.0]]), (total, 1))
  centers = jp.asarray(np.array([T.tile_center(int(a), int(b)) for a, b in grid]), dtype=jp.float32)
  center_xy = jp.repeat(centers, n_per, axis=0)
  n_steps = int(seconds / env.dt)

  def body(carry, _):
    state, alive, exited, rng = carry
    rng, key = jax.random.split(rng)
    action, _ = policy(state.obs, key)
    nstate = jax.vmap(env.step)(state, action)
    alive = alive & (nstate.done == 0)
    cheb = jp.max(jp.abs(nstate.data.qpos[:, :2] - center_xy), axis=-1)
    exited = exited | (alive & (cheb >= T.TILE / 2))
    return (nstate, alive, exited, rng), None

  @jax.jit
  def run(rng):
    rng, kr = jax.random.split(rng)
    state = jax.vmap(env.reset_at)(jax.random.split(kr, total), lv, ty)
    state.info["command"] = cmd
    state.info["steps_until_next_cmd"] = jp.full((total,), 10**8, dtype=jp.int32)
    obs = dict(state.obs)
    obs["state"] = obs["state"].at[:, -3:].set(cmd)
    state = state.replace(obs=obs)
    init = (state, jp.ones(total, dtype=bool), jp.zeros(total, dtype=bool), rng)
    (_, alive, exited, _), _ = jax.lax.scan(body, init, None, length=n_steps)
    return alive, exited

  alive, exited = run(rng)
  exited, alive = np.asarray(exited), np.asarray(alive)
  ok = exited.astype(float).reshape(T.N_LEVELS, T.N_TYPES, n_per).mean(-1)   # [難度, 地形]
  fell = (~exited & ~alive).astype(float).reshape(T.N_LEVELS, T.N_TYPES, n_per).mean(-1)   # 還沒離開就倒
  hard = [i for i, name in enumerate(T.TYPES) if name != "flat"]
  hl = min(headline_levels, T.N_LEVELS)
  return {
      "success_rate": float(ok[:hl][:, hard].mean(0).min()),
      "success_mean": float(ok[:hl][:, hard].mean()),
      "weakest": T.TYPES[hard[int(ok[:hl][:, hard].mean(0).argmin())]],
      "fall_rate": float(fell[:hl][:, hard].mean()),
      "success_by": {name: [round(float(x), 3) for x in ok[:, i]] for i, name in enumerate(T.TYPES)},
      "episodes_per_cell": n_per,
  }


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
  p.add_argument("--on", default=None, metavar="任務", help="拿這個策略去別的任務的環境上量（例如 --on terrain，看走路策略在樓梯斜坡上的基準線）")
  p.add_argument("--no-push", action="store_true",
                 help="評估時關掉隨機推擠（rough 任務用來區分「地形造成的跌倒」和「被推倒」），結果存到 eval_nopush.json")
  p.add_argument("--json", default=None, help="結果另存的 JSON 路徑（預設存到該次訓練資料夾的 eval.json）")
  args = p.parse_args(argv)

  import jax
  from robodog.policy_io import load_run

  env_sets = {"pert_config": {"enable": False}} if args.no_push else None
  cfg, env, policy, ckpt = load_run(args.run, impl=args.impl, env_sets=env_sets, task=args.on)
  task = cfg["task"]
  rng = jax.random.PRNGKey(args.seed)
  fn = {"walk": eval_walk, "rough": eval_walk, "terrain": eval_terrain, "getup": eval_getup, "navigate": eval_navigate}[task]
  result = fn(env, policy, args.episodes, rng)
  verdict = check_targets(task, result)

  print(f"\n=== 評估結果：{task}（{args.episodes} 回合{'，不推擠' if args.no_push else ''}）===")
  print(f"checkpoint：{ckpt}")
  for k, v in result.items():
    if not isinstance(v, (int, float)):
      if k == "weakest":
        print(f"  最弱的一項       {v}")
      continue  # 表格類的結果（success_by）在下面另外印
    mark = ""
    if k in verdict:
      op, target = TARGETS[task][k]
      mark = f"   目標 {op} {target}  {'✅ 通過' if verdict[k] else '❌ 未達標'}"
    print(f"  {k:16s} {v:8.4f}{mark}")

  if "success_by" in result:
    print("\n各難度的通過率（難度 0 最簡單；樓梯每階 2cm → 12cm，斜坡 3° → 20°）：")
    print(f"  {'地形':12s}" + "".join(f"  L{i}  " for i in range(len(next(iter(result['success_by'].values()))))))
    for name, row in result["success_by"].items():
      print(f"  {name:12s}" + "".join(f" {x * 100:4.0f}% " for x in row))

  suffix = "_nopush" if args.no_push else ""
  suffix = f"_{args.on}" if args.on else suffix
  out = args.json or (ckpt.parent.parent / f"eval{suffix}.json")
  with open(out, "w", encoding="utf-8") as f:
    json.dump({"task": task, "checkpoint": str(ckpt), "step": int(ckpt.name), "policy_task": cfg["task"] if not args.on else "（別的任務的策略）",
               "episodes": args.episodes, "push": not args.no_push, **result,
               "targets": {k: list(v) for k, v in TARGETS.get(task, {}).items()},
               "verdict": verdict, "passed": all(verdict.values()), "time": time.time()},
              f, ensure_ascii=False, indent=2)
  print(f"結果已存到 {out}")


if __name__ == "__main__":
  main()
