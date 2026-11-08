"""基本測試：pytest -q

這些測試用 CPU（impl=jax）就能跑，不需要顯示卡，也不需要先訓練。
"""

import jax
import jax.numpy as jp
import numpy as np
import pytest

from robodog import config as rcfg
from robodog import envs
from robodog.envs import navigate
from robodog.policy_io import resolve_checkpoint


def _zero_policy(action_size):
  return lambda obs, rng: (jp.zeros(obs["state"].shape[:-1] + (action_size,)), {})


# ---------- 設定檔 ----------

def test_deep_merge_and_set():
  base = {"ppo": {"num_envs": 4096, "network": {"a": 1}}, "task": "walk"}
  merged = rcfg.deep_merge(base, {"ppo": {"num_envs": 8}})
  assert merged["ppo"] == {"num_envs": 8, "network": {"a": 1}}
  assert base["ppo"]["num_envs"] == 4096  # 原本的不能被改到
  out = rcfg.apply_sets(merged, ["ppo.network.a=2", "env.impl=jax"])
  assert out["ppo"]["network"]["a"] == 2 and out["env"]["impl"] == "jax"


def test_all_configs_load_and_match_env_fields():
  for name in ("walk", "getup", "navigate", "rough"):
    cfg = rcfg.load_config(f"configs/{name}.yaml")
    env_cfg = envs.default_env_config(cfg["task"])
    env_over = {k: v for k, v in cfg["env"].items() if k != "fall_penalty"}  # fall_penalty 是我們自己加的欄位
    rcfg.update_config_dict(env_cfg, env_over)  # 欄位打錯字會在這裡報錯
    n, b, m = cfg["ppo"]["num_envs"], cfg["ppo"]["batch_size"], cfg["ppo"]["num_minibatches"]
    assert (b * m) % n == 0


def test_typo_in_env_config_is_rejected():
  env_cfg = envs.default_env_config("walk")
  with pytest.raises(KeyError):
    rcfg.update_config_dict(env_cfg, {"reward_config": {"scales": {"tracking_lin_vell": 1.0}}})


# ---------- 雷射 ----------

def test_ray_hits_obstacle_and_wall():
  centers = jp.array([[2.0, 0.0], [100.0, 100.0]])
  d = navigate.ray_distances(jp.zeros(2), jp.array(0.0), centers, 0.4, 5.0, n_rays=4, max_range=10.0)
  # 方向 0°, 90°, 180°, 270°
  np.testing.assert_allclose(np.asarray(d), [1.6, 5.0, 5.0, 5.0], atol=1e-5)


def test_ray_respects_yaw_and_max_range():
  centers = jp.array([[0.0, 2.0]])
  d = navigate.ray_distances(jp.zeros(2), jp.array(np.pi / 2), centers, 0.4, 5.0, n_rays=4, max_range=3.0)
  np.testing.assert_allclose(np.asarray(d), [1.6, 3.0, 3.0, 3.0], atol=1e-5)


# ---------- 導航環境 ----------

@pytest.fixture(scope="module")
def nav_env():
  return envs.make_env(
      "navigate", {"impl": "jax", "episode_length": 20}, low_level_policy_fn=_zero_policy(12)
  )


def test_navigate_layout_and_step(nav_env):
  cfg = nav_env._config
  state = jax.jit(nav_env.reset)(jax.random.PRNGKey(3))
  start = np.asarray(state.data.qpos[:2])
  goal = np.asarray(state.info["goal"])
  obstacles = np.asarray(state.info["obstacles"])
  active = obstacles[obstacles[:, 0] < 50]
  assert len(active) <= cfg.n_obstacles
  keep_out = cfg.obstacle_radius + cfg.robot_radius
  assert np.all(np.linalg.norm(active - start, axis=1) > keep_out)
  assert np.all(np.linalg.norm(active - goal, axis=1) > keep_out)
  assert np.all(np.abs(goal) <= cfg.arena_half_size)

  obs_size = 2 + 1 + 3 + 3 + 3 + cfg.n_rays + 3
  assert state.obs["state"].shape == (obs_size,)

  step = jax.jit(nav_env.step)
  state = step(state, jp.array([1.0, 0.0, 0.0]))
  assert np.isfinite(float(state.reward))
  # 走了 0.1 秒（5 個低階步）
  assert int(state.info["ll"]["info"]["last_act"].shape[0]) == 12


# ---------- checkpoint 路徑 ----------

def test_resolve_checkpoint(tmp_path):
  for run in ("20260101-000000", "20260102-000000"):
    for step in ("000000001000", "000000002000"):
      d = tmp_path / "walk" / run / "checkpoints" / step
      d.mkdir(parents=True)
      (d / "ppo_network_config.json").write_text("{}")
  latest = resolve_checkpoint(tmp_path / "walk" / "latest")
  assert latest.parts[-3:] == ("20260102-000000", "checkpoints", "000000002000")
  run = resolve_checkpoint(tmp_path / "walk" / "20260101-000000")
  assert run.name == "000000002000"


# ---------- 播放（錄影流程，不實際算圖） ----------

def test_play_record_video(tmp_path, monkeypatch):
  from robodog import play

  env = envs.make_env("walk", {"impl": "jax"})
  monkeypatch.setattr(env, "render", lambda traj, **kw: [np.zeros((16, 16, 3), np.uint8)] * len(traj))
  out = tmp_path / "walk.mp4"
  play.record_video(env, _zero_policy(12), "walk", out.as_posix(), seconds=0.2, seed=0, width=16, height=16)
  assert out.exists() and out.stat().st_size > 0


def test_fall_penalty_wrapper():
  import jax
  import jax.numpy as jp

  env = envs.make_env("walk", {"impl": "jax", "fall_penalty": 3.0})
  assert isinstance(env, envs.FallPenalty) and env._config.episode_length == 1000  # 屬性會轉給裡面的環境
  state = env.reset(jax.random.PRNGKey(0))
  state = state.replace(done=jp.float32(1.0))
  inner = env.env.step(state, jp.zeros(env.action_size))
  outer = env.step(state, jp.zeros(env.action_size))
  assert float(outer.reward) == pytest.approx(float(inner.reward) - 3.0 * float(inner.done))
