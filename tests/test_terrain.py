"""樓梯與斜坡地形：地形產生器、和 MuJoCo 的對應、出生位置。"""

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest

from robodog import envs
from robodog.envs import terrain as T


@pytest.fixture(scope="module")
def cfg():
  return T.default_config()


@pytest.fixture(scope="module")
def built(cfg):
  return T.build_heightmap(cfg)


def test_heightmap_ranges_and_tile_edges_match(built):
  H, centers = built
  assert H.shape == (T.N, T.N) and H.min() >= 0 and H.max() <= T.ZMAX
  # 每個區塊的最外圍一圈高度都是 BASE_H（除了方塊和平地的微凹凸本來就不碰邊緣）：從一個區塊走到另一個不會掉下懸崖
  xs = np.linspace(-T.R, T.R, T.N)
  for level in range(T.N_LEVELS):
    for typ in range(T.N_TYPES):
      cx, cy = T.tile_center(level, typ)
      ix = int(np.argmin(np.abs(xs - (cx + T.TILE / 2 - 0.07))))   # 區塊東邊緣內側一格
      iy = int(np.argmin(np.abs(xs - cy)))
      edge = H[ix, iy]
      if typ in (1, 2, 3, 4):
        assert abs(edge - T.BASE_H) < 0.13, (level, T.TYPES[typ], edge)  # 樓梯最外一階最多差一階高度


def test_platform_heights_and_directions(cfg, built):
  H, centers = built
  # 上坡/上樓梯：中央低，往外變高；下坡/下樓梯相反
  for level in (2, 5):
    p = T.level_params(cfg, level)
    assert centers[level, 1, 2] == pytest.approx(T.BASE_H - np.tan(p["slope"]) * 1.25)
    assert centers[level, 2, 2] == pytest.approx(T.BASE_H + np.tan(p["slope"]) * 1.25)
    assert centers[level, 3, 2] == pytest.approx(T.BASE_H - 5 * p["step"])
    assert centers[level, 4, 2] == pytest.approx(T.BASE_H + 5 * p["step"])
  # 難度越高，起伏越大
  assert centers[7, 4, 2] > centers[0, 4, 2] and centers[7, 3, 2] < centers[0, 3, 2]
  # 樓梯高度：第 0 級 2cm、第 7 級 11.8cm
  assert T.level_params(cfg, 0)["step"] == pytest.approx(0.02) and T.level_params(cfg, 7)["step"] == pytest.approx(0.118)


def test_height_fn_matches_numpy_and_clips(built):
  H, _ = built
  f = T.make_height_fn(H)
  xs = np.linspace(-T.R, T.R, T.N)
  for ix, iy in [(10, 20), (200, 300), (447, 0)]:   # 格點上要完全一樣
    assert float(f(jp.array([xs[ix], xs[iy]], dtype=jp.float32))) == pytest.approx(H[ix, iy], abs=1e-4)
  assert np.isfinite(float(f(jp.array([100.0, -100.0]))))   # 地圖外不會爆掉（夾到邊界）


@pytest.fixture(scope="module")
def env():
  return envs.make_env("terrain", {"impl": "jax"})


def test_hfield_orientation_matches_mujoco(env):
  """射線從上往下打，量到的高度要和我的高度函式一致：x、y 沒有顛倒。"""
  m = env.mj_model
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  rng = np.random.default_rng(0)
  for level, typ in [(3, 3), (5, 4), (4, 1), (6, 2)]:
    cx, cy = T.tile_center(level, typ)
    for _ in range(10):
      x, y = cx + rng.uniform(-1.9, 1.9), cy + rng.uniform(-1.9, 1.9)
      dist = mujoco.mj_ray(m, d, np.array([x, y, 3.0]), np.array([0, 0, -1.0]), None, 1, -1, np.zeros(1, dtype=np.int32))
      assert 3.0 - dist == pytest.approx(float(env.terrain_height(np.array([x, y], dtype=np.float32))), abs=0.01)


def test_reset_spawns_on_platform_and_respects_max_level():
  e = envs.make_env("terrain", {"impl": "jax", "terrain": {"max_level": 2}})
  reset = jax.jit(jax.vmap(e.reset))
  s = reset(jax.random.split(jax.random.PRNGKey(0), 64))
  xy = np.asarray(s.data.qpos[:, :2])
  z = np.asarray(s.data.qpos[:, 2])
  # 出生在某個區塊的中央平台上：離該區塊中心不到 0.75m，高度是平台高度 + 約 0.35m
  centers = np.array([[T.tile_center(lv, ty) for ty in range(T.N_TYPES)] for lv in range(T.N_LEVELS)])
  for i in range(64):
    d = np.abs(centers - xy[i]).max(-1)       # [level, type] 離各區塊中心的 Chebyshev 距離
    lv, ty = np.unravel_index(d.argmin(), d.shape)
    assert d[lv, ty] < T.PLATFORM and lv <= 2           # 沒有抽到比 max_level 更難的
    assert z[i] - e._centers_np[lv, ty, 2] == pytest.approx(0.35, abs=0.01)
  assert len({tuple(np.round(p, 0)) for p in xy}) > 10   # 不是全部擠在同一個區塊


def _spawn_types(e, n=256):
  s = jax.jit(jax.vmap(e.reset))(jax.random.split(jax.random.PRNGKey(0), n))
  xy = np.asarray(s.data.qpos[:, :2])
  centers = np.array([[T.tile_center(lv, ty) for ty in range(T.N_TYPES)] for lv in range(T.N_LEVELS)])
  return np.array([np.unravel_index(np.abs(centers - p).max(-1).argmin(), centers.shape[:2])[1] for p in xy])


def test_type_weights_oversample_and_exclude():
  stairs_up = T.TYPES.index("stairs_up")
  # 只給上樓梯 → 全部都在上樓梯出生
  only = envs.make_env("terrain", {"impl": "jax", "terrain": {"max_level": 5, "type_weights": [0, 0, 0, 1, 0, 0]}})
  assert set(_spawn_types(only)) == {stairs_up}
  # 上樓梯 3.5、其他 1 → 約 41%（256 次抽樣，給寬一點的範圍）
  more = envs.make_env("terrain", {"impl": "jax", "terrain": {"max_level": 5, "type_weights": [1, 1, 1, 3.5, 1, 1]}})
  frac = float(np.mean(_spawn_types(more) == stairs_up))
  assert 0.30 < frac < 0.53
  with pytest.raises(ValueError):
    envs.make_env("terrain", {"impl": "jax", "terrain": {"type_weights": [1, 1, 1]}})


def test_terrain_config_typo_is_rejected():
  with pytest.raises(KeyError):
    envs.make_env("terrain", {"impl": "jax", "terrain": {"max_levl": 3}})


def _curriculum_env(**terrain):
  from mujoco_playground import wrapper
  e = envs.make_env("terrain", {"impl": "jax", "terrain": {"curriculum": True, "max_level": 3, **terrain}})
  return e, wrapper.wrap_for_brax_training(e, episode_length=4)


def _tiles(e, xy):
  return np.array([np.asarray(e.unwrapped._tile_of(jp.asarray(p))) for p in xy])


def test_curriculum_moves_spawn_to_next_level():
  # curriculum_up = 0 → 每一回合都「走出區塊」→ 下一回合難度 +1，直到上限 3
  e, w = _curriculum_env(curriculum_start=0, curriculum_up=0.0)
  reset, step = jax.jit(w.reset), jax.jit(w.step)
  s = reset(jax.random.split(jax.random.PRNGKey(0), 8))
  assert np.all(_tiles(e, np.asarray(s.data.qpos[:, :2]))[:, 0] == 0)   # 從難度 0 出生
  act = jp.zeros((8, e.action_size))
  for episode in range(1, 5):
    for _ in range(4):
      s = step(s, act)
    tiles = _tiles(e, np.asarray(s.data.qpos[:, :2]))
    assert np.all(tiles[:, 0] == min(episode, 3)), (episode, tiles)  # 回合結束後在新難度的平台上出生
    z = np.asarray(s.data.qpos[:, 2])
    plat = np.array([e.unwrapped._centers_np[lv, ty, 2] for lv, ty in tiles])
    assert np.all(np.abs(z - plat - 0.35) < 0.05)                     # 高度也跟著新平台


def test_curriculum_moves_down_when_not_following_command():
  # 指令要求的距離門檻設 0、倍數設很大 → 一定算「跟不上」→ 每回合 -1，直到 0
  e, w = _curriculum_env(curriculum_start=3, curriculum_up=100.0, curriculum_down=1e6, curriculum_min_cmd=0.0)
  reset, step = jax.jit(w.reset), jax.jit(w.step)
  s = reset(jax.random.split(jax.random.PRNGKey(1), 8))
  start = _tiles(e, np.asarray(s.data.qpos[:, :2]))[:, 0]
  act = jp.zeros((8, e.action_size))
  for episode in range(1, 5):
    for _ in range(4):
      s = step(s, act)
    lv = _tiles(e, np.asarray(s.data.qpos[:, :2]))[:, 0]
    assert np.all(lv == np.maximum(start - episode, 0)), (episode, start, lv)


def test_curriculum_off_by_default_keeps_spawn():
  from mujoco_playground import wrapper
  e = envs.make_env("terrain", {"impl": "jax", "terrain": {"max_level": 3}})
  w = wrapper.wrap_for_brax_training(e, episode_length=4)
  s = jax.jit(w.reset)(jax.random.split(jax.random.PRNGKey(2), 8))
  before = _tiles(e, np.asarray(s.data.qpos[:, :2]))
  step = jax.jit(w.step)
  for _ in range(8):
    s = step(s, jp.zeros((8, e.action_size)))
  assert np.array_equal(_tiles(e, np.asarray(s.data.qpos[:, :2])), before)


# ---------- 地形掃描（height_scan）與擴充舊策略 ----------

def test_height_scan_obs_layout_and_values():
  # 雜訊關掉：加了掃描之後亂數的取用順序不同，有雜訊就沒辦法逐格比較
  blind = envs.make_env("terrain", {"impl": "jax", "noise_config": {"level": 0.0}, "terrain": {"max_level": 0}})
  seen = envs.make_env("terrain", {"impl": "jax", "noise_config": {"level": 0.0}, "terrain": {"max_level": 0, "height_scan": True}})
  k = jax.random.PRNGKey(0)
  a = jax.jit(blind.unwrapped.reset_at, static_argnums=(1, 2))(k, 0, 0)
  b = jax.jit(seen.unwrapped.reset_at, static_argnums=(1, 2))(k, 0, 0)
  sa, sb = np.asarray(a.obs["state"]), np.asarray(b.obs["state"])
  assert sb.shape[0] == sa.shape[0] + T.N_SCAN
  np.testing.assert_allclose(sb[:-3 - T.N_SCAN], sa[:-3], atol=1e-6)   # 原本的輸入位置不變
  np.testing.assert_allclose(sb[-3:], sa[-3:], atol=1e-6)               # 速度指令還在最後 3 格
  pa, pb = np.asarray(a.obs["privileged_state"]), np.asarray(b.obs["privileged_state"])
  np.testing.assert_allclose(pb[:sb.shape[0]], sb, atol=1e-6)
  np.testing.assert_allclose(pb[sb.shape[0]:], pa[sa.shape[0]:], atol=1e-6)
  # 平地中央：掃描值 ≈ 0（機身高度 ≈ 地面 + SCAN_NOMINAL）
  assert np.all(np.abs(sb[-3 - T.N_SCAN:-3]) < 0.08)

  # 上樓梯（難度 5）平台中央、面向 +x：前方 1 m 的點落在第一階上 → 地面比較高 → 掃描值是負的
  e = seen.unwrapped
  lv, ty = 5, T.TYPES.index("stairs_up")
  c = e._centers_np[lv, ty]
  scan = np.asarray(e.height_scan_at(jp.asarray([c[0], c[1], c[2] + T.SCAN_NOMINAL]), jp.asarray(0.0)))
  grid = scan.reshape(T.SCAN_X.size, T.SCAN_Y.size)
  front, here = grid[-1, 2], grid[list(T.SCAN_X).index(0.0), 2]
  assert abs(here) < 0.01 and front < -0.05


def test_expand_checkpoint_keeps_actions_identical(tmp_path):
  import functools
  from brax.training import checkpoint as brax_ckpt
  from brax.training.acme import running_statistics
  from brax.training.agents.ppo import networks as ppo_networks
  from robodog import expand_ckpt
  from robodog.policy_io import _CKPT_CONFIG, load_inference_fn

  blind = envs.make_env("terrain", {"impl": "jax", "noise_config": {"level": 0.0}, "terrain": {"max_level": 0}}).unwrapped
  seen = envs.make_env("terrain", {"impl": "jax", "noise_config": {"level": 0.0}, "terrain": {"max_level": 0, "height_scan": True}}).unwrapped
  sizes = {"state": (48,), "privileged_state": (123,)}
  factory = functools.partial(ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=(32, 32),
                              value_hidden_layer_sizes=(32, 32), policy_obs_key="state", value_obs_key="privileged_state")
  net = factory(sizes, 12, preprocess_observations_fn=running_statistics.normalize)
  key = jax.random.PRNGKey(3)
  pol = net.policy_network.init(key)
  val = net.value_network.init(key)
  spec = {k: jax.ShapeDtypeStruct(v, jp.float32) for k, v in sizes.items()}
  norm = running_statistics.init_state(spec)
  batch = {k: jax.random.normal(key, (64,) + v) * 2 + 1 for k, v in sizes.items()}
  norm = running_statistics.update(norm, batch)       # 有非 0 的平均和標準差，才測得出正規化有沒有對齊
  # 和訓練時存的格式一樣：{"state": {"shape": (48,), "dtype": float32}, ...}
  cfg = brax_ckpt.network_config({k: {"shape": v, "dtype": jp.float32} for k, v in sizes.items()}, 12, True, factory)
  brax_ckpt.save(tmp_path / "blind" / "checkpoints", 1000, (norm, pol, val), cfg, config_fname=_CKPT_CONFIG)
  src = tmp_path / "blind" / "checkpoints" / f"{1000:012d}"

  mean, std = expand_ckpt.scan_stats(seen, n=512)
  assert mean.shape == (T.N_SCAN,) and np.all(std >= 0.02)
  dst = expand_ckpt.expand_checkpoint(src, tmp_path / "seen", T.N_SCAN, mean, std)
  with pytest.raises(ValueError):                      # 擴充過的不能再擴充一次
    expand_ckpt.expand_checkpoint(dst, tmp_path / "again", T.N_SCAN, mean, std)

  f_blind, f_seen = load_inference_fn(src), load_inference_fn(dst)
  for i, (lv, ty) in enumerate([(0, 0), (3, 3), (5, 4), (7, 5)]):
    k = jax.random.PRNGKey(10 + i)
    a, b = blind.reset_at(k, lv, ty), seen.reset_at(k, lv, ty)
    act_a, _ = f_blind(a.obs, k)
    act_b, _ = f_seen(b.obs, k)
    np.testing.assert_allclose(np.asarray(act_b), np.asarray(act_a), atol=1e-5)   # 擴充後動作完全一樣
