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


def test_terrain_config_typo_is_rejected():
  with pytest.raises(KeyError):
    envs.make_env("terrain", {"impl": "jax", "terrain": {"max_levl": 3}})
