"""樓梯與斜坡：多種地形、多個難度的訓練場。

地圖是 28m x 28m 的高度場，切成 7 x 7 個 4m x 4m 的區塊。每個區塊是「一種地形 x 一個難度」：

  地形（type）：flat 平地(微凹凸)、slope_up 上坡、slope_down 下坡、stairs_up 上樓梯、
               stairs_down 下樓梯、boxes 散落的方塊
  難度（level）：0～7。樓梯每階高度 2cm → 12cm，斜坡 3° → 20°，方塊起伏 2cm → 10cm

樓梯和斜坡都是「金字塔」：中央是 1.5m 見方的平台，機器狗出生在平台中央；往四面八方走都是
同一種地形（上坡區塊是中間低、往外變高，下坡區塊相反），所以不管機器狗朝哪個方向都會遇到。
所有區塊的邊緣高度相同，從一個區塊走到另一個不會掉下懸崖。

策略是「盲」的（跟 walk、rough 一樣只看自己的關節、IMU 和指令），所以可以直接接續它們的權重。
難度用分階段訓練來控制（env.terrain.max_level），和我們先前其他任務的做法一樣。
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import base as go1_base
from mujoco_playground._src.locomotion.go1 import joystick as go1_joystick

TERRAIN_XML = Path(__file__).resolve().parent.parent / "assets" / "go1_terrain_scene.xml"

# 和 go1_terrain_scene.xml 裡的 hfield 一致（nrow = ncol = N，半邊長 R，最大高度 ZMAX）
N = 448
R = 14.0
ZMAX = 1.3
TILE = 4.0
GRID = 7
N_LEVELS = 8
TYPES = ("flat", "slope_up", "slope_down", "stairs_up", "stairs_down", "boxes")
N_TYPES = len(TYPES)
BASE_H = 0.65          # 所有區塊邊緣的高度（要夠高，讓最低的「上坡」區塊中心不會低於 0）
PLATFORM = 0.75        # 中央平台的半邊長（1.5m 見方）
STEPS = 4              # 平台外圍的階數（加上平台往外跨的那一階，一共 5 階）

# MuJoCo 的 hfield 資料：data[row * ncol + col]，row 對應 y 還是 x？（用 tests 裡的射線測試確認）
ROW_IS_Y = True


def default_config() -> config_dict.ConfigDict:
  cfg = go1_joystick.default_config()
  cfg.naconmax = 8 * 8192   # 和 Playground 的崎嶇地形一樣
  cfg.njmax = 12 + 48
  cfg.episode_length = 1000
  cfg.terrain = config_dict.create(
      max_level=7,          # 最難會抽到的難度（0～7）。分階段訓練時由淺到深
      step_base=0.02,       # 樓梯第 0 級每階高度（m）
      step_gain=0.014,      # 每升一級增加多少 → 第 7 級 0.118 m
      slope_base_deg=3.0,   # 斜坡第 0 級角度
      slope_gain_deg=2.4,   # 每升一級增加多少度 → 第 7 級 19.8°
      seed=0,
  )
  # 在樓梯上，這幾項要放寬，不然會懲罰「爬樓梯本來就會有的動作」：
  s = cfg.reward_config.scales
  s.feet_height = 0.0       # 原本用腳的絕對高度，在高處會失真（改用 feet_clearance 的「離地高度」版本）
  s.lin_vel_z = -0.1        # 爬樓梯機身本來就會上下移動
  s.orientation = -2.0      # 上下坡機身本來就會前傾或後仰
  return cfg


def level_params(cfg, level: int) -> dict:
  t = cfg.terrain
  return {
      "step": t.step_base + t.step_gain * level,
      "slope": np.deg2rad(t.slope_base_deg + t.slope_gain_deg * level),
      "amp": 0.02 + 0.012 * level,       # 方塊起伏
      "noise": 0.004 + 0.004 * level,    # 平地的微凹凸
  }


def tile_center(level: int, typ: int) -> tuple[float, float]:
  t = level * N_TYPES + typ
  gi, gj = divmod(t, GRID)
  return (gj - GRID // 2) * TILE, (gi - GRID // 2) * TILE


def _tile_height(typ: int, level: int, u, v, p: dict, rng) -> np.ndarray:
  """一個區塊的高度。u、v 是離區塊中心的座標（m），範圍 [-2, 2)。"""
  d = np.maximum(np.abs(u), np.abs(v))          # 方形金字塔的「圈數」
  inside = d >= PLATFORM
  half = TILE / 2
  if typ == 1 or typ == 2:                       # 斜坡
    rise = np.tan(p["slope"]) * (half - np.maximum(d, PLATFORM))
    return BASE_H - rise if typ == 1 else BASE_H + rise
  if typ == 3 or typ == 4:                       # 樓梯
    w = (half - PLATFORM) / STEPS
    k = np.floor((np.maximum(d, PLATFORM) - PLATFORM) / w).clip(0, STEPS - 1)
    n_up = np.where(inside, STEPS - k, STEPS + 1)   # 離區塊邊緣還差幾階
    return BASE_H - p["step"] * n_up if typ == 3 else BASE_H + p["step"] * n_up
  if typ == 5:                                   # 散落的方塊：0.5m 見方、高度隨機
    cells = int(TILE / 0.5)
    blocks = rng.uniform(-p["amp"], p["amp"], (cells, cells))
    iu = ((u + half) / 0.5).astype(int).clip(0, cells - 1)
    iv = ((v + half) / 0.5).astype(int).clip(0, cells - 1)
    return BASE_H + np.where(inside, blocks[iv, iu], 0.0)
  # typ == 0：平地，疊上平滑的微凹凸（平台範圍內保持平的）
  g = rng.uniform(-1, 1, (9, 9))
  gu = ((u + half) / TILE * 8)
  gv = ((v + half) / TILE * 8)
  i0, j0 = np.floor(gu).astype(int).clip(0, 7), np.floor(gv).astype(int).clip(0, 7)
  fu, fv = gu - i0, gv - j0
  noise = (g[j0, i0] * (1 - fu) * (1 - fv) + g[j0, i0 + 1] * fu * (1 - fv)
           + g[j0 + 1, i0] * (1 - fu) * fv + g[j0 + 1, i0 + 1] * fu * fv)
  return BASE_H + np.where(inside, noise * p["noise"], 0.0)


def build_heightmap(cfg) -> tuple[np.ndarray, np.ndarray]:
  """回傳 (H, centers)。

  H[ix, iy] 是 (x_ix, y_iy) 的高度（公尺），x、y 各 N 格，從 -R 到 R。
  centers[level, type] = (x, y, 平台高度)：機器狗出生的位置。
  """
  xs = np.linspace(-R, R, N)
  X, Y = np.meshgrid(xs, xs, indexing="ij")
  H = np.full((N, N), BASE_H)
  centers = np.zeros((N_LEVELS, N_TYPES, 3))
  half = TILE / 2
  for level in range(N_LEVELS):
    p = level_params(cfg, level)
    for typ in range(N_TYPES):
      cx, cy = tile_center(level, typ)
      m = (X >= cx - half) & (X < cx + half) & (Y >= cy - half) & (Y < cy + half)
      rng = np.random.default_rng(cfg.terrain.seed * 1000 + level * N_TYPES + typ)
      H[m] = _tile_height(typ, level, X[m] - cx, Y[m] - cy, p, rng)
      centers[level, typ] = (cx, cy, float(_tile_height(typ, level, np.zeros(1), np.zeros(1), p, rng)[0]))
  assert H.min() >= 0.0 and H.max() <= ZMAX, f"地形高度超出範圍：{H.min():.3f}～{H.max():.3f}（上限 {ZMAX}）"
  return H, centers


def to_hfield_data(H: np.ndarray) -> np.ndarray:
  """轉成 MuJoCo hfield_data（0～1，攤平）。"""
  grid = H.T if ROW_IS_Y else H
  return (grid / ZMAX).astype(np.float32).ravel()


def make_height_fn(H: np.ndarray):
  """回傳 f(xy) -> 地面高度（雙線性內插）。xy 的形狀 (..., 2)。給獎勵函式算「腳離地多高」用。"""
  Hj = jp.asarray(H, dtype=jp.float32)

  def f(xy):
    g = (xy + R) / (2 * R) * (N - 1)
    g = jp.clip(g, 0.0, N - 1.001)
    i0 = jp.floor(g).astype(jp.int32)
    fr = g - i0
    ix, iy = i0[..., 0], i0[..., 1]
    fx, fy = fr[..., 0], fr[..., 1]
    return (Hj[ix, iy] * (1 - fx) * (1 - fy) + Hj[ix + 1, iy] * fx * (1 - fy)
            + Hj[ix, iy + 1] * (1 - fx) * fy + Hj[ix + 1, iy + 1] * fx * fy)

  return f


class TerrainJoystick(go1_joystick.Joystick):
  """Go1 Joystick 環境，換成多種地形的訓練場。觀測、動作都和 walk / rough 相同。"""

  def __init__(self, config: config_dict.ConfigDict):
    mjx_env.ensure_menagerie_exists()  # 同導航環境：Go1 的 mesh 要先確認已經下載
    go1_base.Go1Env.__init__(self, xml_path=TERRAIN_XML.as_posix(), config=config)
    m = self._mj_model
    assert (m.hfield_nrow[0], m.hfield_ncol[0]) == (N, N) and abs(m.hfield_size[0][0] - R) < 1e-6, \
        "go1_terrain_scene.xml 的 hfield 大小和 terrain.py 不一致"
    self._heights, self._centers_np = build_heightmap(config)
    m.hfield_data[:] = to_hfield_data(self._heights)
    # Go1Env 在建構時就把 mj_model 轉成 mjx 了，高度資料是之後才填的，所以要再轉一次
    self._mjx_model = mjx.put_model(m, impl=self._config.impl)
    self._post_init()
    self._ground = make_height_fn(self._heights)
    self._centers = jp.asarray(self._centers_np, dtype=jp.float32)
    self._max_level = int(config.terrain.max_level)

  # ---- 出生 ----

  def reset_at(self, rng: jax.Array, level, typ) -> mjx_env.State:
    """在指定區塊（難度 level、地形 typ）的中央平台出生。評估時用它指定地形。"""
    state = super().reset(rng)   # 父類別的 reset：在世界原點出生（高度 0.35）
    cx, cy, cz = self._centers[level, typ]
    qpos = state.data.qpos
    qpos = qpos.at[0].add(cx).at[1].add(cy).at[2].add(cz)   # 平移到該區塊平台上
    data = mjx.forward(self.mjx_model, state.data.replace(qpos=qpos))
    return state.replace(data=data, obs=self._get_obs(data, state.info))

  def reset(self, rng: jax.Array) -> mjx_env.State:
    """訓練用：隨機抽一個區塊（難度不超過 terrain.max_level，地形六種等機率）。"""
    rng, k_level, k_type = jax.random.split(rng, 3)
    level = jax.random.randint(k_level, (), 0, self._max_level + 1)
    typ = jax.random.randint(k_type, (), 0, N_TYPES)
    return self.reset_at(rng, level, typ)

  # ---- 獎勵：離地高度要扣掉地面高度 ----

  def _cost_feet_clearance(self, data: mjx.Data) -> jax.Array:
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_norm = jp.sqrt(jp.linalg.norm(feet_vel[..., :2], axis=-1))
    foot_pos = data.site_xpos[self._feet_site_id]
    height = foot_pos[..., 2] - self._ground(foot_pos[..., :2])
    delta = jp.abs(height - self._config.reward_config.max_foot_height)
    return jp.sum(delta * vel_norm)

  def terrain_height(self, xy):
    return self._ground(xy)
