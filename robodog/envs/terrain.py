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

# 地形掃描（height_scan）：機身周圍的網格點，在「機身朝向」的座標系裡（x 往前、y 往左），單位公尺。
# 前方看得比較遠（1 m），後面 0.6 m，左右各 0.3 m，共 17 x 5 = 85 個點。
SCAN_X = np.round(np.arange(-0.6, 1.0001, 0.1), 3)
SCAN_Y = np.array([-0.3, -0.15, 0.0, 0.15, 0.3])
N_SCAN = SCAN_X.size * SCAN_Y.size
SCAN_NOMINAL = 0.3     # 站在平地上時，機身大約高出地面多少（掃描值會減掉它，平地 ≈ 0）

# MuJoCo 的 hfield 資料：data[row * ncol + col]，row 對應 y 還是 x？（用 tests 裡的射線測試確認）
ROW_IS_Y = True


def default_config() -> config_dict.ConfigDict:
  cfg = go1_joystick.default_config()
  cfg.naconmax = 8 * 8192   # 和 Playground 的崎嶇地形一樣
  # 約束列數上限（每個環境）。Playground 崎嶇地形用 12 + 48 = 60，但樓梯邊緣一隻腳可能同時碰到
  # 好幾個高度場面，GPU 上會出現「nefc overflow - please increase njmax beyond 60」，
  # 超過的接觸會被直接丟掉（腳可能陷進台階）。放大一倍，多用的顯示卡記憶體很少。
  cfg.njmax = 120
  cfg.episode_length = 1000
  cfg.terrain = config_dict.create(
      max_level=7,          # 最難會抽到的難度（0～7）。分階段訓練時由淺到深
      step_base=0.02,       # 樓梯第 0 級每階高度（m）
      step_gain=0.014,      # 每升一級增加多少 → 第 7 級 0.118 m
      slope_base_deg=3.0,   # 斜坡第 0 級角度
      slope_gain_deg=2.4,   # 每升一級增加多少度 → 第 7 級 19.8°
      seed=0,
      # 訓練時每種地形被抽到的相對比例，順序同 TYPES：
      #   flat, slope_up, slope_down, stairs_up, stairs_down, boxes
      # 全部一樣 = 等機率。想加練某種地形就調高它，例如上樓梯 3.5 → 約 41%。
      type_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
      # 自動調難度（課程學習）：每個環境依自己的表現升降難度，只在訓練時有效（評估固定難度）。
      #   這一回合走出區塊（離區塊中心 > curriculum_up 公尺）→ 下一回合難度 +1
      #   走的距離不到「指令要求距離」的 curriculum_down 倍（而且指令要求 > curriculum_min_cmd 公尺）→ -1
      # 難度上限仍是 max_level；一開始每個環境的難度在 0 ~ curriculum_start 之間隨機。
      curriculum=False,
      curriculum_start=2,
      # 地形掃描：觀測多加機身周圍 85 個點的地面高度（讓狗「看得到」台階）。觀測大小會改變，
      # 舊的盲走策略要先用 python -m robodog.expand_ckpt 擴充才能接著練。
      height_scan=False,
      curriculum_up=2.0,        # 半個區塊（區塊 4m，中央平台到邊緣約 2m）
      curriculum_down=0.5,
      curriculum_min_cmd=1.0,
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
    w = np.asarray(config.terrain.type_weights, dtype=np.float64)
    if w.shape != (N_TYPES,) or np.any(w < 0) or w.sum() <= 0:
      raise ValueError(f"terrain.type_weights 要是 {N_TYPES} 個不小於 0 的數字（順序：{', '.join(TYPES)}），收到 {tuple(w)}")
    self._type_uniform = bool(np.all(w == w[0]))  # 等機率時用原本的抽法（結果和加這個設定之前完全一樣）
    with np.errstate(divide="ignore"):
      self._type_logits = jp.asarray(np.log(w / w.sum()), dtype=jp.float32)  # 比例 0 → log 0 = -inf，永遠抽不到
    t = config.terrain
    self._scan = bool(t.height_scan)
    gx, gy = np.meshgrid(SCAN_X, SCAN_Y, indexing="ij")
    self._scan_pts = jp.asarray(np.stack([gx.ravel(), gy.ravel()], -1), dtype=jp.float32)   # [N_SCAN, 2]
    self._curriculum = bool(t.curriculum)
    self._cur_start = min(int(t.curriculum_start), self._max_level)
    self._cur_up, self._cur_down, self._cur_min_cmd = float(t.curriculum_up), float(t.curriculum_down), float(t.curriculum_min_cmd)

  # ---- 出生 ----

  def reset_at(self, rng: jax.Array, level, typ) -> mjx_env.State:
    """在指定區塊（難度 level、地形 typ）的中央平台出生。評估時用它指定地形。"""
    state = super().reset(rng)   # 父類別的 reset：在世界原點出生（高度 0.35）
    cx, cy, cz = self._centers[level, typ]
    qpos = state.data.qpos
    qpos = qpos.at[0].add(cx).at[1].add(cy).at[2].add(cz)   # 平移到該區塊平台上
    data = mjx.forward(self.mjx_model, state.data.replace(qpos=qpos))
    return state.replace(data=data, obs=self._get_obs(data, state.info))

  def _sample_type(self, key):
    if self._type_uniform:
      return jax.random.randint(key, (), 0, N_TYPES)
    return jax.random.categorical(key, self._type_logits).astype(jp.int32)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    """訓練用：隨機抽一個區塊（難度不超過 terrain.max_level；地形依 terrain.type_weights 的比例）。"""
    rng, k_level, k_type = jax.random.split(rng, 3)
    top = self._cur_start if self._curriculum else self._max_level
    level = jax.random.randint(k_level, (), 0, top + 1)
    typ = self._sample_type(k_type)
    if not self._curriculum:
      return self.reset_at(rng, level, typ)
    rng, k_cur = jax.random.split(rng)
    state = self.reset_at(rng, level, typ)
    state.info.update({
        "terrain_level": level, "terrain_type": typ, "next_type": typ,
        "cmd_dist": jp.zeros(()), "curriculum_rng": k_cur,
    })
    state.metrics["terrain/level"] = self._level_metric(level)
    return state

  # ---- 自動調難度 ----
  #
  # Playground 的 BraxAutoResetWrapper（full_reset=False）在回合結束時，會把機器狗放回
  # info["AutoResetWrapper_first_data"]（第一次出生的狀態）。這裡每一步都依「這回合到目前為止的表現」
  # 把那份出生狀態平移到下一回合該去的區塊，所以回合一結束就會在新難度的區塊出生。
  # 只有在包了那個 wrapper（訓練時）才會生效；評估、播放直接呼叫環境，難度不會變。

  _FIRST = "AutoResetWrapper_first_data"

  def _level_metric(self, level):
    # brax 的評估會把每一步的 metric 加總成「整回合的值」，除以回合長度後，加總結果就是平均難度
    return level.astype(jp.float32) / self._config.episode_length

  def _tile_of(self, xy):
    d = jp.sum(jp.square(self._centers[..., :2] - xy), axis=-1)   # [level, type]
    idx = jp.argmin(d)
    return (idx // N_TYPES).astype(jp.int32), (idx % N_TYPES).astype(jp.int32)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    nstate = super().step(state, action)
    if not self._curriculum or self._FIRST not in state.info or "steps" not in state.info:
      return nstate
    info = dict(nstate.info)
    new_ep = state.info["steps"] == 0          # 上一步剛重置過 → 這是新回合的第一步
    rng, k = jax.random.split(info["curriculum_rng"])
    # 新回合：從出生位置認出現在在哪個區塊，並先抽好「下一回合」的地形種類
    lv0, ty0 = self._tile_of(state.data.qpos[:2])
    level = jp.where(new_ep, lv0, info["terrain_level"])
    typ = jp.where(new_ep, ty0, info["terrain_type"])
    next_type = jp.where(new_ep, self._sample_type(k), info["next_type"])
    cmd_dist = jp.where(new_ep, 0.0, info["cmd_dist"]) + jp.linalg.norm(info["command"][:2]) * self.dt

    # 到目前為止的表現 → 下一回合的難度
    dist = jp.linalg.norm(nstate.data.qpos[:2] - self._centers[level, typ, :2])
    up = dist > self._cur_up
    down = (dist < self._cur_down * cmd_dist) & (cmd_dist > self._cur_min_cmd) & ~up
    next_level = jp.clip(level + up.astype(jp.int32) - down.astype(jp.int32), 0, self._max_level)

    # 把「回合結束後的出生狀態」平移到下一回合的區塊
    first = info[self._FIRST]
    fl, ft = self._tile_of(first.qpos[:2])
    shift = self._centers[next_level, next_type] - self._centers[fl, ft]
    info[self._FIRST] = first.replace(qpos=first.qpos.at[:3].add(shift))
    info.update(terrain_level=level, terrain_type=typ, next_type=next_type, cmd_dist=cmd_dist, curriculum_rng=rng)
    metrics = dict(nstate.metrics)
    metrics["terrain/level"] = self._level_metric(level)
    return nstate.replace(info=info, metrics=metrics)

  # ---- 地形掃描 ----

  def height_scan_at(self, base_xyz: jax.Array, yaw: jax.Array) -> jax.Array:
    """機身在 base_xyz、朝向 yaw 時，周圍 N_SCAN 個點「機身高度 - 地面高度 - SCAN_NOMINAL」，限制在 ±1 m。

    平地上 ≈ 0；前面是上樓梯 → 變成負的（地面比較高）；下樓梯 → 正的。
    """
    c, s = jp.cos(yaw), jp.sin(yaw)
    px, py = self._scan_pts[:, 0], self._scan_pts[:, 1]
    world = jp.stack([base_xyz[0] + c * px - s * py, base_xyz[1] + s * px + c * py], axis=-1)
    return jp.clip(base_xyz[2] - self._ground(world) - SCAN_NOMINAL, -1.0, 1.0)

  def _get_obs(self, data: mjx.Data, info: dict) -> dict:
    obs = super()._get_obs(data, info)
    if not self._scan:
      return obs
    q = data.qpos[3:7]
    yaw = jp.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    scan = self.height_scan_at(data.qpos[:3], yaw)
    info["rng"], k = jax.random.split(info["rng"])
    nz = self._config.noise_config.level * 0.02   # 和其他感測器一樣加一點雜訊（約 ±2 cm）
    scan = scan + (2 * jax.random.uniform(k, scan.shape) - 1) * nz
    s = obs["state"]
    # 插在最後 3 格（速度指令）前面：播放、錄影時會直接改 state 的最後 3 格當指令，位置不能變
    state = jp.concatenate([s[:-3], scan, s[-3:]])
    priv = jp.concatenate([state, obs["privileged_state"][s.shape[0]:]])   # privileged_state 開頭就是 state
    return {"state": state, "privileged_state": priv}

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
