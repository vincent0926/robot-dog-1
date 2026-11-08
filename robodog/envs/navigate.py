"""導航避障任務（階層式）。

架構：
  高階策略（這裡要訓練的）每 0.1 秒輸出一個速度指令 (vx, vy, wz)，
  低階策略（已訓練好的 walk 策略，凍結不訓練）負責把指令變成 12 個關節動作。

所以導航策略只需要學「往哪走、怎麼繞」，不用從頭學走路，訓練快很多。

觀測（全部在機器狗自己的座標系）：
  - 目標點的相對位置與距離
  - 機身速度、角速度、重力方向
  - n_rays 條水平雷射（類似 2D 光達）量到障礙物或圍牆的距離
  - 上一次的速度指令
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jp
from ml_collections import config_dict
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import base as go1_base
from mujoco_playground._src.locomotion.go1 import joystick as go1_joystick

NAV_XML = Path(__file__).resolve().parent.parent / "assets" / "go1_nav_scene.xml"
MAX_OBSTACLES = 8  # 和 go1_nav_scene.xml 裡的 obstacle body 數量一致
FAR_AWAY = 100.0   # 停用的障礙物放到這麼遠


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.1,               # 高階策略的控制週期（秒）
      sim_dt=0.004,
      episode_length=200,        # 高階步數，200 * 0.1 = 20 秒
      low_level_policy="runs/walk/latest",  # walk 策略的路徑（run 資料夾或 checkpoint）
      low_level_dt=0.02,         # 低階 walk 策略的控制週期，要和訓練 walk 時一樣
      command_scale=[1.0, 0.6, 1.2],  # 動作 [-1,1] 對應到的最大速度 vx, vy, wz
      arena_half_size=5.0,       # 場地是 [-5,5] x [-5,5] 公尺
      goal_distance=[2.5, 4.5],  # 目標距離起點的範圍
      goal_radius=0.4,           # 進入這個距離就算抵達
      n_obstacles=5,             # 最多 8
      obstacle_radius=0.4,       # 要和 XML 的圓柱半徑一致
      obstacles_on_path=3,       # 其中幾個障礙物故意擺在起點→目標的路徑附近
      robot_radius=0.3,          # 機身中心離障礙物表面小於這個距離就算撞到
      n_rays=16,
      ray_max_range=3.0,
      reward_config=config_dict.create(
          scales=config_dict.create(
              progress=2.0,       # 每秒往目標靠近多少公尺
              heading=0.1,        # 面向目標
              reached=10.0,       # 抵達目標（一次性）
              collision=-10.0,    # 撞到障礙物或出界（一次性）
              fall=-10.0,         # 跌倒（一次性）
              action_rate=-0.05,  # 指令變化太劇烈
              time=-0.02,         # 每一步的時間成本，鼓勵快點到
          ),
      ),
      impl="warp",
      naconmax=4 * 8192,
      njmax=40,
  )


class _LowLevelJoystick(go1_joystick.Joystick):
  """Go1 Joystick 環境，但換成有障礙物的導航場景 XML。"""

  def __init__(self, config: config_dict.ConfigDict):
    # Go1 的 mesh（hip.stl 等）在 mujoco_menagerie 裡，Playground 第一次用到時才會下載。
    # registry.load() 會先呼叫 ensure_menagerie_exists()，但這裡是直接建 Go1Env，
    # 不自己確認的話，在 menagerie 還沒下載的電腦上就會找不到 mesh 檔
    # （之前完整跑 pytest 時，要等後面的 walk 測試觸發下載，順序在前的導航測試就會失敗）。
    mjx_env.ensure_menagerie_exists()
    go1_base.Go1Env.__init__(self, xml_path=NAV_XML.as_posix(), config=config)
    self._post_init()


def ray_distances(
    origin: jax.Array,
    yaw: jax.Array,
    centers: jax.Array,
    radius: float,
    arena_half_size: float,
    n_rays: int,
    max_range: float,
) -> jax.Array:
  """從 origin 往 n_rays 個方向打水平雷射，回傳每條雷射撞到圓柱或圍牆的距離。

  圓柱：解 |origin + t*d - c| = r 的二次方程式。
  圍牆：x = ±L、y = ±L 四條直線。
  """
  angles = yaw + jp.linspace(0.0, 2 * jp.pi, n_rays, endpoint=False)
  dirs = jp.stack([jp.cos(angles), jp.sin(angles)], axis=-1)  # (n_rays, 2)

  # 圓柱
  oc = origin[None, :] - centers  # (n_obs, 2)
  b = dirs @ oc.T                  # (n_rays, n_obs)
  c = jp.sum(oc**2, axis=-1) - radius**2  # (n_obs,)
  disc = b**2 - c[None, :]
  sqrt_disc = jp.sqrt(jp.maximum(disc, 0.0))
  t = -b - sqrt_disc
  t = jp.where(c[None, :] < 0.0, 0.0, t)  # 起點在圓柱裡面 → 距離 0
  t = jp.where((disc > 0.0) & (t >= 0.0), t, jp.inf)
  t_obs = jp.min(t, axis=-1)

  # 圍牆
  safe = jp.where(jp.abs(dirs) < 1e-6, 1e-6, dirs)
  t_pos = (arena_half_size - origin[None, :]) / safe
  t_neg = (-arena_half_size - origin[None, :]) / safe
  t_wall = jp.where(dirs > 0, t_pos, t_neg)
  t_wall = jp.where(jp.abs(dirs) < 1e-6, jp.inf, t_wall)
  t_wall = jp.maximum(jp.min(t_wall, axis=-1), 0.0)

  return jp.clip(jp.minimum(t_obs, t_wall), 0.0, max_range)


def _yaw_from_quat(q: jax.Array) -> jax.Array:
  w, x, y, z = q[0], q[1], q[2], q[3]
  return jp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _rotate_to_body(vec_xy: jax.Array, yaw: jax.Array) -> jax.Array:
  c, s = jp.cos(yaw), jp.sin(yaw)
  return jp.array([c * vec_xy[0] + s * vec_xy[1], -s * vec_xy[0] + c * vec_xy[1]])


class Navigate(mjx_env.MjxEnv):
  """高階導航環境。"""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Any]] = None,
      low_level_policy_fn: Optional[Callable] = None,
  ):
    super().__init__(config, config_overrides)
    cfg = self._config
    if cfg.n_obstacles > MAX_OBSTACLES:
      raise ValueError(f"n_obstacles 最多 {MAX_OBSTACLES}")

    # 低階環境：關掉推擠干擾，雜訊保留（和 walk 訓練時一樣）
    ll_cfg = go1_joystick.default_config()
    ll_cfg.ctrl_dt = cfg.low_level_dt
    ll_cfg.sim_dt = cfg.sim_dt
    ll_cfg.impl = cfg.impl
    ll_cfg.naconmax = cfg.naconmax
    ll_cfg.njmax = cfg.njmax
    ll_cfg.pert_config.enable = False
    self._ll_env = _LowLevelJoystick(ll_cfg)

    if low_level_policy_fn is None:
      from robodog.policy_io import load_inference_fn  # 延遲載入，避免循環 import
      low_level_policy_fn = load_inference_fn(cfg.low_level_policy)
    self._ll_policy = low_level_policy_fn

    self._ll_steps = int(round(cfg.ctrl_dt / cfg.low_level_dt))
    self._cmd_scale = jp.array(cfg.command_scale)
    self._goal_mocap = self.mj_model.body("goal").mocapid[0]
    self._obstacle_mocap = np.array(
        [self.mj_model.body(f"obstacle{i}").mocapid[0] for i in range(MAX_OBSTACLES)]
    )

  # ---- MjxEnv 需要的屬性：全部沿用低階環境 ----
  @property
  def xml_path(self) -> str:
    return self._ll_env.xml_path

  @property
  def action_size(self) -> int:
    return 3

  @property
  def mj_model(self):
    return self._ll_env.mj_model

  @property
  def mjx_model(self):
    return self._ll_env.mjx_model

  @property
  def model_assets(self):
    return self._ll_env.model_assets

  # ---- 場地產生 ----
  def _sample_layout(self, rng, start_xy):
    cfg = self._config
    L = cfg.arena_half_size
    rng, k_ang, k_dist = jax.random.split(rng, 3)
    ang = jax.random.uniform(k_ang, minval=-jp.pi, maxval=jp.pi)
    dist = jax.random.uniform(k_dist, minval=cfg.goal_distance[0], maxval=cfg.goal_distance[1])
    goal = start_xy + dist * jp.array([jp.cos(ang), jp.sin(ang)])
    goal = jp.clip(goal, -L + 0.6, L - 0.6)

    # 一部分障礙物沿著起點→目標的路徑擺（強迫機器狗學繞路），其餘隨機擺
    rng, k_s, k_lat, k_rand = jax.random.split(rng, 4)
    s = jax.random.uniform(k_s, (MAX_OBSTACLES,), minval=0.3, maxval=0.7)
    path = goal - start_xy
    normal = jp.array([-path[1], path[0]]) / (jp.linalg.norm(path) + 1e-6)
    lateral = jax.random.normal(k_lat, (MAX_OBSTACLES,)) * 0.4
    on_path = start_xy + s[:, None] * path + lateral[:, None] * normal
    random_pos = jax.random.uniform(k_rand, (MAX_OBSTACLES, 2), minval=-L + 0.5, maxval=L - 0.5)
    idx = jp.arange(MAX_OBSTACLES)
    centers = jp.where((idx < cfg.obstacles_on_path)[:, None], on_path, random_pos)

    # 離起點或目標太近的障礙物、以及超過 n_obstacles 的，都移到遠處停用
    keep_out = cfg.obstacle_radius + cfg.robot_radius + 0.3
    too_close = (jp.linalg.norm(centers - start_xy, axis=-1) < keep_out) | (
        jp.linalg.norm(centers - goal, axis=-1) < keep_out
    )
    disabled = too_close | (idx >= cfg.n_obstacles)
    centers = jp.where(disabled[:, None], FAR_AWAY, centers)
    return goal, centers

  # ---- 低階 State 打包 / 解包 ----
  @staticmethod
  def _pack_ll(ll: mjx_env.State) -> dict:
    return {"obs": ll.obs, "info": ll.info, "metrics": ll.metrics}

  @staticmethod
  def _unpack_ll(data, packed: dict) -> mjx_env.State:
    return mjx_env.State(
        data=data,
        obs=dict(packed["obs"]),
        reward=jp.zeros(()),
        done=jp.zeros(()),
        metrics=dict(packed["metrics"]),
        info=dict(packed["info"]),
    )

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, ll_rng, layout_rng = jax.random.split(rng, 3)
    ll = self._ll_env.reset(ll_rng)
    start_xy = ll.data.qpos[:2]
    goal, centers = self._sample_layout(layout_rng, start_xy)

    mocap_pos = ll.data.mocap_pos
    mocap_pos = mocap_pos.at[self._obstacle_mocap, :2].set(centers)
    mocap_pos = mocap_pos.at[self._obstacle_mocap, 2].set(self._config.obstacle_radius)
    mocap_pos = mocap_pos.at[self._goal_mocap].set(jp.array([goal[0], goal[1], 0.02]))
    data = ll.data.replace(mocap_pos=mocap_pos)

    info = {
        "rng": rng,
        "goal": goal,
        "obstacles": centers,
        "last_act": jp.zeros(3),
        "prev_dist": jp.linalg.norm(goal - start_xy),
        "ll": self._pack_ll(ll),
    }
    metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}
    metrics.update({"reached": jp.zeros(()), "collided": jp.zeros(()), "fell": jp.zeros(())})
    obs = self._get_obs(data, info)
    return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    cfg = self._config
    action = jp.clip(action, -1.0, 1.0)
    cmd = action * self._cmd_scale

    def ll_step(carry, _):
      ll, fell = carry
      obs = dict(ll.obs)
      obs["state"] = obs["state"].at[-3:].set(cmd)  # 觀測最後 3 格是速度指令
      ll_action, _ = self._ll_policy(obs, jax.random.PRNGKey(0))
      ll.info["command"] = cmd
      ll = self._ll_env.step(ll, ll_action)
      return (ll, fell | (ll.done > 0)), None

    ll = self._unpack_ll(state.data, state.info["ll"])
    (ll, fell), _ = jax.lax.scan(ll_step, (ll, jp.array(False)), None, length=self._ll_steps)
    data = ll.data

    info = dict(state.info)
    info["ll"] = self._pack_ll(ll)

    base_xy = data.qpos[:2]
    yaw = _yaw_from_quat(data.qpos[3:7])
    to_goal = info["goal"] - base_xy
    dist = jp.linalg.norm(to_goal)
    clearance = jp.min(jp.linalg.norm(info["obstacles"] - base_xy, axis=-1)) - cfg.obstacle_radius
    out_of_arena = jp.any(jp.abs(base_xy) > cfg.arena_half_size)

    reached = dist < cfg.goal_radius
    collided = (clearance < cfg.robot_radius) | out_of_arena
    fell = fell | (self._ll_env.get_upvector(data)[-1] < 0.0)

    goal_body = _rotate_to_body(to_goal, yaw)
    rewards = {
        "progress": (info["prev_dist"] - dist) / cfg.ctrl_dt,
        "heading": goal_body[0] / (dist + 1e-6),
        "reached": reached.astype(jp.float32),
        "collision": collided.astype(jp.float32),
        "fall": fell.astype(jp.float32),
        "action_rate": jp.sum((action - info["last_act"]) ** 2),
        "time": jp.ones(()),
    }
    rewards = {k: v * cfg.reward_config.scales[k] for k, v in rewards.items()}
    reward = sum(rewards.values())

    info["prev_dist"] = dist
    info["last_act"] = action
    done = reached | collided | fell

    metrics = dict(state.metrics)
    for k, v in rewards.items():
      metrics[f"reward/{k}"] = v
    metrics["reached"] = reached.astype(jp.float32)
    metrics["collided"] = collided.astype(jp.float32)
    metrics["fell"] = fell.astype(jp.float32)

    obs = self._get_obs(data, info)
    return state.replace(
        data=data, obs=obs, reward=reward, done=done.astype(jp.float32),
        metrics=metrics, info=info,
    )

  def _get_obs(self, data, info) -> Dict[str, jax.Array]:
    cfg = self._config
    base_xy = data.qpos[:2]
    yaw = _yaw_from_quat(data.qpos[3:7])
    to_goal = info["goal"] - base_xy
    dist = jp.linalg.norm(to_goal)
    goal_body = _rotate_to_body(to_goal, yaw)
    rays = ray_distances(
        base_xy, yaw, info["obstacles"], cfg.obstacle_radius,
        cfg.arena_half_size, cfg.n_rays, cfg.ray_max_range,
    )
    state = jp.hstack([
        goal_body / (dist + 1e-6),                # 目標方向（單位向量）2
        jp.minimum(dist, 5.0) / 5.0,              # 目標距離 1
        self._ll_env.get_local_linvel(data),      # 3
        self._ll_env.get_gyro(data),              # 3
        self._ll_env.get_gravity(data),           # 3
        rays / cfg.ray_max_range,                 # n_rays
        info["last_act"],                         # 3
    ])
    return {"state": state, "privileged_state": state}
