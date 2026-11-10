"""任務註冊表：把 walk / getup / navigate 對應到實際的環境類別。"""

from __future__ import annotations

from typing import Callable, Optional

from mujoco_playground import registry
from mujoco_playground._src import wrapper
from mujoco_playground._src.locomotion.go1 import randomize as go1_randomize

from robodog.config import update_config_dict

# walk 和 getup 直接用 MuJoCo Playground 調好的 Go1 環境
PLAYGROUND_TASKS = {
    "walk": "Go1JoystickFlatTerrain",
    "getup": "Go1Getup",
    "rough": "Go1JoystickRoughTerrain",   # 崎嶇地形（高度場），觀測和 walk 一樣，可以直接接續 walk 的策略
}
TASKS = ("walk", "getup", "navigate", "rough")
WALK_LIKE = ("walk", "rough")  # 用搖桿速度指令走路的任務


def default_env_config(task: str):
  if task in PLAYGROUND_TASKS:
    return registry.get_default_config(PLAYGROUND_TASKS[task])
  if task == "navigate":
    from robodog.envs import navigate
    return navigate.default_config()
  raise ValueError(f"未知任務 '{task}'，可用：{TASKS}")


class FallPenalty(wrapper.Wrapper):
  """跌倒時直接扣分。

  Playground 的 Go1 環境會把每一步的總 reward 截到 >= 0，裡面的 termination 懲罰因此幾乎沒有作用，
  策略不太怕跌倒（崎嶇地形上高速直衝時跌倒率超過 20%）。這裡在截斷「之後」再扣 penalty 分，跌倒就真的會痛。
  """

  def __init__(self, env, penalty: float):
    super().__init__(env)
    self.penalty = float(penalty)

  def step(self, state, action):
    nstate = self.env.step(state, action)
    return nstate.replace(reward=nstate.reward - self.penalty * nstate.done)


GAIT_DEFAULTS = {
    "trot": 1.0,          # 對角小跑：左前＋右後一組、右前＋左後一組，兩組輪流著地
    "flight": 1.0,        # 四隻腳同時離地（用跳的）要扣分
    "short_swing": 0.3,   # 每一步騰空時間太短（碎步）要扣分
    "swing_time": 0.2,    # 希望每一步騰空至少幾秒；0.2 秒 ≈ 每秒 2.5 步
}


class GaitShaping(wrapper.Wrapper):
  """把走路姿勢糾正成「對角小跑」（trot），也就是 Spot 這類四足機器人最常見的走法。

  只靠「跟上速度」的獎勵，策略會學出能走但很醜的步態（實測：跛腳、左後腳幾乎不著地、
  24% 的時間四腳騰空、每秒 4 步的碎步）。這裡用腳的著地訊號另外加三項，只在有移動指令時計算：
    trot        對角的兩隻腳同時著地／離地，而且兩組對角腳相反
    flight      四隻腳同時離地
    short_swing 腳落地時，如果這一步騰空不到 swing_time 秒就扣分
  腳的順序是 FR、FL、RR、RL（右前、左前、右後、左後）。
  """

  KEYS = ("gait_trot", "gait_flight", "gait_short_swing")

  def __init__(self, env, **weights):
    super().__init__(env)
    unknown = set(weights) - set(GAIT_DEFAULTS)
    if unknown:
      raise KeyError(f"env.gait 沒有這些欄位：{sorted(unknown)}，可用：{sorted(GAIT_DEFAULTS)}")
    self.w = {**GAIT_DEFAULTS, **{k: float(v) for k, v in weights.items()}}

  def reset(self, rng):
    import jax.numpy as jp

    state = self.env.reset(rng)
    return state.replace(metrics={**state.metrics, **{f"reward/{k}": jp.zeros(()) for k in self.KEYS}})

  def step(self, state, action):
    import jax.numpy as jp

    prev_air = state.info["feet_air_time"]  # 這一步之前每隻腳已經騰空多久
    metrics_in = {k: v for k, v in state.metrics.items() if not k.startswith("reward/gait_")}
    nstate = self.env.step(state.replace(metrics=metrics_in), action)
    c = nstate.info["last_contact"].astype(jp.float32)  # 這一步各腳有沒有著地
    fr, fl, rr, rl = c[0], c[1], c[2], c[3]
    moving = (jp.linalg.norm(nstate.info["command"]) > 0.1).astype(jp.float32)

    same = lambda a, b: 1.0 - jp.abs(a - b)
    trot = (same(fr, rl) + same(fl, rr) + (1.0 - same(fr, fl)) + (1.0 - same(rr, rl))) / 4.0
    flight = (jp.sum(c) == 0).astype(jp.float32)
    touchdown = (prev_air > 0.0) * c
    short = jp.sum(jp.clip(self.w["swing_time"] - (prev_air + self.dt), 0.0, None) * touchdown)

    terms = {
        "gait_trot": self.w["trot"] * trot * self.dt * moving,
        "gait_flight": -self.w["flight"] * flight * self.dt * moving,
        "gait_short_swing": -self.w["short_swing"] * short * moving,  # 落地瞬間才算一次，不乘 dt
    }
    alive = 1.0 - nstate.done
    bonus = sum(terms.values()) * alive
    metrics = {**nstate.metrics, **{f"reward/{k}": v * alive for k, v in terms.items()}}
    return nstate.replace(reward=nstate.reward + bonus, metrics=metrics)


def make_env(task: str, env_overrides: Optional[dict] = None, **kwargs):
  """建立環境。env_overrides 是 YAML 裡 env: 區塊的內容。

  另外支援兩個 Playground 沒有的欄位：fall_penalty（見 FallPenalty）、gait（見 GaitShaping）。
  """
  overrides = dict(env_overrides or {})
  fall_penalty = overrides.pop("fall_penalty", 0.0)
  gait = overrides.pop("gait", None)
  config = default_env_config(task)
  update_config_dict(config, overrides)
  if task in PLAYGROUND_TASKS:
    env = registry.load(PLAYGROUND_TASKS[task], config=config)
    if gait is not None:
      if task not in WALK_LIKE:
        raise ValueError(f"env.gait 只能用在走路任務 {WALK_LIKE}，不能用在 {task}")
      env = GaitShaping(env, **(gait or {}))
    return FallPenalty(env, fall_penalty) if fall_penalty else env
  from robodog.envs import navigate
  return navigate.Navigate(config=config, **kwargs)


def randomization_fn(task: str) -> Optional[Callable]:
  """回傳 domain randomization 函式（摩擦力、質量、重心…）。導航任務不需要。"""
  if task in PLAYGROUND_TASKS:
    return go1_randomize.domain_randomize
  return None
