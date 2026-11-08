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


def make_env(task: str, env_overrides: Optional[dict] = None, **kwargs):
  """建立環境。env_overrides 是 YAML 裡 env: 區塊的內容。

  另外支援一個 Playground 沒有的欄位 fall_penalty（見 FallPenalty）。
  """
  overrides = dict(env_overrides or {})
  fall_penalty = overrides.pop("fall_penalty", 0.0)
  config = default_env_config(task)
  update_config_dict(config, overrides)
  if task in PLAYGROUND_TASKS:
    env = registry.load(PLAYGROUND_TASKS[task], config=config)
    return FallPenalty(env, fall_penalty) if fall_penalty else env
  from robodog.envs import navigate
  return navigate.Navigate(config=config, **kwargs)


def randomization_fn(task: str) -> Optional[Callable]:
  """回傳 domain randomization 函式（摩擦力、質量、重心…）。導航任務不需要。"""
  if task in PLAYGROUND_TASKS:
    return go1_randomize.domain_randomize
  return None
