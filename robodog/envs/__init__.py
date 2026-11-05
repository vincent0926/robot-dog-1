"""任務註冊表：把 walk / getup / navigate 對應到實際的環境類別。"""

from __future__ import annotations

from typing import Callable, Optional

from mujoco_playground import registry
from mujoco_playground._src.locomotion.go1 import randomize as go1_randomize

from robodog.config import update_config_dict

# walk 和 getup 直接用 MuJoCo Playground 調好的 Go1 環境
PLAYGROUND_TASKS = {
    "walk": "Go1JoystickFlatTerrain",
    "getup": "Go1Getup",
}
TASKS = ("walk", "getup", "navigate")


def default_env_config(task: str):
  if task in PLAYGROUND_TASKS:
    return registry.get_default_config(PLAYGROUND_TASKS[task])
  if task == "navigate":
    from robodog.envs import navigate
    return navigate.default_config()
  raise ValueError(f"未知任務 '{task}'，可用：{TASKS}")


def make_env(task: str, env_overrides: Optional[dict] = None, **kwargs):
  """建立環境。env_overrides 是 YAML 裡 env: 區塊的內容。"""
  config = default_env_config(task)
  update_config_dict(config, env_overrides or {})
  if task in PLAYGROUND_TASKS:
    return registry.load(PLAYGROUND_TASKS[task], config=config)
  from robodog.envs import navigate
  return navigate.Navigate(config=config, **kwargs)


def randomization_fn(task: str) -> Optional[Callable]:
  """回傳 domain randomization 函式（摩擦力、質量、重心…）。導航任務不需要。"""
  if task in PLAYGROUND_TASKS:
    return go1_randomize.domain_randomize
  return None
