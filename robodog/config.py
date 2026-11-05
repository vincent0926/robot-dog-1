"""設定檔處理：讀 YAML、合併多份設定、套用命令列 --set 覆寫。"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


def deep_merge(base: dict, override: dict) -> dict:
  """把 override 遞迴合併進 base（回傳新的 dict，不改動輸入）。"""
  out = copy.deepcopy(base)
  for key, value in (override or {}).items():
    if isinstance(value, dict) and isinstance(out.get(key), dict):
      out[key] = deep_merge(out[key], value)
    else:
      out[key] = copy.deepcopy(value)
  return out


def load_yaml(path: str | Path) -> dict:
  path = Path(path)
  if not path.exists() and (CONFIG_DIR / path).exists():
    path = CONFIG_DIR / path
  with open(path, encoding="utf-8") as f:
    return yaml.safe_load(f) or {}


def parse_set(expr: str) -> tuple[list[str], Any]:
  """解析 'ppo.num_envs=2048' 這種字串，值用 YAML 規則轉型。"""
  if "=" not in expr:
    raise ValueError(f"--set 格式要是 key.path=value，收到：{expr}")
  key, raw = expr.split("=", 1)
  return key.strip().split("."), yaml.safe_load(raw)


def apply_sets(cfg: dict, sets: list[str] | None) -> dict:
  cfg = copy.deepcopy(cfg)
  for expr in sets or []:
    keys, value = parse_set(expr)
    node = cfg
    for k in keys[:-1]:
      node = node.setdefault(k, {})
    node[keys[-1]] = value
  return cfg


def load_config(
    path: str | Path,
    extra_files: list[str | Path] | None = None,
    sets: list[str] | None = None,
) -> dict:
  """讀主設定，依序疊上 extra_files（例如 smoke.yaml），最後套用 --set。"""
  cfg = load_yaml(path)
  for extra in extra_files or []:
    cfg = deep_merge(cfg, load_yaml(extra))
  cfg = apply_sets(cfg, sets)
  if "task" not in cfg:
    raise ValueError(f"{path} 缺少 task 欄位（walk / getup / navigate）")
  return cfg


def update_config_dict(config_dict, overrides: dict, prefix: str = "") -> None:
  """把巢狀 dict 寫進 ml_collections.ConfigDict，遇到不存在的 key 直接報錯，避免打錯字沒發現。"""
  for key, value in (overrides or {}).items():
    if key not in config_dict:
      raise KeyError(
          f"環境設定沒有 '{prefix}{key}' 這個欄位，請檢查 YAML 是否打錯字。"
      )
    current = config_dict[key]
    if isinstance(value, dict) and hasattr(current, "keys"):
      update_config_dict(current, value, prefix=f"{prefix}{key}.")
    else:
      if isinstance(current, tuple) and isinstance(value, list):
        value = tuple(value)
      config_dict[key] = value
