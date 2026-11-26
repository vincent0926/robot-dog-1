#!/usr/bin/env bash
# 一小時訓練排程（RTX 5060）：崎嶇地形 → 樓梯斜坡階段一，前一段成功才接下一段。
#
# 用法（在專案資料夾裡，背景執行，關掉視窗也會繼續跑）：
#   setsid nohup bash scripts/plan_1h.sh > /dev/null 2>&1 &
#   cat chain_1h.log          # 看排程進度
#
# - 如果已經有訓練在跑（例如導航），會先等它結束再開始，不會搶顯示卡。
# - 顯示卡記憶體不足時，自動改用 2048 個環境重跑同一段。
# - 想換起點：WALK=runs/walk/<run 名稱> bash scripts/plan_1h.sh
set -u
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
WALK=${WALK:-runs/walk/20261111-005756-gait3}   # 步態糾正好的走路策略
LOG=chain_1h.log
export PYTHONUNBUFFERED=1

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# run <名稱> <設定檔與其他參數…>
run() {
  local name=$1; shift
  log "開始 $name"
  if "$PY" -m robodog.train "$@" --name "$name" > "train_$name.log" 2>&1; then
    log "完成 $name"
    return 0
  fi
  if grep -q -E "RESOURCE_EXHAUSTED|out of memory|Failed to allocate" "train_$name.log"; then
    log "$name 顯示卡記憶體不足，改用 2048 個環境重跑"
    if "$PY" -m robodog.train "$@" --name "$name-2048" --set ppo.num_envs=2048 > "train_$name-2048.log" 2>&1; then
      log "完成 $name-2048"
      return 0
    fi
    log "$name-2048 也失敗了（看 train_$name-2048.log）"
    return 1
  fi
  log "$name 失敗（看 train_$name.log 最後幾行）"
  return 1
}

if pgrep -f "python[^ ]* -m robodog[.]train" > /dev/null; then
  log "有訓練正在跑，等它結束再開始…"
  while pgrep -f "python[^ ]* -m robodog[.]train" > /dev/null; do sleep 30; done
  log "前一個訓練結束了"
fi

# 1. 崎嶇地形：從步態糾正好的走路策略出發（1.5 億步，約 20～30 分鐘）
run gpu-rough configs/rough.yaml --resume "$WALK" || { log "排程停止"; exit 1; }
# 2. 樓梯斜坡階段一（難度 0~2：每階 2~4.8 cm、斜坡 3~8°，2 億步，約 20～30 分鐘）
run stage1 configs/terrain.yaml --resume runs/rough/latest || { log "排程停止"; exit 1; }

log "全部完成。下一步：樓梯斜坡階段二（見 configs/terrain.yaml 開頭的說明）"
