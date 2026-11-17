#!/usr/bin/env bash
# 讓狗「看得到」地形：擴充第二階段的策略 → 開地形掃描 + 自動調難度訓練 → 自動跑大評估。
#
# 用法（在專案資料夾裡，背景執行）：
#   setsid nohup bash scripts/plan_scan.sh > /dev/null 2>&1 &
#   cat chain_scan.log          # 看進度
#
# 訓練時請保留一個 Ubuntu 視窗開著、筆電插電並關掉睡眠（WSL 被關掉的話訓練會中斷）。
# 想換起點：SRC=runs/terrain/<run 名稱> bash scripts/plan_scan.sh
# 想練久一點：STEPS=900000000 bash scripts/plan_scan.sh
set -u
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
SRC=${SRC:-runs/terrain/20261115-204812-stage2}   # 盲走版目前最好的（大評估 75.5%）
STEPS=${STEPS:-600000000}                          # 6 億步，RTX 5060 約 1 小時
INIT=runs_init/scan-from-$(basename "$SRC")
LOG=chain_scan.log
export PYTHONUNBUFFERED=1

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

if pgrep -f "python[^ ]* -m robodog[.]train" > /dev/null; then
  log "有訓練正在跑，等它結束再開始…"
  while pgrep -f "python[^ ]* -m robodog[.]train" > /dev/null; do sleep 30; done
fi

# 1. 擴充：加上 85 個地形掃描輸入，對應的權重設 0 → 一開始的動作和原本完全一樣
log "擴充 $SRC → $INIT"
"$PY" -m robodog.expand_ckpt "$SRC" --out "$INIT" >> "$LOG" 2>&1 || { log "擴充失敗，排程停止"; exit 1; }

# 2. 訓練（顯示卡記憶體不足時自動改用 2048 個環境）
train() {
  local name=$1; shift
  "$PY" -m robodog.train configs/terrain.yaml --name "$name" --resume "$INIT" \
    --set env.terrain.height_scan=true --set env.terrain.curriculum=true --set env.terrain.max_level=7 \
    --set ppo.num_timesteps="$STEPS" "$@" > "train_$name.log" 2>&1
}
log "開始 scan（$STEPS 步）"
NAME=scan
if ! train scan; then
  if grep -q -E "RESOURCE_EXHAUSTED|out of memory|Failed to allocate" train_scan.log; then
    log "顯示卡記憶體不足，改用 2048 個環境重跑"
    NAME=scan-2048
    train scan-2048 --set ppo.num_envs=2048 || { log "scan-2048 失敗（看 train_scan-2048.log）"; exit 1; }
  else
    log "scan 失敗（看 train_scan.log 最後幾行）"; exit 1
  fi
fi
log "完成 $NAME"

# 3. 大評估（每格 32 回合），和盲走版的 75.5% 比較
RUN=runs/terrain/$(ls runs/terrain | grep -- "-$NAME\$" | tail -1)
log "大評估 $RUN"
"$PY" -m robodog.evaluate "$RUN" --episodes 1536 --json "eval_big_$NAME.json" > "eval_big_$NAME.txt" 2>&1
tail -n 12 "eval_big_$NAME.txt" | tee -a "$LOG"
log "全部完成。結果在 eval_big_$NAME.txt"
