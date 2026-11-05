#!/usr/bin/env bash
# 冒煙測試：用 CPU 把三個任務的「訓練 → 評估」都跑過一次，確認程式沒壞。
# 這個規模學不會任何東西；整個腳本在一般電腦上約 10 分鐘。
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pytest -q
python -m robodog.train configs/walk.yaml --smoke --name smoke
python -m robodog.train configs/getup.yaml --smoke --name smoke
python -m robodog.train configs/navigate.yaml --smoke --name smoke
for t in walk getup navigate; do
  python -m robodog.evaluate "runs_smoke/$t/latest" --episodes 4 --impl jax
done
echo "冒煙測試全部通過 ✅"
