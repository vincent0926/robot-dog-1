# 機器狗強化學習訓練模擬器

用強化學習（PPO）在 GPU 上同時模擬幾千隻機器狗，讓牠學會三種技能：

| 任務 | 設定檔 | 學什麼 | 驗收標準（見 [PRD](docs/PRD.md)） |
|---|---|---|---|
| `walk` | `configs/walk.yaml` | 跟著搖桿速度指令走路 | 速度誤差 < 0.15 m/s，跌倒率 < 5% |
| `getup` | `configs/getup.yaml` | 跌倒後自己站起來 | 5 秒內站穩的比例 ≥ 85% |
| `navigate` | `configs/navigate.yaml` | 繞過障礙物走到目標點 | 抵達率 ≥ 80% |

技術組合：

- 物理模擬：[MuJoCo](https://mujoco.org/) + MuJoCo Warp（在 GPU 上平行跑）
- 環境基礎：[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
- PPO 演算法：[Brax](https://github.com/google/brax)
- 機器狗模型：Unitree Go1

> **為什麼是 Go1 不是 Go2？** Playground 已經把 Go1 的模型調整成適合 GPU 平行模擬（簡化碰撞、調好 PD 增益），可以直接訓練。Go2 換模型的步驟見文末「換一隻機器狗」。

---

## 1. 安裝（Windows + RTX 5060）

JAX 的 GPU 版只支援 Linux，所以在 Windows 上要透過 **WSL2** 執行。整個安裝大約 30 分鐘。

### 1-1 更新顯示卡驅動（在 Windows 做）

1. 到 NVIDIA 官網或 NVIDIA App 安裝最新的 **Game Ready** 或 **Studio** 驅動。RTX 50 系列需要 570 版以上。
2. **不要**在 WSL 裡面另外安裝 Linux 版的 NVIDIA 驅動，WSL 會直接使用 Windows 的驅動。

### 1-2 安裝 WSL2 + Ubuntu（在 PowerShell 做）

用「以系統管理員身分執行」打開 PowerShell，輸入：

```powershell
wsl --install -d Ubuntu-24.04
```

裝完重開機，打開「Ubuntu」設定帳號密碼。之後的指令都在 **Ubuntu 視窗**裡執行。

### 1-3 確認 WSL 看得到顯示卡

```bash
nvidia-smi
```

有看到 `GeForce RTX 5060` 就代表 GPU 接上了。

### 1-4 下載專案、安裝套件

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip git
```

```bash
git clone https://github.com/vincent0926/robot-dog-1.git
```

```bash
cd robot-dog-1
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-gpu.txt
```

> 專案放在 WSL 自己的檔案系統（`~/robot-dog-1`），**不要**放在 `/mnt/c/...`，不然讀寫會慢很多。

### 1-5 確認 JAX 用的是 GPU

```bash
python -c "import jax; print(jax.devices())"
```

要看到 `CudaDevice(id=0)`。如果只看到 `CpuDevice`，請看「常見問題」。

### 1-6 跑測試

```bash
python -m pytest -q
```

---

## 2. 訓練

建議依序執行，**導航會用到訓練好的 walk 策略**。

```bash
python -m robodog.train configs/walk.yaml --name first
```

```bash
python -m robodog.train configs/getup.yaml --name first
```

```bash
python -m robodog.train configs/navigate.yaml --name first
```

訓練結果存在 `runs/<任務>/<時間>-<名稱>/`：

```
runs/walk/20261105-200000-first/
├── config.yaml        這次用的完整設定（方便日後對照）
├── tb/                TensorBoard 紀錄
├── checkpoints/       每次評估存一份策略
└── summary.json       花了多久
```

### 常用參數

| 想做的事 | 指令 |
|---|---|
| 顯示卡記憶體不夠 | `--set ppo.num_envs=2048` |
| 改 reward 權重做實驗 | `--set env.reward_config.scales.feet_air_time=0.3 --name airtime03` |
| 從上次的權重繼續訓練 | `--resume runs/walk/latest` |
| 先跑小規模確認程式沒壞（CPU 就能跑） | `--smoke` |

`runs/walk/latest` 會自動指到最新一次 walk 訓練的最後一個 checkpoint。

### 看訓練曲線

另開一個 Ubuntu 視窗：

```bash
cd ~/robot-dog-1 && source .venv/bin/activate
tensorboard --logdir runs
```

然後在 Windows 瀏覽器打開 http://localhost:6006 。每一項 reward 都有自己的曲線，例如 `eval/episode_reward/tracking_lin_vel`，方便找出是哪一項造成奇怪的步態。

---

## 3. 評估（對照驗收標準）

```bash
python -m robodog.evaluate runs/walk/latest
```

輸出範例：

```
=== 評估結果：walk（256 回合）===
  vel_error_mps      0.1123   目標 < 0.15  ✅ 通過
  fall_rate          0.0156   目標 < 0.05  ✅ 通過
```

---

## 4. 播放

### 開 3D 視窗

Windows 11 的 WSL2 可以直接開視窗：

```bash
python -m robodog.play runs/walk/latest
```

walk 任務可以用鍵盤控制，先用滑鼠點一下視窗：

| 按鍵 | 動作 |
|---|---|
| ↑ / ↓ | 前進 / 後退速度 ±0.25 m/s |
| ← / → | 左轉 / 右轉 |
| `,` / `.` | 向左 / 向右平移 |
| `/` | 停下來 |

### 錄成影片

```bash
python -m robodog.play runs/walk/latest --video walk.mp4 --seconds 12
```

---

## 5. 專案結構

```
configs/            每個任務一份 YAML（reward 權重、環境數量、網路大小…）
  smoke.yaml        --smoke 時疊上去的「超小規模」設定
robodog/
  train.py          訓練入口（PPO、TensorBoard、checkpoint、NaN 偵測、記憶體不足提示）
  evaluate.py       依 PRD 的驗收標準評估
  play.py           3D 視窗 / 錄影
  config.py         YAML 讀取與 --set 覆寫
  policy_io.py      找 checkpoint、載入策略
  envs/__init__.py  任務註冊表（walk / getup 直接用 Playground 的 Go1 環境）
  envs/navigate.py  導航避障環境（自己寫的，階層式）
  assets/go1_nav_scene.xml  導航場景（障礙物、目標點）
tests/              pytest
scripts/smoke_test.sh  三個任務的訓練加評估全部跑一遍
docs/PRD.md         需求規格
```

### 導航任務是怎麼運作的？

```
導航策略（要訓練）── 每 0.1 秒 ──▶ 速度指令 (vx, vy, wz)
                                        │
                                        ▼
walk 策略（已訓練、凍結）── 每 0.02 秒 ──▶ 12 個關節目標角度 ──▶ MuJoCo
```

導航策略看到的資訊：目標方向與距離、機身速度、16 條水平雷射量到障礙物的距離。它只要學「往哪走」，不用重新學走路，所以訓練很快。

障礙物不參與物理碰撞，「撞到」是用機身中心到障礙物的距離判斷，雷射距離也是用幾何公式直接算出來的。這樣比較省 GPU 記憶體。

---

## 6. 常見問題

**`jax.devices()` 只看到 CPU**

```bash
pip install -U "jax[cuda12]==0.9.2"
```

裝完再確認 `nvidia-smi` 正常。如果還是不行，把錯誤訊息貼出來。

**訓練時出現 `RESOURCE_EXHAUSTED` / out of memory**

程式會提示建議的 `num_envs`。也可以限制 JAX 只用部分顯示卡記憶體：

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
```

**出現 `jax.device_put_replicated is deprecated`**

JAX 版本太新。請用 `requirements.txt` 鎖定的版本（JAX 0.9.2）。

**沒有 NVIDIA 顯示卡也想跑**

用縮小版設定：

```bash
python -m robodog.train configs/walk_cpu.yaml
```

這份設定是 256 個環境、2,000 萬步。4 核 CPU 約每秒 3,700 步，跑完大概 1.5～2 小時，品質不如 GPU 版。

---

## 7. 換一隻機器狗（P2，尚未實作）

1. 把新機器狗的 MJCF 放進 `robodog/assets/`。可以參考 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) 的 `unitree_go2`。
2. 參考 Playground 的 `go1/xmls/go1_mjx_feetonly.xml` 做簡化：只留腳的碰撞、加上 IMU 和腳底感測器。
3. 寫一個繼承 `Joystick` 的類別，換掉 `xml_path`、PD 增益和站姿關鍵影格。

---

## 進度

| 項目 | 狀態 |
|---|---|
| P0-1 GPU 平行模擬 | ✅ 已完成（需要在 5060 上實測吞吐量和記憶體） |
| P0-2 預設機器狗 | ✅ 改用 Go1（原因見上方說明） |
| P0-3 行走 / P0-4 站起 | ✅ 環境和訓練流程完成，待在 5060 上實際訓練驗收 |
| P0-5 checkpoint / 接續訓練 / seed | ✅ 已完成 |
| P0-6 YAML 設定 | ✅ 已完成（打錯欄位名稱會直接報錯） |
| P0-7 TensorBoard 分項 reward | ✅ 已完成 |
| P0-8 播放、鍵盤控制、mp4 | ✅ 程式完成（雲端沒有 OpenGL，畫面輸出需要在你的電腦上實測） |
| P1-1 導航避障 | ✅ 環境完成，待實際訓練 |
| P1-2 domain randomization | ✅ 已完成（walk / getup） |
| P1-3 課程學習 | ⬜ 尚未做 |
| P1-4 記憶體不足提示 | ✅ 已完成（OOM 時提示建議的 num_envs） |
