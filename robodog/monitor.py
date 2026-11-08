"""訓練監看網站：在瀏覽器（電腦或手機）隨時看訓練進度。

用法：
  python -m robodog.monitor                 # 開在 http://localhost:8000
  python -m robodog.monitor --port 9000
  python -m robodog.monitor --no-video      # 不自動錄影

它會：
  - 每 5 秒讀一次 runs/ 底下所有訓練（TensorBoard 紀錄、status.json、checkpoint）
  - 訓練中每十幾秒就有即時數據（不用等評估），還有訓練紀錄（train.log）和電腦狀態（顯示卡、CPU、記憶體）
  - 有新的 checkpoint 就在背景用 CPU 錄一段 8 秒影片（不搶訓練用的顯示卡）
  - 訓練完成後自動跑一次正式評估（evaluate），結果顯示在網頁上

手機要看：手機和電腦連同一個 Wi-Fi，打開啟動時印出來的「區網網址」。
（WSL2 的話，需要先在 Windows 做連接埠轉送，見 README。）
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

STATIC_DIR = Path(__file__).resolve().parent / "monitor_static"
CKPT_CONFIG = "ppo_network_config.json"
DEFAULT_EPISODE_LENGTH = {"walk": 1000, "getup": 300, "navigate": 200}  # 和各環境 default_config 一致


# --------------------------------------------------------------------------- #
# 讀取訓練資料
# --------------------------------------------------------------------------- #

def _pid_alive(pid) -> bool:
  if not pid:
    return False
  try:
    os.kill(int(pid), 0)
  except (OSError, ValueError):
    return False
  return True


def _read_json(path: Path):
  try:
    return json.loads(path.read_text(encoding="utf-8"))
  except (OSError, ValueError):
    return None


def _checkpoint_steps(run: Path) -> list[int]:
  d = run / "checkpoints"
  if not d.is_dir():
    return []
  return sorted(int(c.name) for c in d.iterdir() if c.is_dir() and (c / CKPT_CONFIG).exists())


class TBCache:
  """TensorBoard 紀錄讀起來不便宜，用檔案修改時間當快取鍵。"""

  def __init__(self):
    self._cache: dict[str, tuple[float, dict]] = {}
    self._lock = threading.Lock()

  def scalars(self, run: Path) -> dict[str, list[tuple[int, float]]]:
    tb = run / "tb"
    if not tb.is_dir():
      return {}
    files = list(tb.glob("events.*"))
    if not files:
      return {}
    key = max(f.stat().st_mtime for f in files) + len(files)
    with self._lock:
      hit = self._cache.get(str(tb))
      if hit and hit[0] == key:
        return hit[1]
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(tb.as_posix(), size_guidance={"scalars": 0})
    ea.Reload()
    out = {t: [(int(e.step), float(e.value)) for e in ea.Scalars(t)] for t in ea.Tags()["scalars"]}
    with self._lock:
      self._cache[str(tb)] = (key, out)
    return out


def _is_our_trainer(pid, run: Path) -> bool:
  """pid 還活著，而且真的是這次訓練的 robodog.train。

  雲端或電腦重開機後 pid 會被別的程式重複使用，只看 pid 會把早就中斷的訓練誤判成「訓練中」。
  """
  if not _pid_alive(pid):
    return False
  try:
    cmd = Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
  except OSError:
    return True  # 沒有 /proc（例如 macOS）就只能相信 pid
  if "robodog.train" not in cmd:
    return False
  name = run.name.split("-", 2)[2] if run.name.count("-") >= 2 else ""
  return not name or name in cmd


def run_state(run: Path) -> dict:
  st = _read_json(run / "status.json") or {}
  state = st.get("state")
  if state in ("running", "starting") and not _is_our_trainer(st.get("pid"), run):
    state = "interrupted"  # 程式被關掉或當機，沒機會寫 status
  if state is None:  # 舊版訓練沒有 status.json
    state = "done" if (run / "summary.json").exists() else "interrupted"
  return {**st, "state": state}


def run_summary(run: Path, tb: TBCache) -> dict:
  import yaml

  cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8")) or {}
  st = run_state(run)
  sc = tb.scalars(run)
  rewards = sc.get("eval/episode_reward", [])
  total = st.get("total_steps") or int((cfg.get("ppo") or {}).get("num_timesteps", 0))
  last_step = max([st.get("step") or 0] + [s for s, _ in rewards])
  return {
      "task": cfg.get("task", run.parent.name),
      "name": run.name,
      "state": st["state"],
      "step": last_step,
      "total_steps": total,
      "reward": rewards[-1][1] if rewards else None,
      "live_reward": st.get("live_reward"),
      "eval_passed": (_read_json(run / "eval.json") or {}).get("passed"),
      "sps": st.get("sps"),
      "device": st.get("device"),
      "updated": st.get("updated") or (run / "config.yaml").stat().st_mtime,
  }


def _downsample(series: list, n: int) -> list:
  if len(series) <= n:
    return series
  stride = len(series) / n
  out = [series[int(i * stride)] for i in range(n)]
  return out[:-1] + [series[-1]]  # 一定保留最新一點


def run_detail(run: Path, tb: TBCache, jobs: "Worker | None") -> dict:
  import yaml

  cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8")) or {}
  st = run_state(run)
  sc = tb.scalars(run)

  def series(tag):
    return {s: v for s, v in sc.get(tag, [])}

  reward = series("eval/episode_reward")
  steps = sorted(reward)
  length = series("eval/avg_episode_length")
  components = {}
  for tag in sc:
    if tag.startswith("eval/episode_reward/") and not tag.endswith("_std"):
      name = tag.split("/", 2)[2]
      s = series(tag)
      components[name] = [s.get(x) for x in steps]

  prog = _read_json(run / "videos" / "progress.json") or {}
  clips = {c["step"]: c for c in prog.get("clips", [])}
  ckpts = _checkpoint_steps(run)

  points = []
  for x in steps:
    c = clips.get(x, {})
    points.append({
        "step": x,
        "reward": reward[x],
        "length": length.get(x),
        "video": c.get("video"),
        "fwd": c.get("fwd_speed"),
        "falls": c.get("falls"),
    })

  ep_len = (cfg.get("env") or {}).get("episode_length") or DEFAULT_EPISODE_LENGTH.get(cfg.get("task"))
  ppo = cfg.get("ppo") or {}
  total = st.get("total_steps") or int(ppo.get("num_timesteps", 0))
  sps_series = sc.get("training/sps", [])
  sps = st.get("sps") or (sps_series[-1][1] if sps_series else None)
  last_step = max([st.get("step") or 0] + steps) if steps or st.get("step") else 0
  eta_min = None
  if st["state"] == "running" and sps and total > last_step:
    eta_min = (total - last_step) / sps / 60

  live = _downsample(sc.get("episode/sum_reward", []), 300)
  live_len = dict(sc.get("episode/length", []))

  return {
      "task": cfg.get("task", run.parent.name),
      "name": run.name,
      "status": st,
      "live": [{"step": s_, "reward": v, "length": live_len.get(s_)} for s_, v in live],
      "live_every": max(total // 200, 1) if total else None,
      "step": last_step,
      "total_steps": total,
      "sps": sps,
      "eta_min": eta_min,
      "num_envs": ppo.get("num_envs"),
      "episode_length": ep_len,
      "points": points,
      "components": components,
      "checkpoints": ckpts,
      "pending_videos": [c for c in ckpts if c not in clips],
      "eval": _read_json(run / "eval.json"),
      "job": jobs.job_for(run) if jobs else None,
      "video_error": jobs.video_error if jobs else None,
  }


class SystemStats:
  """電腦的即時狀態：顯示卡（nvidia-smi）、CPU、記憶體、硬碟。結果快取 3 秒，很多人同時看也不會變慢。"""

  def __init__(self, root: Path):
    self.root = root
    self._lock = threading.Lock()
    self._cache = (0.0, None)
    self._cpu_prev = None
    self._has_smi = None

  def get(self) -> dict:
    with self._lock:
      t, data = self._cache
      if data is not None and time.time() - t < 3:
        return data
      data = {"cpu": self._cpu(), "mem": self._mem(), "disk": self._disk(), "gpus": self._gpus(),
              "cores": os.cpu_count(), "time": time.time()}
      self._cache = (time.time(), data)
      return data

  def _cpu(self):
    try:
      with open("/proc/stat") as f:
        vals = [int(x) for x in f.readline().split()[1:]]
    except OSError:
      return None
    idle, total = vals[3] + (vals[4] if len(vals) > 4 else 0), sum(vals)
    prev, self._cpu_prev = self._cpu_prev, (idle, total)
    if prev and total > prev[1]:
      return round(100 * (1 - (idle - prev[0]) / (total - prev[1])), 1)
    try:
      return round(min(100.0, os.getloadavg()[0] / (os.cpu_count() or 1) * 100), 1)
    except OSError:
      return None

  @staticmethod
  def _mem():
    try:
      info = {}
      with open("/proc/meminfo") as f:
        for line in f:
          k, v = line.split(":", 1)
          info[k] = int(v.split()[0]) * 1024
      return {"used": info["MemTotal"] - info["MemAvailable"], "total": info["MemTotal"]}
    except (OSError, KeyError, ValueError):
      return None

  def _disk(self):
    try:
      st = os.statvfs(self.root)
      return {"free": st.f_bavail * st.f_frsize, "total": st.f_blocks * st.f_frsize}
    except OSError:
      return None

  def _gpus(self):
    if self._has_smi is False:
      return []
    q = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
      out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.TimeoutExpired):
      self._has_smi = False
      return []
    if out.returncode != 0:
      return []
    self._has_smi = True

    def num(x):
      try:
        return float(x)
      except ValueError:
        return None  # [N/A]

    gpus = []
    for line in out.stdout.strip().splitlines():
      f = [x.strip() for x in line.split(",")]
      if len(f) >= 6:
        gpus.append({"name": f[0], "util": num(f[1]), "mem_used": num(f[2]), "mem_total": num(f[3]),
                     "temp": num(f[4]), "power": num(f[5])})
    return gpus


def tail_log(run: Path, lines: int = 200) -> dict:
  f = run / "train.log"
  if not f.is_file():
    return {"lines": [], "size": 0}
  size = f.stat().st_size
  with open(f, "rb") as fh:
    fh.seek(max(0, size - 64 * 1024))
    text = fh.read().decode("utf-8", "replace")
  # jax/absl 的進度訊息常用 \r 覆寫同一行，只留最後一段
  # 不用 splitlines()：它把 \r 也當換行
  out = [ln.rstrip("\r").rsplit("\r", 1)[-1] for ln in text.rstrip("\n").split("\n")]
  if size > 64 * 1024:
    out = out[1:]  # 第一行可能被切一半
  return {"lines": out[-lines:], "size": size}


def list_runs(root: Path) -> list[Path]:
  if not root.is_dir():
    return []
  runs = []
  for task_dir in root.iterdir():
    if task_dir.is_dir():
      runs += [r for r in task_dir.iterdir() if (r / "config.yaml").exists()]
  return sorted(runs, key=lambda r: r.name, reverse=True)


# --------------------------------------------------------------------------- #
# 背景工作：自動錄影、自動評估
# --------------------------------------------------------------------------- #

def _pick_gl() -> str | None:
  """找一個能用的離屏算圖方式（egl → osmesa），找不到就回傳 None。"""
  candidates = [os.environ["MUJOCO_GL"]] if os.environ.get("MUJOCO_GL") else ["egl", "osmesa"]
  probe = (
      "import mujoco;"
      "m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"1\"/></worldbody></mujoco>');"
      "r=mujoco.Renderer(m,16,16);d=mujoco.MjData(m);r.update_scene(d);r.render()"
  )
  for gl in candidates:
    env = {**os.environ, "MUJOCO_GL": gl, "PYOPENGL_PLATFORM": gl}
    try:
      ok = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, timeout=60).returncode == 0
    except subprocess.TimeoutExpired:
      ok = False
    if ok:
      return gl
  return None


class Worker(threading.Thread):
  """一次只做一件事：先補錄影片，再補做評估。全部用 CPU，不影響正在訓練的顯示卡。"""

  def __init__(self, root: Path, tb: TBCache, videos: bool, evals: bool, eval_episodes: int):
    super().__init__(daemon=True)
    self.root, self.tb = root, tb
    self.videos, self.evals, self.eval_episodes = videos, evals, eval_episodes
    self.current: dict | None = None
    self.video_error: str | None = None
    self.gl: str | None = None
    self.failed: dict[tuple[str, str], int] = {}  # 同一件事最多重試 2 次

  def job_for(self, run: Path):
    cur = self.current
    if cur and cur["run"] == str(run):
      return {"kind": cur["kind"], "since": cur["since"]}
    return None

  def _env(self):
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    if self.gl:
      env.update(MUJOCO_GL=self.gl, PYOPENGL_PLATFORM=self.gl)
    return env

  def _run(self, run: Path, kind: str, cmd: list[str]) -> None:
    key = (str(run), kind)
    if self.failed.get(key, 0) >= 2:
      return
    self.current = {"run": str(run), "kind": kind, "since": time.time()}
    log = run / "monitor.log"
    try:
      with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        f.flush()
        rc = subprocess.run(cmd, env=self._env(), stdout=f, stderr=subprocess.STDOUT,
                            cwd=Path.cwd()).returncode
      if rc != 0:
        self.failed[key] = self.failed.get(key, 0) + 1
    finally:
      self.current = None

  def run(self) -> None:
    if self.videos:
      self.gl = _pick_gl()
      if self.gl is None:
        self.video_error = "這台電腦沒辦法離屏算圖，所以不會自動錄影。WSL/Ubuntu 請執行：sudo apt install libosmesa6 libegl1"
        self.videos = False
      else:
        print(f"[monitor] 錄影使用 MUJOCO_GL={self.gl}", flush=True)
    while True:
      try:
        self._tick()
      except Exception:  # noqa: BLE001
        traceback.print_exc()
      time.sleep(15)

  def _tick(self) -> None:
    for run in list_runs(self.root):
      ckpts = _checkpoint_steps(run)
      if not ckpts:
        continue
      if self.videos:
        prog = _read_json(run / "videos" / "progress.json") or {}
        done = {c["step"] for c in prog.get("clips", [])}
        if any(c not in done for c in ckpts):
          self._run(run, "video", [sys.executable, "-m", "robodog.progress_video", str(run),
                                   "--only-new", "--impl", "jax"])
          return  # 一次只做一件，下一輪再看
      if self.evals and run_state(run)["state"] == "done":
        ev = _read_json(run / "eval.json")
        if ev is None or ev.get("step") != ckpts[-1]:
          self._run(run, "eval", [sys.executable, "-m", "robodog.evaluate", str(run),
                                  "--impl", "jax", "--episodes", str(self.eval_episodes)])
          return


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

def make_handler(root: Path, tb: TBCache, worker: "Worker | None", system: SystemStats | None = None):
  root = root.resolve()
  system = system or SystemStats(root)

  class Handler(BaseHTTPRequestHandler):
    server_version = "robodog-monitor"

    def log_message(self, *args):  # 不要每個請求都印一行
      pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
      self.send_response(code)
      self.send_header("Content-Type", ctype)
      self.send_header("Content-Length", str(len(body)))
      self.send_header("Cache-Control", "no-store")
      for k, v in (extra or {}).items():
        self.send_header(k, v)
      self.end_headers()
      if self.command != "HEAD":
        self.wfile.write(body)

    def _json(self, obj, code=200):
      self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                 "application/json; charset=utf-8")

    def _run_dir(self, task: str, name: str) -> Path | None:
      p = (root / task / name).resolve()
      if p.parent.parent != root or not (p / "config.yaml").exists():
        return None
      return p

    def do_HEAD(self):
      self.do_GET()

    def do_GET(self):
      url = urlparse(self.path)
      path = unquote(url.path)
      try:
        if path in ("/", "/index.html"):
          body = (STATIC_DIR / "index.html").read_bytes()
          return self._send(200, body, "text/html; charset=utf-8")
        if path == "/api/runs":
          runs, errors = [], []
          for r in list_runs(root):
            try:
              runs.append(run_summary(r, tb))
            except Exception as e:  # noqa: BLE001  一個壞掉的 run 不要拖垮整個列表
              errors.append(f"{r.parent.name}/{r.name}：{type(e).__name__}: {e}")
          job = worker.current if worker else None
          return self._json({
              "runs": runs, "errors": errors, "root": str(root), "now": time.time(),
              "job": {"run": Path(job["run"]).name, "kind": job["kind"]} if job else None,
          })
        if path == "/api/run":
          q = parse_qs(url.query)
          run = self._run_dir(q.get("task", [""])[0], q.get("name", [""])[0])
          if run is None:
            return self._json({"error": "找不到這次訓練"}, 404)
          return self._json(run_detail(run, tb, worker))
        if path == "/api/system":
          return self._json(system.get())
        if path == "/api/log":
          q = parse_qs(url.query)
          run = self._run_dir(q.get("task", [""])[0], q.get("name", [""])[0])
          if run is None:
            return self._json({"error": "找不到這次訓練"}, 404)
          try:
            n = max(1, min(int(q.get("lines", ["200"])[0]), 1000))
          except ValueError:
            n = 200
          return self._json(tail_log(run, n))
        if path in ("/manifest.webmanifest", "/icon.svg"):
          f = STATIC_DIR / path.lstrip("/")
          ctype = "application/manifest+json" if f.suffix == ".webmanifest" else "image/svg+xml"
          return self._send(200, f.read_bytes(), ctype)
        if path.startswith("/media/"):
          parts = path.split("/")  # /media/<task>/<name>/<file>
          if len(parts) != 5:
            return self._json({"error": "bad path"}, 400)
          run = self._run_dir(parts[2], parts[3])
          f = (run / "videos" / parts[4]).resolve() if run else None
          if not f or f.parent != run / "videos" or not f.is_file():
            return self._json({"error": "找不到影片"}, 404)
          return self._file(f)
        return self._json({"error": "not found"}, 404)
      except (BrokenPipeError, ConnectionResetError):
        pass
      except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        try:
          self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        except Exception:  # noqa: BLE001
          pass

    def _file(self, f: Path):
      """支援 Range，iPhone Safari 播影片一定要。"""
      size = f.stat().st_size
      ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
      rng = self.headers.get("Range")
      if rng and rng.startswith("bytes="):
        start_s, _, end_s = rng[6:].split(",")[0].partition("-")
        if start_s:
          start = int(start_s)
          end = min(int(end_s), size - 1) if end_s else size - 1
        else:  # bytes=-N：最後 N 個位元組
          start, end = max(0, size - int(end_s)), size - 1
        if start >= size or start > end:
          return self._send(416, b"", ctype, {"Content-Range": f"bytes */{size}"})
        with open(f, "rb") as fh:
          fh.seek(start)
          body = fh.read(end - start + 1)
        return self._send(206, body, ctype,
                          {"Content-Range": f"bytes {start}-{end}/{size}", "Accept-Ranges": "bytes"})
      return self._send(200, f.read_bytes(), ctype, {"Accept-Ranges": "bytes"})

  return Handler


def _lan_ip() -> str | None:
  try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
      s.connect(("192.168.0.1", 80))  # 不會真的送封包，只是讓系統選網卡
      return s.getsockname()[0]
  except OSError:
    return None


def main(argv=None) -> None:
  p = argparse.ArgumentParser(description="訓練監看網站")
  p.add_argument("--runs", default="runs", help="訓練輸出的根目錄")
  p.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = 區網其他裝置也能連；127.0.0.1 = 只有本機")
  p.add_argument("--port", type=int, default=8000)
  p.add_argument("--no-video", action="store_true", help="不要自動錄影")
  p.add_argument("--no-eval", action="store_true", help="訓練完成後不要自動評估")
  p.add_argument("--eval-episodes", type=int, default=128)
  args = p.parse_args(argv)

  root = Path(args.runs).resolve()
  root.mkdir(parents=True, exist_ok=True)
  tb = TBCache()
  worker = Worker(root, tb, videos=not args.no_video, evals=not args.no_eval,
                  eval_episodes=args.eval_episodes)
  worker.start()

  server = ThreadingHTTPServer((args.host, args.port), make_handler(root, tb, worker))
  print(f"[monitor] 讀取：{root}")
  print(f"[monitor] 本機：http://localhost:{args.port}")
  ip = _lan_ip()
  if args.host == "0.0.0.0" and ip:
    print(f"[monitor] 區網：http://{ip}:{args.port}  （手機連同一個 Wi-Fi 打開這個）")
  print("[monitor] 按 Ctrl+C 結束", flush=True)
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  main()
