"""監看網站的 API 測試：用假的訓練資料夾，不需要真的訓練。"""

import json
import os
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
import yaml
from tensorboardX import SummaryWriter

from robodog import monitor


def _fake_run(root, name, *, state=None, pid=None, steps=(0, 1000, 2000), video_for=(1000,)):
  run = root / "walk" / name
  (run / "videos").mkdir(parents=True)
  (run / "config.yaml").write_text(yaml.safe_dump({"task": "walk", "ppo": {"num_timesteps": 4000, "num_envs": 8}}))
  w = SummaryWriter((run / "tb").as_posix())
  for i, s in enumerate(steps):
    w.add_scalar("eval/episode_reward", float(i), s)
    w.add_scalar("eval/avg_episode_length", 100.0 * (i + 1), s)
    w.add_scalar("eval/episode_reward/termination", -0.5 / (i + 1), s)
    w.add_scalar("eval/episode_reward/termination_std", 0.1, s)
  w.close()
  for s in steps[1:]:
    c = run / "checkpoints" / f"{s:012d}"
    c.mkdir(parents=True)
    (c / "ppo_network_config.json").write_text("{}")
  clips = [{"step": s, "reward": 1.0, "fwd_speed": 0.5, "falls": 0, "video": f"step_{s:012d}.mp4"} for s in video_for]
  for c in clips:
    (run / "videos" / c["video"]).write_bytes(bytes(range(256)) * 4)
  (run / "videos" / "progress.json").write_text(json.dumps({"clips": clips}))
  if state:
    (run / "status.json").write_text(json.dumps({"state": state, "pid": pid, "step": steps[-1],
                                                  "total_steps": 4000, "sps": 100.0, "updated": 1.0}))
  return run


@pytest.fixture()
def server(tmp_path, monkeypatch):
  # 假訓練用 pytest 自己的 pid 代表「活著」，這裡不檢查是不是 robodog.train
  monkeypatch.setattr(monitor, "_is_our_trainer", lambda pid, run: monitor._pid_alive(pid))
  _fake_run(tmp_path, "20260101-000000-alive", state="running", pid=os.getpid())
  _fake_run(tmp_path, "20250101-000000-dead", state="running", pid=999999)
  httpd = ThreadingHTTPServer(("127.0.0.1", 0), monitor.make_handler(tmp_path, monitor.TBCache(), None))
  threading.Thread(target=httpd.serve_forever, daemon=True).start()
  yield f"http://127.0.0.1:{httpd.server_address[1]}"
  httpd.shutdown()


def _get(url, headers=None):
  req = urllib.request.Request(url, headers=headers or {})
  try:
    with urllib.request.urlopen(req) as r:
      return r.status, r.headers, r.read()
  except urllib.error.HTTPError as e:
    return e.code, e.headers, e.read()


def test_runs_list_and_state(server):
  code, _, body = _get(server + "/api/runs")
  runs = {r["name"]: r for r in json.loads(body)["runs"]}
  assert code == 200
  assert runs["20260101-000000-alive"]["state"] == "running"
  assert runs["20250101-000000-dead"]["state"] == "interrupted"  # pid 不存在 → 被中斷
  assert runs["20260101-000000-alive"]["reward"] == 2.0


def test_run_detail(server):
  code, _, body = _get(server + "/api/run?task=walk&name=20260101-000000-alive")
  d = json.loads(body)
  assert code == 200
  assert [p["step"] for p in d["points"]] == [0, 1000, 2000]
  assert d["points"][1]["video"] == "step_000000001000.mp4" and d["points"][1]["fwd"] == 0.5
  assert d["pending_videos"] == [2000]
  assert set(d["components"]) == {"termination"}  # _std 要被濾掉
  assert d["eta_min"] == pytest.approx((4000 - 2000) / 100 / 60)
  assert d["episode_length"] == 1000


def test_video_range_and_traversal(server):
  url = server + "/media/walk/20260101-000000-alive/step_000000001000.mp4"
  code, headers, body = _get(url, {"Range": "bytes=10-19"})
  assert code == 206 and body == bytes(range(10, 20))
  assert headers["Content-Range"] == "bytes 10-19/1024"
  code, _, _ = _get(server + "/media/walk/20260101-000000-alive/..%2Fconfig.yaml")
  assert code in (400, 404)  # 擋下來就好
  code, _, _ = _get(server + "/api/run?task=walk&name=../../etc")
  assert code == 404


def test_index_served(server):
  code, headers, body = _get(server + "/")
  assert code == 200 and "Robodog 訓練監看".encode() in body


def test_system_stats(server):
  code, _, body = _get(server + "/api/system")
  d = json.loads(body)
  assert code == 200 and "gpus" in d and "cores" in d  # 沒有 NVIDIA 顯示卡時 gpus 是空清單


def test_log_tail(server, tmp_path):
  run = tmp_path / "walk" / "20260101-000000-alive"
  (run / "train.log").write_text("".join(f"line {i}\n" for i in range(500)) + "進度 10%\r進度 50%\n", encoding="utf-8")
  code, _, body = _get(server + "/api/log?task=walk&name=20260101-000000-alive&lines=3")
  d = json.loads(body)
  assert code == 200 and d["lines"] == ["line 498", "line 499", "進度 50%"]  # \r 覆寫的只留最後一段
  code, _, _ = _get(server + "/api/log?task=walk&name=../../etc")
  assert code == 404


def test_live_series(server, tmp_path):
  run = tmp_path / "walk" / "20260101-000000-alive"
  w = SummaryWriter((run / "tb").as_posix())
  for s in range(0, 4000, 100):
    w.add_scalar("episode/sum_reward", s / 1000, s)
  w.close()
  d = json.loads(_get(server + "/api/run?task=walk&name=20260101-000000-alive")[2])
  assert len(d["live"]) == 40 and d["live"][-1]["step"] == 3900
  assert d["live"][-1]["reward"] == pytest.approx(3.9)
  assert d["live_every"] == 20


def test_static_assets(server):
  assert _get(server + "/manifest.webmanifest")[0] == 200
  assert _get(server + "/icon.svg")[0] == 200


def test_reused_pid_is_not_running(tmp_path):
  # 自己（pytest）的 pid 活著，但不是 robodog.train → 應判斷為中斷
  run = _fake_run(tmp_path, "20260101-000000-x", state="running", pid=os.getpid())
  assert monitor.run_state(run)["state"] == "interrupted"


def test_export_static_merges_and_prunes(tmp_path):
  runs = tmp_path / "runs"
  out = tmp_path / "site"
  _fake_run(runs, "20260101-000000-a", state="done", pid=None)
  assert monitor.export_static(runs, out) == ["walk/20260101-000000-a"]
  d = json.loads((out / "api" / "run" / "walk" / "20260101-000000-a.json").read_text(encoding="utf-8"))
  assert d["exported"] > 0 and d["summary"]["name"] == "20260101-000000-a"
  assert (out / "media" / "walk" / "20260101-000000-a" / "step_000000001000.mp4").is_file()
  html = (out / "index.html").read_text(encoding="utf-8")
  assert "window.STATIC_MODE = true" in html and 'href="manifest.webmanifest"' in html and (out / ".nojekyll").exists()

  # 另一台電腦匯出另一個訓練到同一個資料夾：兩個都要在
  other = tmp_path / "runs2"
  _fake_run(other, "20260201-000000-b", state="done", pid=None)
  monitor.export_static(other, out)
  names = [r["name"] for r in json.loads((out / "api" / "runs.json").read_text(encoding="utf-8"))["runs"]]
  assert names == ["20260201-000000-b", "20260101-000000-a"]  # 新的在前

  # 只留最新的 keep 個：舊的連影片一起刪掉
  monitor.export_static(other, out, keep=1)
  names = [r["name"] for r in json.loads((out / "api" / "runs.json").read_text(encoding="utf-8"))["runs"]]
  assert names == ["20260201-000000-b"]
  assert not (out / "media" / "walk" / "20260101-000000-a").exists()
  assert not (out / "api" / "run" / "walk" / "20260101-000000-a.json").exists()


def test_publish_pages_url():
  from robodog import publish

  assert publish.pages_url("https://github.com/vincent0926/robot-dog-1.git") == "https://vincent0926.github.io/robot-dog-1/"
  assert publish.pages_url("git@github.com:vincent0926/robot-dog-1.git") == "https://vincent0926.github.io/robot-dog-1/"
  assert publish.pages_url("/some/local/path") is None
