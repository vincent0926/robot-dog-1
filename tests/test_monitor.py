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
def server(tmp_path):
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
