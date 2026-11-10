"""把訓練結果發佈成固定的網址（GitHub Pages），不用開著電腦也能看。

用法（在訓練的電腦上，專案資料夾裡）：
  python -m robodog.publish

做的事：
  1. 把 runs/ 匯出成靜態網站（和 python -m robodog.monitor 看到的是同一個畫面，只是沒有「即時」的部分）
  2. 推到這個 repo 的 gh-pages 分支（不會動你的工作資料夾和其他分支）
  3. 網址：https://<帳號>.github.io/<repo 名>/

第一次要到 GitHub 的 Settings → Pages，把 Source 設成「Deploy from a branch」，Branch 選 gh-pages、資料夾選 /(root)。

不同電腦可以各自發佈：網站上已經有的訓練會保留，只更新這台電腦有的那幾個。
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from robodog.monitor import export_static


def _git(*args, cwd=None, check=True, capture=True):
  r = subprocess.run(["git", *args], cwd=cwd, capture_output=capture, text=True)
  if check and r.returncode != 0:
    raise RuntimeError(f"git {' '.join(args)} 失敗：\n{(r.stderr or r.stdout).strip()}")
  return r


def pages_url(remote: str) -> str | None:
  m = re.search(r"github\.com[:/]([^/]+)/([^/.]+?)(?:\.git)?$", remote.strip())
  return f"https://{m.group(1)}.github.io/{m.group(2)}/" if m else None


def publish(runs: Path, keep: int = 30, branch: str = "gh-pages", message: str | None = None, dry_run: bool = False) -> str | None:
  remote = _git("config", "--get", "remote.origin.url").stdout.strip()
  if not remote:
    raise RuntimeError("找不到 git remote origin。請在專案資料夾裡執行。")
  name = _git("config", "user.name", check=False).stdout.strip() or "robodog"
  email = _git("config", "user.email", check=False).stdout.strip() or "robodog@users.noreply.github.com"

  tmp = Path(tempfile.mkdtemp(prefix="robodog-pages-"))
  try:
    site = tmp / "site"
    # 先把現有的網站抓下來（保留別台電腦發佈過的訓練）；還沒有這個分支就從空的開始
    r = _git("clone", "--depth", "1", "--branch", branch, remote, site.as_posix(), check=False)
    if r.returncode != 0:
      print(f"[publish] 遠端還沒有 {branch} 分支，從頭建立")
      site.mkdir(parents=True)
      _git("init", "-q", cwd=site)
    wrote = export_static(runs, site, keep=keep)
    print(f"[publish] 這台電腦匯出 {len(wrote)} 個訓練")
    for w in wrote[:8]:
      print(f"          {w}")
    # 每次都用全新的一個 commit 覆蓋（不留歷史），影片很大，保留歷史會讓 repo 越來越肥
    _git("checkout", "-q", "--orphan", "publish-tmp", cwd=site)
    _git("add", "-A", cwd=site)
    size = sum(f.stat().st_size for f in site.rglob("*") if f.is_file() and ".git" not in f.parts)
    print(f"[publish] 網站大小 {size / 2**20:.1f} MB")
    if dry_run:
      print(f"[publish] --dry-run：不推送。匯出的內容在 {site}")
      return None
    _git("-c", f"user.name={name}", "-c", f"user.email={email}", "commit", "-q", "-m",
         message or "更新訓練監看快照", cwd=site)
    # 臨時 repo 的 remote 要指到真正的 repo；用 URL 直接推，不改任何設定
    _git("push", "--force", remote, f"publish-tmp:{branch}", cwd=site, capture=False)
    url = pages_url(remote)
    print(f"\n[publish] 完成。網址：{url}\n[publish] （第一次要到 GitHub 的 Settings → Pages 啟用，1～2 分鐘後才看得到）")
    return url
  finally:
    shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None) -> None:
  p = argparse.ArgumentParser(description="把訓練結果發佈到 GitHub Pages")
  p.add_argument("--runs", default="runs", help="訓練輸出的根目錄")
  p.add_argument("--keep", type=int, default=30, help="網站上最多保留幾個訓練")
  p.add_argument("--branch", default="gh-pages")
  p.add_argument("--message", default=None, help="commit 訊息")
  p.add_argument("--dry-run", action="store_true", help="只匯出和檢查，不推送")
  args = p.parse_args(argv)
  try:
    publish(Path(args.runs), args.keep, args.branch, args.message, args.dry_run)
  except RuntimeError as e:
    print(f"[publish] {e}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
  main()
