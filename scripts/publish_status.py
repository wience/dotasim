"""Publish a compact live snapshot of a run for the hosted status page (web/index.html).

Default channel: an orphan branch `live` of this repository holding one file, live.json, rewritten in place
(commit --amend + force push) every N seconds, so the branch never grows. The page reads it through the GitHub
contents API. Usage (Windows or WSL, git must be able to push):
    python scripts/publish_status.py runs/sp_v12 [--every 60] [--branch live]
Alternative channel: a gist, `--gist <id>` (needs `gh` logged in)."""
import argparse
import json
import subprocess
import time
from pathlib import Path

KEEP = ("iteration", "skipped", "steps", "ends", "outcomes", "mean_last_hits", "mean_denies", "last_hits_radiant", "last_hits_dire", "mean_return", "kl_anchor", "entropy",
        "act_ms", "speed_x", "update_wall_s", "wall_s", "eval", "error")
REPO = Path(__file__).resolve().parents[1]


DROP = ("traceback", "error_samples")


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def snapshot(run: Path) -> dict:
    out = {"run": run.name, "published": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "iterations": []}
    st = _read_json(run / "run" / "status.json")
    if st:
        out["status"] = st
    cfg = _read_json(run / "config.json")
    if cfg:
        out["config"] = cfg
    maps = {}
    for f in sorted((run / "run").glob("ws_*.json")):
        m = _read_json(f)
        if m:
            maps[f.stem.split("_")[1]] = m
    if maps:
        out["maps"] = maps
    log = run / "log.jsonl"
    if log.exists():
        for line in log.read_text().splitlines()[-200:]:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            out["iterations"].append({k: v for k, v in d.items() if k not in DROP})
    return out


def git(*args, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def branch_worktree(branch: str) -> Path:
    """A worktree on the orphan branch, created on first use (runs/ is gitignored)."""
    wt = REPO / "runs" / f"publish_{branch}"
    if not (wt / ".git").exists():
        git("worktree", "prune", cwd=REPO)
        if git("rev-parse", "--verify", f"refs/heads/{branch}", cwd=REPO).returncode != 0:
            git("fetch", "origin", branch, cwd=REPO)
            if git("rev-parse", "--verify", f"refs/remotes/origin/{branch}", cwd=REPO).returncode == 0:
                git("branch", branch, f"origin/{branch}", cwd=REPO)
        if git("rev-parse", "--verify", f"refs/heads/{branch}", cwd=REPO).returncode == 0:
            r = git("worktree", "add", str(wt), branch, cwd=REPO)
        else:
            r = git("worktree", "add", "--detach", str(wt), cwd=REPO)
            git("checkout", "--orphan", branch, cwd=wt)
            git("rm", "-rfq", ".", cwd=wt)
        if r.returncode != 0:
            raise SystemExit(f"git worktree add failed: {r.stderr}")
    return wt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--every", type=float, default=60)
    p.add_argument("--branch", default="live")
    p.add_argument("--gist", default=None)
    a = p.parse_args()
    run = Path(a.run)
    wt = None if a.gist else branch_worktree(a.branch)
    while True:
        snap = snapshot(run)
        if a.gist:
            Path("live.json").write_text(json.dumps(snap))
            r = subprocess.run(["gh", "gist", "edit", a.gist, "-f", "live.json", "live.json"], capture_output=True, text=True)
        else:
            git("worktree", "repair", str(wt), cwd=REPO)   # Windows git rewrites the worktree's gitdir pointer to a C:/ path; WSL git cannot read it
            (wt / "live.json").write_text(json.dumps(snap))
            git("add", "live.json", cwd=wt)
            has_commit = git("rev-parse", "--verify", "HEAD", cwd=wt).returncode == 0
            git("commit", "-q", *(["--amend"] if has_commit else []), "-m", f"live {snap['published']}", cwd=wt)
            r = git("push", "-q", "-f", "origin", f"HEAD:refs/heads/{a.branch}", cwd=wt)
        n = len(snap["iterations"])
        print(time.strftime("%H:%M:%S"), "pushed" if r.returncode == 0 else f"push failed: {r.stderr.strip()[:160]}",
              f"iterations={n}", f"last={snap['iterations'][-1].get('iteration') if n else None}", flush=True)
        time.sleep(a.every)


if __name__ == "__main__":
    main()
