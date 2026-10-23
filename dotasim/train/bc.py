"""Behavioral cloning: masked cross-entropy per head over windows of consecutive steps (truncated BPTT), with the GRU
state carried across the windows of an episode so training sees the states serving sees (bc_v1 restarted from zero
every 32 steps and served thousands of steps from one state)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from dotasim.model.labels import ATTACK, MOVE
from dotasim.model.net import Policy, masked_log_probs
from dotasim.train.dataset import EpisodeWindows

NEG = float("-inf")


def _gather(lp: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean negative log-prob of y over rows where valid; 0 when no row is valid."""
    if valid.sum() == 0:
        return lp.new_zeros(())
    picked = lp[valid].gather(1, y[valid].unsqueeze(1)).squeeze(1)
    picked = torch.where(torch.isfinite(picked), picked, picked.new_zeros(()))  # a label the mask forbids contributes 0
    return -picked.mean()


def window_losses(net: Policy, batch: dict[str, torch.Tensor], h: torch.Tensor | None = None
                  ) -> tuple[dict[str, torch.Tensor], dict[str, float], torch.Tensor]:
    B, T = batch["y_type"].shape
    if h is None:
        h = net.initial_state(B)
    tot = {"type": 0.0, "dir": 0.0, "dist": 0.0, "target": 0.0}
    correct_type = correct_target = n_target = 0
    for t in range(T):
        logits, h = net(batch["units"][:, t], batch["unit_mask"][:, t], batch["hero"][:, t], h)
        lp = masked_log_probs(logits, batch["type_mask"][:, t])
        y_type = batch["y_type"][:, t]
        is_move, is_attack = y_type == MOVE, y_type == ATTACK
        tot["type"] = tot["type"] + _gather(lp["type"], y_type, torch.ones_like(is_move))
        tot["dir"] = tot["dir"] + _gather(lp["dir"], batch["y_dir"][:, t].clamp(min=0), is_move)
        tot["dist"] = tot["dist"] + _gather(lp["dist"], batch["y_dist"][:, t].clamp(min=0), is_move)
        tot["target"] = tot["target"] + _gather(lp["target"], batch["y_target"][:, t].clamp(min=0), is_attack)
        correct_type += int((lp["type"].argmax(1) == y_type).sum())
        if is_attack.any():
            correct_target += int((lp["target"][is_attack].argmax(1) == batch["y_target"][:, t][is_attack]).sum())
            n_target += int(is_attack.sum())
    losses = {k: v / T for k, v in tot.items()}
    stats = {"acc_type": correct_type / (B * T), "acc_target": correct_target / max(n_target, 1), "n_target": n_target}
    return losses, stats, h


def evaluate(net: Policy, ds: EpisodeWindows, batch: int) -> tuple[float, float]:
    """Type and target accuracy over every window of every episode (teacher-forced, state carried, no grad)."""
    ct = ca = nt = na = 0
    h = None
    with torch.inference_mode():
        for b, first in ds.streams(batch):
            B, T = b["y_type"].shape
            _, st, h = window_losses(net, b, None if first else h)
            ct += st["acc_type"] * B * T
            nt += B * T
            ca += st["acc_target"] * st["n_target"]
            na += st["n_target"]
    return ct / max(nt, 1), ca / max(na, 1)


def train(data: Path, out: Path, epochs: int = 8, batch: int = 16, lr: float = 3e-4, max_episodes: int | None = None,
          log_every: int = 20, threads: int = 8) -> dict:
    """batch = episodes trained as parallel streams (16 of 46 gives 3 groups and about 370 updates per epoch)."""
    torch.set_num_threads(threads)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ds = EpisodeWindows(Path(data), max_episodes=max_episodes)
    stats = ds.label_stats()
    rng = np.random.default_rng(0)
    net = Policy()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    writer = SummaryWriter(str(out / "tb"))
    step, t0, first = 0, time.time(), None
    config = {"epochs": epochs, "batch": batch, "lr": lr, "window": ds.window, "frames": ds.frames, "episodes": len(ds.episodes)}
    acc_type = acc_target = 0.0
    for epoch in range(epochs):
        h = None
        for b, start in ds.streams(batch, rng):
            losses, st, h = window_losses(net, b, None if start else h)
            h = h.detach()
            loss = losses["type"] + losses["dir"] + losses["dist"] + losses["target"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            step += 1
            acc_type, acc_target = st["acc_type"], st["acc_target"]
            if first is None:
                first = float(losses["type"])
            if step % log_every == 0:
                for k, v in losses.items():
                    writer.add_scalar(f"loss/{k}", float(v), step)
                writer.add_scalar("acc/type", acc_type, step)
                writer.add_scalar("acc/target", acc_target, step)
                writer.add_scalar("steps_per_s", step * batch * ds.window / (time.time() - t0), step)
        torch.save({"state_dict": net.state_dict(), "frames": ds.frames, "hero_pool": ["npc_dota_hero_nevermore"],
                    "config": config, "epoch": epoch + 1}, out / "policy.pt")
        print(json.dumps({"epoch": epoch + 1, "loss_type": round(float(losses["type"]), 4), "acc_type": round(acc_type, 3),
                          "acc_target": round(acc_target, 3), "wall_s": round(time.time() - t0)}), flush=True)
    acc_type, acc_target = evaluate(net, ds, batch)
    writer.add_scalar("acc/type_full", acc_type, step)
    writer.add_scalar("acc/target_full", acc_target, step)
    writer.close()
    result = {"loss_type_first": first, "loss_type_last": float(losses["type"]), "acc_type": acc_type,
              "acc_target": acc_target, "majority_rate": stats["majority_rate"], "steps": stats["steps"],
              "type_counts": stats["type_counts"], "wall_s": round(time.time() - t0, 1), "updates": step}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description="Behavioral cloning from recorded Valve games.")
    p.add_argument("--data", default="data/sf_mirror_v1")
    p.add_argument("--out", default="runs/bc_v1")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=16, help="episodes per group of parallel streams")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-episodes", type=int, default=None)
    a = p.parse_args(argv)
    print(json.dumps(train(Path(a.data), Path(a.out), a.epochs, a.batch, a.lr, a.max_episodes), indent=1))


if __name__ == "__main__":
    main()
