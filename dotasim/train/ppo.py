"""PPO over GRU windows: clipped objective, value loss, entropy bonus, KL penalty to a frozen anchor."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

import torch

from dotasim.model.labels import ATTACK, MOVE
from dotasim.model.net import Policy, masked_log_probs
from dotasim.train.buffer import Buffer, Window


@dataclass
class PPOConfig:
    lr: float = 1e-4
    clip: float = 0.2
    epochs: int = 2   # 3 epochs took 231 s for a full iteration (budget 180 s); see plan Task 3
    minibatch: int = 64
    vf_coef: float = 0.5
    ent_coef: float = 0.003
    kl_coef: float = 0.02
    max_grad: float = 1.0
    gamma: float = 0.9995
    lam: float = 0.95


def joint_log_prob(lp: dict[str, torch.Tensor], a: dict[str, torch.Tensor]) -> torch.Tensor:
    """log pi(a|s) for a batch of rows: type, plus dir+dist when moving, plus target when attacking."""
    t = a["type"]
    out = lp["type"].gather(1, t.unsqueeze(1)).squeeze(1)
    move, attack = (t == MOVE).float(), (t == ATTACK).float()
    out = out + move * (lp["dir"].gather(1, a["dir"].unsqueeze(1)).squeeze(1) + lp["dist"].gather(1, a["dist"].unsqueeze(1)).squeeze(1))
    out = out + attack * lp["target"].gather(1, a["target"].unsqueeze(1)).squeeze(1)
    return out


FLOOR = -30.0   # masked (-inf) log-probs are clamped so 0 * -inf never reaches the backward pass


def _entropy(lp: torch.Tensor) -> torch.Tensor:
    lp = lp.clamp(min=FLOOR)
    return -(lp.exp() * lp).sum(-1)


def _kl(lp_new: torch.Tensor, lp_old: torch.Tensor) -> torch.Tensor:
    lp_new, lp_old = lp_new.clamp(min=FLOOR), lp_old.clamp(min=FLOOR)
    return (lp_new.exp() * (lp_new - lp_old)).sum(-1)


def evaluate_windows(net: Policy, batch: dict[str, torch.Tensor], anchor: Policy | None = None):
    """Teacher-forced pass through the GRU from the stored h0. Returns per-step logp, entropy, value, kl-to-anchor [B,T]."""
    B, T = batch["logp"].shape
    h = batch["h0"]
    ha = batch["h0"].clone() if anchor is not None else None
    logp, ent, val, kl = [], [], [], []
    for t in range(T):
        units, mask, hero, tmask = batch["units"][:, t], batch["unit_mask"][:, t], batch["hero"][:, t], batch["type_mask"][:, t]
        logits, h = net(units, mask, hero, h)
        lp = masked_log_probs(logits, tmask)
        a = {k: batch[f"a_{k}"][:, t] for k in ("type", "dir", "dist", "target")}
        logp.append(joint_log_prob(lp, a))
        ent.append(_entropy(lp["type"]))
        val.append(logits["value"])
        if anchor is not None:
            with torch.no_grad():
                la, ha = anchor(units, mask, hero, ha)
                lpa = masked_log_probs(la, tmask)
            kl.append(_kl(lp["type"], lpa["type"]) + _kl(lp["dir"], lpa["dir"]) + _kl(lp["dist"], lpa["dist"])
                      + _kl(lp["target"], lpa["target"]))
    stack = lambda xs: torch.stack(xs, 1)
    return stack(logp), stack(ent), stack(val), (stack(kl) if kl else torch.zeros(B, T))


def ppo_update(net: Policy, anchor: Policy, windows: list[Window], cfg: PPOConfig, opt: torch.optim.Optimizer,
               version: int | None = None) -> dict:
    t0 = time.time()
    dev = next(net.parameters()).device   # windows are collected on the CPU; the update runs wherever net lives (GPU learner)
    all_adv = torch.cat([w.adv[w.valid] for w in windows])
    mean, std = all_adv.mean(), all_adv.std().clamp(min=1e-6)
    stats = {"loss_pi": 0.0, "loss_v": 0.0, "entropy": 0.0, "kl_anchor": 0.0, "clipfrac": 0.0, "approx_kl": 0.0}
    n_batches = 0
    for epoch in range(cfg.epochs):
        order = list(range(len(windows)))
        random.shuffle(order)
        for i in range(0, len(order), cfg.minibatch):
            batch = Buffer.collate([windows[j] for j in order[i:i + cfg.minibatch]], expect_version=version)
            if dev.type != "cpu":
                batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            valid = batch["valid"]
            adv = ((batch["adv"] - mean) / std)[valid]
            logp, ent, val, kl = evaluate_windows(net, batch, anchor)
            ratio = torch.exp(logp - batch["logp"])[valid]
            loss_pi = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
            loss_v = ((val - batch["ret"]) ** 2)[valid].mean()
            entropy = ent[valid].mean()
            kl_a = kl[valid].mean()
            loss = loss_pi + cfg.vf_coef * loss_v - cfg.ent_coef * entropy + cfg.kl_coef * kl_a
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad)
            opt.step()
            with torch.no_grad():
                stats["loss_pi"] += float(loss_pi); stats["loss_v"] += float(loss_v); stats["entropy"] += float(entropy)
                stats["kl_anchor"] += float(kl_a)
                stats["clipfrac"] += float(((ratio - 1).abs() > cfg.clip).float().mean())
                stats["approx_kl"] += float((batch["logp"][valid] - logp[valid]).mean())
            n_batches += 1
    for k in stats:
        stats[k] /= max(n_batches, 1)
    stats.update({"batches": n_batches, "windows": len(windows), "steps": int(sum(int(w.valid.sum()) for w in windows)),
                  "wall_s": round(time.time() - t0, 1)})
    return stats
