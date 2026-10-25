"""One PPO update in a separate process, so the ROCm runtime's busy-waiting threads (two cores, even idle) live only
for the 20 s of the update instead of starving the twelve game servers during rollouts.

    python -m dotasim.train.gpu_learner <in.pt> <out.pt>

in.pt: {"state_dict", "anchor", "opt_state" | None, "windows": list[Window], "cfg": PPOConfig, "version", "device"}
out.pt: {"state_dict" (cpu), "opt_state", "stats"}
"""
from __future__ import annotations

import sys

import torch

from dotasim.model.net import Policy
from dotasim.train.ppo import ppo_update


def main(in_path: str, out_path: str) -> None:
    payload = torch.load(in_path, map_location="cpu", weights_only=False)
    dev = payload.get("device", "cuda")
    net, anchor = Policy().to(dev), Policy().to(dev)
    net.load_state_dict(payload["state_dict"])
    anchor.load_state_dict(payload["anchor"])
    opt = torch.optim.Adam(net.parameters(), lr=payload["cfg"].lr)
    if payload.get("opt_state"):
        opt.load_state_dict(payload["opt_state"])
    stats = ppo_update(net, anchor, payload["windows"], payload["cfg"], opt, version=payload.get("version"))
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "opt_state": opt.state_dict(), "stats": stats}, out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
