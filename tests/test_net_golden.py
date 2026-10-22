import time

import numpy as np
import torch

from dotasim.env.worldstate import parse
from dotasim.model.features import K, N_TYPES, featurize
from dotasim.model.net import Policy, masked_log_probs, sample


def _batch(f, b=1):
    return (torch.from_numpy(np.repeat(f.units[None], b, 0)), torch.from_numpy(np.repeat(f.unit_mask[None], b, 0)),
            torch.from_numpy(np.repeat(f.hero[None], b, 0)), torch.from_numpy(np.repeat(f.type_mask[None], b, 0)))


def test_policy_shapes_masks_and_speed(golden_ws_bytes):
    torch.set_num_threads(4)
    f = featurize(parse(golden_ws_bytes), 2)
    net = Policy()
    n = sum(p.numel() for p in net.parameters())
    assert 0.7e6 < n < 1.5e6, n
    units, mask, hero, tmask = _batch(f)
    with torch.inference_mode():
        logits, h = net(units, mask, hero, net.initial_state(1))
    assert logits["type"].shape == (1, N_TYPES) and logits["dir"].shape == (1, 16)
    assert logits["dist"].shape == (1, 4) and logits["target"].shape == (1, K) and h.shape == (1, 256)
    assert torch.isinf(logits["target"][0, ~mask[0]]).all() and torch.isfinite(logits["target"][0, mask[0]]).all()
    lp = masked_log_probs(logits, tmask)
    assert torch.isinf(lp["type"][0, ~tmask[0]]).all()
    a = sample(logits, tmask, greedy=True)
    assert set(a) == {"type", "dir", "dist", "target"} and tmask[0, a["type"]]
    units12, mask12, hero12, _ = _batch(f, 12)
    h12 = net.initial_state(12)
    with torch.inference_mode():
        for _ in range(3):
            net(units12, mask12, hero12, h12)
        t = time.perf_counter()
        for _ in range(10):
            net(units12, mask12, hero12, h12)
    ms = (time.perf_counter() - t) / 10 * 1000
    assert ms < 15, ms
