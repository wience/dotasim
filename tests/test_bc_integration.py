from pathlib import Path

import pytest

from dotasim.env.channel import Action
from dotasim.env.worldstate import parse

DATA = Path(__file__).resolve().parents[1] / "data" / "sf_mirror_v1"


@pytest.mark.skipif(not DATA.exists(), reason="dataset not recorded")
def test_bc_learns_and_policy_acts(tmp_path, golden_ws_bytes):
    from dotasim.train.bc import train
    from dotasim.train.policy import NetPolicy

    result = train(data=DATA, out=tmp_path / "bc", epochs=5, batch=8, max_episodes=4, log_every=5)
    assert result["loss_type_first"] * 0.7 > result["loss_type_last"], result
    assert result["acc_type"] > result["majority_rate"], result
    ckpt = tmp_path / "bc" / "policy.pt"
    assert ckpt.exists()
    pol = NetPolicy(ckpt, greedy=True)
    assert pol.frames == 5
    ws = parse(golden_ws_bytes)
    acts = pol.act({2: ws})
    assert isinstance(acts[2], Action) and acts[2].kind in ("none", "move", "attack", "cast")
    acts2 = pol.act({2: ws})  # a second call keeps a GRU state per team and still returns a valid action
    assert acts2[2].kind in ("none", "move", "attack", "cast")
