"""Time NetPolicy.act_batch_full for 12 games (what one fleet decision costs) with the installed torch."""
import sys, time
import torch
from dotasim.env.worldstate import parse
from dotasim.train.policy import NetPolicy
raw = open("tests/golden/worldstate_6965.bin", "rb").read()
ws = parse(raw)
pol = NetPolicy(sys.argv[1] if len(sys.argv) > 1 else "runs/bc_v2/policy.pt", greedy=False, threads=4)
items = [(i, 2, ws) for i in range(12)]
for _ in range(5): pol.act_batch_full(items)
t = time.perf_counter()
for _ in range(20): pol.act_batch_full(items)
print(f"torch {torch.__version__} threads {torch.get_num_threads()} act_batch_full(12): {(time.perf_counter()-t)/20*1000:.1f} ms")
