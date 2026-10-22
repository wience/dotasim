"""Time a WarcraftSim-sized policy forward pass on CPU: 48 unit tokens, width 128, 4 layers, GRU 256."""
import time

import torch

torch.set_num_threads(4)
d, units, layers = 128, 48, 4
enc = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d, 4, 2 * d, batch_first=True, dropout=0.0), layers)
gru = torch.nn.GRUCell(d, 256)
head = torch.nn.Linear(256, 7 + 64 + 64)
params = sum(p.numel() for m in (enc, gru, head) for p in m.parameters())
x, h = torch.randn(1, units, d), torch.zeros(1, 256)
with torch.inference_mode():
    for _ in range(5):
        head(gru(enc(x).mean(1), h))
    t = time.perf_counter()
    for _ in range(50):
        head(gru(enc(x).mean(1), h))
    one = (time.perf_counter() - t) / 50 * 1000
    xb = torch.randn(12, units, d)
    t = time.perf_counter()
    for _ in range(20):
        head(gru(enc(xb).mean(1), torch.zeros(12, 256)))
    batch = (time.perf_counter() - t) / 20 * 1000
print(f"torch {torch.__version__} threads={torch.get_num_threads()} params={params/1e6:.2f}M "
      f"single fwd {one:.2f} ms, batch-of-12 fwd {batch:.2f} ms ({batch/12:.2f} ms per instance)")
