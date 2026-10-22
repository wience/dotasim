"""Policy network: unit-token transformer encoder, GRU memory, masked action heads."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from dotasim.model.features import HERO_DIM, K, N_TYPES, UNIT_DIM

N_DIRS, N_DISTS = 16, 4
NEG = float("-inf")


class Policy(nn.Module):
    def __init__(self, unit_dim: int = UNIT_DIM, hero_dim: int = HERO_DIM, d: int = 128, layers: int = 4,
                 hidden: int = 256, k: int = K, heads: int = 4):
        super().__init__()
        self.d, self.hidden, self.k = d, hidden, k
        self.unit_embed = nn.Sequential(nn.Linear(unit_dim, d), nn.ReLU(), nn.Linear(d, d))
        self.hero_embed = nn.Sequential(nn.Linear(hero_dim, d), nn.ReLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.gru = nn.GRUCell(2 * d, hidden)
        self.type_head = nn.Linear(hidden, N_TYPES)
        self.dir_head = nn.Linear(hidden, N_DIRS)
        self.dist_head = nn.Linear(hidden, N_DISTS)
        self.target_query = nn.Linear(hidden, d)   # pointer: dot product of a query with each unit token
        self.value_head = nn.Linear(hidden, 1)

    def initial_state(self, batch: int) -> Tensor:
        return torch.zeros(batch, self.hidden)

    def forward(self, units: Tensor, unit_mask: Tensor, hero: Tensor, h: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        empty = ~unit_mask.any(1)
        if empty.any():   # a dead or hidden hero yields no tokens; keep one dummy key so the encoder has something to attend to
            unit_mask = unit_mask.clone()
            unit_mask[empty, 0] = True
        tok = self.unit_embed(units)                                   # [B,K,d]
        enc = self.encoder(tok, src_key_padding_mask=~unit_mask)       # padded tokens are ignored as keys
        maskf = unit_mask.unsqueeze(-1).to(enc.dtype)
        pooled = (enc * maskf).sum(1) / maskf.sum(1).clamp(min=1.0)
        x = torch.cat([pooled, self.hero_embed(hero)], dim=-1)
        h = self.gru(x, h)
        q = self.target_query(h).unsqueeze(1)                           # [B,1,d]
        target = (enc * q).sum(-1) / self.d ** 0.5                      # [B,K]
        target = target.masked_fill(~unit_mask, NEG)
        logits = {"type": self.type_head(h), "dir": self.dir_head(h), "dist": self.dist_head(h),
                  "target": target, "value": self.value_head(h).squeeze(-1)}
        return logits, h


def masked_log_probs(logits: dict[str, Tensor], type_mask: Tensor) -> dict[str, Tensor]:
    out = {"type": torch.log_softmax(logits["type"].masked_fill(~type_mask, NEG), -1),
           "dir": torch.log_softmax(logits["dir"], -1), "dist": torch.log_softmax(logits["dist"], -1),
           "target": torch.log_softmax(logits["target"], -1)}
    return out


def sample(logits: dict[str, Tensor], type_mask: Tensor, greedy: bool = True) -> dict[str, int]:
    """One environment's action (batch index 0) as ints."""
    lp = masked_log_probs(logits, type_mask)
    out = {}
    for k in ("type", "dir", "dist", "target"):
        row = lp[k][0]
        if greedy:
            out[k] = int(row.argmax())
        else:
            out[k] = int(torch.multinomial(row.exp(), 1))
    return out
