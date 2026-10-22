"""Regenerate features_6965.npz from the golden worldstate. Run after an intentional feature change."""
from pathlib import Path

import numpy as np

from dotasim.env.worldstate import parse
from dotasim.model.features import featurize

HERE = Path(__file__).parent

if __name__ == "__main__":
    f = featurize(parse((HERE / "worldstate_6965.bin").read_bytes()), 2)
    np.savez(HERE / "features_6965.npz", units=f.units, unit_mask=f.unit_mask, unit_handles=f.unit_handles,
             hero=f.hero, type_mask=f.type_mask)
    print("units", f.units.shape, "real units", int(f.unit_mask.sum()), "hero", f.hero.round(2))
