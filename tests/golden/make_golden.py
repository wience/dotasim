"""Regenerate golden JSON files from the golden worldstate. Run after an intentional change to summary()."""
import json
from pathlib import Path

from dotasim.env.worldstate import parse, summary

HERE = Path(__file__).parent

if __name__ == "__main__":
    ws = parse((HERE / "worldstate_6965.bin").read_bytes())
    s = summary(ws)
    (HERE / "worldstate_6965.json").write_text(json.dumps(s, indent=1))
    print(s)
