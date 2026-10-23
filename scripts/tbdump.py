"""Print scalar tags from a tensorboard run dir: python scripts/tbdump.py runs/bc_v1/tb loss/type"""
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator(sys.argv[1]); ea.Reload()
for tag in sys.argv[2:] or ea.Tags()["scalars"]:
    ev = ea.Scalars(tag)
    n = max(1, len(ev) // 12)
    print(tag, [(e.step, round(e.value, 3)) for e in ev[::n]])
