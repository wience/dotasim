"""Compile the pinned Valve protos into Python modules.

Valve's .proto files carry custom options (boxed_type, synthetic_default,
additional_includes) that stock protoc rejects, so they are stripped first.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

STRIP = [
    (re.compile(r' \[boxed_type = "[A-Za-z_0-9]+"\]'), ""),
    (re.compile(r', boxed_type = "[A-Za-z_0-9]+"'), ""),
    (re.compile(r', synthetic_default = "[^"]*"'), ""),
    (re.compile(r"^option additional_includes.*$", re.M), ""),
]


def generate(proto_dir: Path) -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        names = []
        for src in sorted(proto_dir.glob("*.proto")):
            text = src.read_text()
            for pat, rep in STRIP:
                text = pat.sub(rep, text)
            (Path(tmp) / src.name).write_text(text)
            names.append(src.name)
        subprocess.run(
            [sys.executable, "-m", "grpc_tools.protoc", f"-I{tmp}", f"--python_out={proto_dir}", *names],
            check=True, capture_output=True, text=True,
        )
    out = proto_dir / "bot_script_pb2.py"
    # protoc emits an absolute import of the sibling module; make it relative so the package imports anywhere
    out.write_text(out.read_text().replace("\nimport valveextensions_pb2 as", "\nfrom . import valveextensions_pb2 as"))
    return out


if __name__ == "__main__":
    print(generate(Path(__file__).parent))
