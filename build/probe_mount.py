"""Throwaway probe: where is the DB6 dataset mounted, and is this a GPU box?"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

print("cwd:", os.getcwd())
print("python:", sys.version)

input_dir = Path("/kaggle/input")
print("=== /kaggle/input tree (depth 3) ===")
if input_dir.is_dir():
    for p in sorted(input_dir.rglob("*")):
        rel = p.relative_to(input_dir)
        if len(rel.parts) > 3:
            continue
        kind = "d" if p.is_dir() else "f"
        size = "" if p.is_dir() else f" {p.stat().st_size}"
        print(f"  [{kind}] {rel}{size}")
else:
    print("  MISSING")

print("=== candidate mat roots ===")
for base in [input_dir, *[q for q in sorted(input_dir.glob("*")) if q.is_dir()]]:
    if not base.is_dir():
        continue
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and sub.name.startswith("DB6_s"):
            mats = list(sub.rglob("*.mat"))
            print(f"  base={base} sub={sub.name} n_mat={len(mats)} first={mats[0] if mats else None}")

print("=== disk ===")
for target in ("/kaggle/working", "/kaggle/input", "/tmp"):
    if Path(target).is_dir():
        u = shutil.disk_usage(target)
        print(f"  {target}: total={u.total/2**30:.1f}GiB free={u.free/2**30:.1f}GiB")

print("=== gpu ===")
try:
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=60)
    print(out.stdout or out.stderr)
except Exception as exc:  # noqa: BLE001
    print("nvidia-smi failed:", exc)

print("=== torch ===")
try:
    import torch

    print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))
        print("capability", torch.cuda.get_device_capability(0))
except Exception as exc:  # noqa: BLE001
    print("torch failed:", type(exc).__name__, exc)
