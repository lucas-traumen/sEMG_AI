import json, os, shutil, sys, traceback
from pathlib import Path

print("=== A. attached inputs ===")
inp = Path("/kaggle/input")
print("exists:", inp.is_dir(), "children:", sorted(p.name for p in inp.iterdir()) if inp.is_dir() else None)

print("=== B. kagglehub dataset_download ===")
try:
    import kagglehub
    print("kagglehub", kagglehub.__version__)
    p = kagglehub.dataset_download("trilucas/semg-db6")
    print("path:", p)
    root = Path(p)
    mats = sorted(root.rglob("*.mat"))
    print("n_mat:", len(mats))
    for m in mats[:3]:
        print("  ", m.relative_to(root), m.stat().st_size)
    total = sum(m.stat().st_size for m in mats)
    print("total GiB:", round(total / 2**30, 3))
except Exception:
    traceback.print_exc()

print("=== C. cache size ===")
u = shutil.disk_usage("/root/.cache")
print("cache total GiB:", round(u.total / 2**30, 1), "free:", round(u.free / 2**30, 1))

print("=== D. torch ===")
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
except Exception:
    traceback.print_exc()
