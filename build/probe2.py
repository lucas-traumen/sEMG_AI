import json
import os
import subprocess
import sys
from pathlib import Path

print("python:", sys.version.split()[0])
inp = Path("/kaggle/input")
print("input exists:", inp.is_dir())
print("input children:", sorted(p.name for p in inp.iterdir()) if inp.is_dir() else None)
for p in sorted(inp.rglob("*.mat"))[:5]:
    print("mat:", p, p.stat().st_size)
print("n_mat_total:", len(list(inp.rglob("*.mat"))))
print("=== packages ===")
for name in ("numpy", "scipy", "sklearn", "pandas", "pyarrow", "yaml", "torch"):
    try:
        mod = __import__(name)
        print(f"  {name}={getattr(mod, '__version__', '?')}")
    except Exception as exc:
        print(f"  {name} MISSING ({type(exc).__name__})")
print("=== env hints ===")
for key in sorted(os.environ):
    if "KAGGLE" in key.upper():
        print(f"  {key}={os.environ[key][:120]}")
