import os
import shutil
import subprocess
from pathlib import Path

print("=== /etc/secrets ===")
for p in sorted(Path("/etc/secrets").rglob("*")):
    print(f"  {'d' if p.is_dir() else 'f'} {p} {'' if p.is_dir() else p.stat().st_size}")

print("=== mounts touching kaggle/input or datasets ===")
try:
    out = subprocess.run(["mount"], capture_output=True, text=True, timeout=60).stdout
    for line in out.splitlines():
        if any(k in line for k in ("kaggle", "dataset", "input", "overlay")):
            print("  ", line[:200])
except Exception as exc:  # noqa: BLE001
    print("  mount failed:", exc)

print("=== disks ===")
for t in ("/kaggle/working", "/root/.cache", "/tmp", "/"):
    if Path(t).is_dir():
        u = shutil.disk_usage(t)
        print(f"  {t}: total={u.total/2**30:.1f}G free={u.free/2**30:.1f}G")

print("=== kagglehub ===")
try:
    import kagglehub

    print("  kagglehub", getattr(kagglehub, "__version__", "?"), kagglehub.__file__)
except Exception as exc:  # noqa: BLE001
    print("  import failed:", type(exc).__name__, exc)

print("=== /kaggle tree depth 2 ===")
for p in sorted(Path("/kaggle").rglob("*")):
    rel = p.relative_to("/kaggle")
    if len(rel.parts) <= 2:
        print(f"  {'d' if p.is_dir() else 'f'} {rel}")

print("=== token can we call the API? ===")
tok = os.environ.get("KAGGLE_DATA_PROXY_TOKEN", "")
print("  token len:", len(tok), "url:", os.environ.get("KAGGLE_DATA_PROXY_URL"))
