import json
import os
import urllib.request
from pathlib import Path

URL = "https://dp.kaggle.net"
TOK = os.environ.get("KAGGLE_DATA_PROXY_TOKEN")

print("=== /etc/secrets/kaggle/api-v1-token (first 40 only) ===")
try:
    raw = Path("/etc/secrets/kaggle/api-v1-token").read_text().strip()
    print("  len:", len(raw), "prefix:", raw[:40])
except Exception as exc:
    print("  read failed:", exc)

USER_TOK = os.environ.get("KAGGLE_USER_SECRETS_TOKEN")
print("user token len:", len(USER_TOK or ""))


def fetch(auth_header: str, path: str) -> str:
    req = urllib.request.Request(f"{URL}{path}", headers={"Authorization": auth_header, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read()
            print(f"  GET {path} auth={auth_header[:18]}... -> {resp.status} bytes={len(body)}")
            return body[:400].decode("utf-8", "replace")
    except Exception as exc:
        body = getattr(exc, "read", lambda: b"")()
        print(f"  GET {path} auth={auth_header[:18]}... -> {type(exc).__name__}: {exc} {body[:300]!r}")
        return ""


print("=== anonymous ===")
fetch("", "/")
print("=== bearer etc/secrets ===")
try:
    raw = Path("/etc/secrets/kaggle/api-v1-token").read_text().strip()
    print("  body:", fetch(f"Bearer {raw}", "/"))
except Exception as exc:
    print("  ", exc)
print("=== bearer env proxy token ===")
print("  body:", fetch(f"Bearer {TOK}", "/api/v1/datasets/view/trilucas/semg-db6"))
print("=== bearer user secrets ===")
print("  body:", fetch(f"Bearer {USER_TOK}", "/api/v1/datasets/view/trilucas/semg-db6"))
