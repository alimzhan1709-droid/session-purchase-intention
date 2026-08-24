"""
fetch_data.py — download and verify the benchmark used by the thesis.

The dataset is public, so it is fetched rather than committed. The archive is
downloaded from the UCI Machine Learning Repository, the CSV is extracted and its
SHA-256 digest is compared with the digest recorded for the analysis, so that a
silently changed upstream file cannot pass unnoticed.

    python fetch_data.py
"""

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://archive.ics.uci.edu/static/public/468/"
       "online+shoppers+purchasing+intention+dataset.zip")
MEMBER = "online_shoppers_intention.csv"
EXPECTED_SHA256 = "b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=MEMBER, help="destination CSV path")
    a = ap.parse_args()
    dest = Path(a.out)

    if dest.exists():
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if digest == EXPECTED_SHA256:
            print(f"{dest} already present and verified.")
            return 0
        print(f"{dest} exists but its digest does not match; re-downloading.")

    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL) as response:
        archive = response.read()

    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        payload = zf.read(MEMBER)

    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SHA256:
        print("SHA-256 mismatch — the upstream file is not the one used for the analysis.",
              file=sys.stderr)
        print(f"  expected {EXPECTED_SHA256}", file=sys.stderr)
        print(f"  received {digest}", file=sys.stderr)
        return 1

    dest.write_bytes(payload)
    print(f"Wrote {dest} ({len(payload):,} bytes), SHA-256 verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
