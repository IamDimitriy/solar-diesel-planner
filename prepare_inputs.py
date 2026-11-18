import argparse
import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--tables", action="store_true")
mode.add_argument("--weather", action="store_true")
args = parser.parse_args()
if args.tables:
    for path in (root / "model").rglob("seasonal-results.csv.gz"):
        path.with_suffix("").write_bytes(gzip.decompress(path.read_bytes()))
else:
    folder = root / "issledovanie-data"
    for row in json.loads((folder / "weather-manifest.json").read_text(encoding="utf-8")):
        with urlopen(row["url"], timeout=180) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"PVGIS returned a different snapshot: {row['point']}")
        (folder / row["file"]).write_bytes(data)
