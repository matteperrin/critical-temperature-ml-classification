from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

DATASET_URL = "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_FILES = ("train.csv", "unique_m.csv")
MANIFEST_PATH = PROJECT_ROOT / "data" / "source_manifest.json"


def download_data():
    """Validate both source identities before replacing any raw CSV files."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    with urlopen(DATASET_URL, timeout=60) as response:
        archive_data = response.read()

    with ZipFile(BytesIO(archive_data)) as archive:
        missing_files = set(DATA_FILES) - set(archive.namelist())
        if missing_files:
            missing = ", ".join(sorted(missing_files))
            raise RuntimeError(f"UCI archive is missing expected files: {missing}")
        contents = {name: archive.read(name) for name in DATA_FILES}

    identities = {}
    for name, content in contents.items():
        identity = {"sha256": sha256(content).hexdigest(), "size_bytes": len(content)}
        expected = manifest["files"][name]
        if any(identity[key] != expected[key] for key in identity):
            raise ValueError(
                f"Source identity/hash mismatch for {name}: got {identity}. "
                "Raw files were not replaced. Investigate upstream changes before "
                "intentionally updating data/source_manifest.json."
            )
        identities[name] = identity

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, content in contents.items():
        output_path = DATA_DIR / name
        output_path.write_bytes(content)
        paths.append(output_path)
    record = {"source_url": DATASET_URL,
              "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
              "files": identities}
    (DATA_DIR / "acquisition.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return paths


def main():
    for path in download_data():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"Saved {path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
