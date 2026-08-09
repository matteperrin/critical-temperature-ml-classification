from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/464/"
    "superconductivty+data.zip"
)
DATA_DIR = Path(__file__).parent / "data" / "raw"
DATA_FILES = ("train.csv", "unique_m.csv")


def download_data():
    """Download and extract the original UCI Superconductivity CSV files."""
    with urlopen(DATASET_URL, timeout=60) as response:
        archive_data = response.read()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with ZipFile(BytesIO(archive_data)) as archive:
        missing_files = set(DATA_FILES) - set(archive.namelist())
        if missing_files:
            missing = ", ".join(sorted(missing_files))
            raise RuntimeError(f"UCI archive is missing expected files: {missing}")

        paths = []
        for filename in DATA_FILES:
            output_path = DATA_DIR / filename
            output_path.write_bytes(archive.read(filename))
            paths.append(output_path)

    return paths


def main():
    for path in download_data():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"Saved {path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
