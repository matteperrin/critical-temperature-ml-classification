# Data directory

Run the acquisition script from the repository root:

```bash
python src/critical_temperature/fetch_data.py
```

It downloads the original archive for [UCI Superconductivity Data (dataset 464)](https://archive.ics.uci.edu/dataset/464/superconductivty+data) and extracts:

| File | Purpose |
| --- | --- |
| `raw/train.csv` | Engineered numerical features and the `critical_temp` target |
| `raw/unique_m.csv` | Elemental quantities, `critical_temp` and chemical formula |

Files under `raw/` are reproducible downloads and are excluded from Git. Do not edit them manually. Keep cleaned and transformed data separate from these source files.
