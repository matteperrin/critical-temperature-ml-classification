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

## Rebuilding processed datasets

After installing `requirements.txt` and acquiring the raw files, run from the
repository root:

```bash
python src/critical_temperature/data_cleaning.py
python src/critical_temperature/data_transformation.py
```

These scripts recreate `processed/train_clean.csv`, `processed/unique_m_clean.csv`
and `processed/train_transformed.csv`. The generated CSVs are excluded from Git.
This cleanup retains the local copies; a fresh checkout needs the commands above.
The scripts overwrite processed outputs, not the raw files.

For the recorded raw inputs and current dependencies, regeneration was verified
byte-for-byte against all three previously tracked CSVs. Phase II configurations,
splits, predictions, rules and completion records under `reports/` remain versioned.
