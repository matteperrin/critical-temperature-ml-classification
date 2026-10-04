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

Files under `raw/` are reproducible downloads and are excluded from Git. Do not
edit them manually. Keep cleaned and transformed data separate from these source
files.

## Source identity and acquisition provenance

[`source_manifest.json`](source_manifest.json) records SHA-256 hashes and byte
sizes of the existing source CSVs, the source URL, DOI and licence. Its identity
recording date is **not** an acquisition date: the original acquisition date is
unknown. No fresh network download is implied by that record.

The downloader validates both archive members against this manifest before
replacing either local CSV. Missing members or changed contents cause an error
without replacing the raw files. A successful download writes a UTC acquisition
time and observed hashes to `raw/acquisition.json`. An upstream change requires
investigation and an intentional manifest revision, not disabling validation.
The filesystem writes themselves are not a transactional two-file operation;
a disk/write failure still needs investigation.

## Rebuilding processed datasets

After installing `requirements.txt` and acquiring the raw files, run from the
repository root:

```bash
python src/critical_temperature/data_cleaning.py
python src/critical_temperature/data_transformation.py
```

These scripts recreate `processed/train_clean.csv`, `processed/unique_m_clean.csv`
and `processed/train_transformed.csv`. The generated CSVs are excluded from Git.
A fresh checkout needs the commands above. The scripts overwrite processed outputs,
not the raw files. Exact engineered-feature duplicates are retained in Phase I:
they may correspond to different material records in the paired composition file.
Potential outliers and valid zeros are not automatically removed.

## Rebuilding Phase I reports

After rebuilding the processed datasets, run:

```bash
python src/critical_temperature/data_inspection.py
python src/critical_temperature/data_analysis.py
```

Inspection prints raw-data checks. Analysis writes the seven original report tables
(`class_balance`, `cleaning_summary`, `data_quality_summary`,
`descriptive_statistics`, `feature_target_correlations`,
`repeated_formula_examples`, `top_iqr_outliers`) and additional tables for near-zero
variance, class-specific feature summaries, compositional complexity, element
prevalence and composition frequency, all as CSVs under `reports/tables/`.
It generates figures under `reports/figures/`. These tables and PNGs are ignored by
Git and regenerated for each checkout; obsolete unrelated files are not deleted.
In particular, old `target_boxplot.png` is superseded by
`critical_temperature_boxplot.png`.

The near-zero-variance criterion is a screening heuristic, not automatic removal:
constants are flagged separately; low-variation predictors have at most 10% unique
values and a most-common/second-most-common frequency ratio greater than 19.
Correlations retain their sign and include Pearson and Spearman coefficients.
All these full-dataset summaries are exploratory; any model feature selection
must remain inside training folds. Element prevalence counts positive quantities,
not physical suitability or causal importance. Analysis does not write assessed
interpretations or alter frozen model evidence.

For the recorded raw inputs and current dependencies, regeneration was verified
byte-for-byte against all three previously tracked CSVs. Phase II configurations,
splits, predictions, rules and completion records under `reports/` remain versioned.
