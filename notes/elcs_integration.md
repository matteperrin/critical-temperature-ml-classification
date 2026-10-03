# scikit-eLCS integration

## What we implemented

This project integrates the existing **scikit-eLCS 1.2.4** library. We implemented
its superconductivity-data loading and evaluation pipeline, not the eLCS
algorithm from scratch. The legacy runners import `eLCS` from the installed
`skeLCS` package. The development/holdout runner now explicitly loads the bundled
implementation and records its provenance. The bundled source includes one local
majority-class fallback repair, documented in [Matte's notes](matte_notes.md).
Historical results used the supplied implementation before this repair.

- Dependency: [`requirements.txt`](../requirements.txt), installs the bundled version 1.2.4 from `./third_party/scikit-eLCS`.
- Runner: [`src/critical_temperature/run_elcs.py`](../src/critical_temperature/run_elcs.py).
- Data loader: [`src/critical_temperature/model_data.py`](../src/critical_temperature/model_data.py).
- Runner tests: [`tests/test_run_elcs.py`](../tests/test_run_elcs.py), using a stand-in estimator rather than training the real algorithm.

The runner classifies whether `critical_temp > 77 K` using the raw numerical
features. It evaluates five `StratifiedGroupKFold` folds, grouping identical
feature vectors, with random seed 42 and a population-size limit of 100.
Grouping does not guarantee separation of repeated compositions or material
families. No scaling, feature selection or deduplication is applied.

The default budget is 1,000 learning iterations per fold. From the repository
root, after installing requirements and fetching the data:

```bash
python src/critical_temperature/run_elcs.py
python src/critical_temperature/run_elcs.py --iterations 10000
```

Each run writes `reports/elcs_raw_<iterations>_iterations.csv`, recording fold
sizes, accuracy, balanced accuracy, precision, recall, F1 and confusion-matrix
counts, followed by fold means and sample standard deviations. Repeating a
budget overwrites that budget's report.

## Recorded baseline results

These are **historical unmodified-library results**, taken from the saved reports.
They have not been regenerated after the majority-class repair and must not be
overwritten or represented as corrected-library results. The legacy runner does
not record implementation hashes; use the development workflow for new runs.

| Learning iterations | Mean accuracy | Mean balanced accuracy | Mean recall | Mean F1 | Report |
| --- | --- | --- | --- | --- | --- |
| 100 | 0.389313 | 0.411878 | 0.447497 | 0.203868 | [Initial baseline](../reports/elcs_raw_baseline.csv) |
| 1,000 | 0.420969 | 0.428072 | 0.439281 | 0.217201 | [1,000-iteration baseline](../reports/elcs_raw_1000_iterations.csv) |

The model is integrated and baseline results are recorded, but it is not yet
tuned or convergence-validated. Both recorded mean balanced accuracies are
below 0.5. These small-budget runs do not establish the algorithm's best
achievable performance.

## Bundled upstream files

[`third_party/scikit-eLCS/`](../third_party/scikit-eLCS/) contains the supplied
`scikit-eLCS-master` folder with the documented local majority-class repair,
including source code, tests,
performance datasets, notebooks, images, generated exports and hidden files.
Its `setup.py` identifies the package as version **1.2.4**. No upstream commit
identifier was available in the supplied folder; the version alone does not
establish byte-for-byte equivalence with the published package.

The copy provides the eLCS dependency as well as reference material and
provenance. `requirements.txt` installs it locally, and the runner continues to
import `from skeLCS import eLCS` without any import-path changes. Run
`python -m pip install -r requirements.txt` from the repository root, including
in existing environments previously using the PyPI package. The relative path
is resolved from the working directory. Other dependencies and any required
build tools still need a package index or local cache; this is not a fully
offline setup. Upstream exports and saved models are
upstream artifacts, **not results from our superconductivity experiments**.
Treat serialized models as untrusted data; do not load them just to inspect the
copy.

## Corrected versus supplied implementation

The new development workflow identifies `corrected` and `unmodified` library
variants explicitly. The latter reproduces the supplied implementation by
reversing only the known fallback repair in an isolated runtime; it does not
modify the installed package or tracked bundled files. Internal imports use a
private namespace to prevent variant mixing. These in-memory models do not
support serialization through the temporary namespace.

Each experiment freezes bundled/effective source hashes, package metadata and
variant. Final evaluation rejects changed or missing provenance; version `1.2.4`
alone is insufficient. See the [README](../README.md#bounded-corrected-library-development-study)
for study commands and selection, and [Matte's notes](matte_notes.md#development-study-progress)
for results and unfinished work.

Useful upstream references:

- [Upstream project](https://github.com/UrbsLab/scikit-eLCS)
- [Bundled README](../third_party/scikit-eLCS/README.md)
- [Bundled user guide](../third_party/scikit-eLCS/eLCS%20User%20Guide.ipynb)
- [Core implementation](../third_party/scikit-eLCS/skeLCS/eLCS.py)
- [Package metadata](../third_party/scikit-eLCS/setup.py)
- [GPL-3.0 licence](../third_party/scikit-eLCS/LICENSE)

The bundled third-party code retains its upstream licence and attribution to
Robert Zhang and Ryan J. Urbanowicz. It is not original project code. Preserve
its licence and comply with the applicable GPL terms when redistributing it;
this note does not assign a licence to the rest of this repository.
