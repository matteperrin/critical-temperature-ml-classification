# Critical Temperature Machine-Learning Classification

An end-to-end data engineering and machine-learning project using the UCI Superconductivity dataset to classify known superconductors according to whether their critical temperature exceeds 77 K—the approximate threshold for liquid-nitrogen cooling.

## Read the results

**[Reader-friendly results page](results.html)**: the main findings, a model
comparison chart, metric explanations, three rule examples and links to evidence.

Download or clone this repository, then double-click **`results.html`** to open it
in your browser. It works offline, without Python, a server or any installation.
GitHub's file viewer shows HTML source rather than rendering the page; open the
file from your downloaded copy instead.

## Research question

> Can the elemental and compositional features of known superconductors be used to identify materials with a critical temperature above 77 K?

## Significance

The critical temperature (`Tc`) is the temperature below which a material becomes superconducting. Nitrogen boils at approximately 77 K at standard atmospheric pressure, so superconductors with a `Tc` above this threshold may be candidates for operation using liquid-nitrogen-based cooling.

This threshold is practically relevant because liquid nitrogen is generally more accessible than the colder cryogenic systems required by conventional low-temperature superconductors. A reliable classification model could support early-stage material screening by identifying compositions that warrant further experimental investigation.

The classification does not establish that a material is commercially viable. Practical performance also depends on factors including operating margin, critical current, applied magnetic field, mechanical properties, manufacturability, stability and cost.

## Dataset

**Source:** [UCI Machine Learning Repository — Superconductivty Data](https://archive.ics.uci.edu/dataset/464/superconductivty+data)  
**DOI:** [10.24432/C53P47](https://doi.org/10.24432/C53P47)  
**Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)  
**Original source:** SuperCon database, National Institute for Materials Science, Japan

The UCI dataset contains:

- 21,263 known superconductors
- 81 numerical input features
- One continuous target, `critical_temp`
- `train.csv`, containing engineered compositional features and critical temperature
- `unique_m.csv`, containing elemental quantities, critical temperature and chemical formula

The features describe aggregate properties of each material, including the number of constituent elements and statistical summaries of elemental properties. Common summaries include:

- Mean and weighted mean
- Geometric mean and weighted geometric mean
- Entropy and weighted entropy
- Range and weighted range
- Standard deviation

These summaries are calculated for properties such as atomic mass, atomic radius, density, electron affinity, electronegativity, thermal conductivity and valence.

### Setup

Use **Python 3.13** (the tested runtime). Other Python versions have not been
verified with these pinned dependencies. From the repository root, create a
virtual environment and install the recorded dependencies. eLCS is installed from the bundled `third_party/scikit-eLCS`
source; other dependencies (and any required build tools) still need access to
a package index or a local cache. This is not a fully offline setup:

```bash
python -m venv .venv
# Activate .venv using the command for your shell, then run:
python -m pip install -r requirements.txt
```

### Fetching the data

Run the acquisition script from the repository root:

```bash
python src/critical_temperature/fetch_data.py
```

The script uses only the Python standard library to download the original UCI dataset 464 archive and extract both source files:

- `data/raw/train.csv` — 81 engineered features and the `critical_temp` target
- `data/raw/unique_m.csv` — elemental quantities, `critical_temp` and chemical formula

The files are excluded from Git because they can be reproduced by rerunning the
script. An internet connection is required. Before replacing either CSV, the
script checks both files against the SHA-256 hashes and byte sizes in
[`data/source_manifest.json`](data/source_manifest.json). A source change fails
without replacing the existing CSVs; investigate it rather than bypassing the
check. Successful downloads record their UTC acquisition time in
`data/raw/acquisition.json`. The original acquisition date of the existing data
is unknown; the manifest records when its local identity was documented, not a
new download. See [`data/README.md`](data/README.md) for details.

### Reproducing the Phase I pipeline

Run each stage from the repository root:

```bash
python src/critical_temperature/fetch_data.py
python src/critical_temperature/data_inspection.py
python src/critical_temperature/data_cleaning.py
python src/critical_temperature/data_transformation.py
python src/critical_temperature/data_analysis.py
```

Generated Phase I outputs are recreated under `data/processed/`,
`reports/tables/` and `reports/figures/`; these outputs are excluded from Git.
The final analysis step regenerates the quality/cleaning summaries, class balance,
descriptive statistics, repeated-formula review, IQR outlier counts, signed
Pearson/Spearman correlations, near-zero-variance assessment, class-conditional
feature summaries, compositional-complexity summaries, and element/composition
frequencies. It also generates distribution, correlation and composition figures.
See [data-directory instructions](data/README.md) for rebuilding them.
`target_boxplot.png` is an obsolete local artifact, not a pipeline output; use
`critical_temperature_boxplot.png`. The scripts do not remove old unrelated files.
These full-dataset analyses are exploratory, not model feature selection.
Phase II experiment records under `reports/` remain versioned.

### Loading Phase II inputs

After installing the dependencies, use this read-only loader from the repository
root. It loads model inputs without training or evaluating a model:

```python
from src.critical_temperature.model_data import load_model_data

X, y, groups = load_model_data()
```

By default, it reads the raw `train.csv` and keeps every row. It leaves
`critical_temp`, `above_77k`, and `material` out of the predictors so the model
cannot use the target or a material identifier as a shortcut. `groups` puts
identical feature vectors together for group-aware validation, but it does not
catch every repeated material or material family. Check those separately before
choosing the final split. Fit scaling and feature selection on training folds
only, and use the same folds for every model.

Run the loader and baseline-runner tests with:

```bash
python -m unittest discover -s tests -v
```

Git must be available on `PATH` for the evidence-checkout regression test.
`.gitattributes` preserves the historical holdout artifacts' CRLF checkout bytes
on every OS so their recorded SHA-256 hashes remain valid. Binary rule exports
are kept byte-for-byte. Do not normalize these artifacts or regenerate their
manifests to bypass integrity failures.

### scikit-eLCS implementation and bundled reference

We integrate **scikit-eLCS 1.2.4** rather than implementing the eLCS algorithm
from scratch. Our code supplies the superconductivity-data loader and the
cross-validation/reporting pipeline around the library, with one documented
local repair to its majority-class fallback. Historical baseline CSVs predate
that repair; see [Matte's notes](notes/matte_notes.md).

See [the integration note](notes/elcs_integration.md) for implementation details,
recorded baseline results, attribution and limitations. A complete copy of the
supplied upstream folder is preserved in [`third_party/scikit-eLCS/`](third_party/scikit-eLCS/),
including its [README](third_party/scikit-eLCS/README.md),
[user guide](third_party/scikit-eLCS/eLCS%20User%20Guide.ipynb) and
[GPL-3.0 licence](third_party/scikit-eLCS/LICENSE). `requirements.txt` installs
this bundled source as the `skeLCS` package used by the runner. Rerun
`python -m pip install -r requirements.txt` from the repository root to switch
an existing environment to the bundled source.
Upstream exports and saved models are not our project's experiment results.

### Development CV and a separate final holdout

`run_holdout.py` provides a separate workflow; the existing full-data CV runners
below are unchanged. It reserves the first of five stratified group folds
(seed 42) as an approximately 20% test set, then runs five-fold grouped CV only
within the remaining approximately 80%. Groups are identical feature vectors;
they do not guarantee separation of all related material families. Group sizes
mean the proportions are approximate, not an exact row-level 80/20 split.

Run development CV from the repository root after installing dependencies and
fetching the data:

```bash
python src/critical_temperature/run_holdout.py cv --model elcs_dedup --iterations 1000 --output-dir reports/holdout/elcs-dedup-1000
```

Available models are `elcs`, `elcs_dedup`, `ensemble`, `ensemble_dedup`,
`logistic_regression`, `svm` and `random_forest`. Ensemble member seeds default to
`11 42 73`; change these with `--seeds` during CV only. `elcs_dedup` and
`ensemble_dedup` remove exact source-row duplicates only from each training
partition, including the continuous temperature in the duplicate check. Other
eLCS configurations use raw training rows. Validation rows are never removed.
The ensemble uses majority voting and three times the single model's total
iteration budget at equal per-member settings. Logistic Regression and SVM fit
scaling within their training partitions. Use `--data PATH` for a different raw dataset and a new output
directory for each configuration. Matching comparisons require the same data
and split environment.

For eLCS, `--population-size` defaults to `100` and `--library-variant` defaults
to `corrected`. Use `unmodified` for a supplied-library baseline. Both variants
load isolated bundled source; see [runtime provenance](notes/elcs_integration.md#corrected-versus-supplied-implementation).

CV writes `config.json`, `splits.csv`, `cv_results.csv`, `cv_predictions.csv`
and a completion record to the experiment directory. eLCS experiments also freeze
implementation provenance, including source fingerprints, so version `1.2.4`
alone cannot conceal a changed implementation. Original zero-based row
positions identify predictions; they never enter the predictors. No reserved
rows are fitted or predicted during this stage. Choose and freeze the final
configurations using development CV results **before viewing any final scores**.
Then explicitly run final evaluation:

```bash
python src/critical_temperature/run_holdout.py test --experiment reports/holdout/elcs-dedup-1000
```

The final stage accepts no model-setting overrides. It validates the saved
configuration, CV artifacts, dataset and split environment, trains on the whole
development partition, and saves `test_results.csv` and `test_predictions.csv`.
CV refuses existing experiment directories. Final testing refuses repeated or
previously interrupted runs; `test_started.json` records that evaluation began.
Do not edit saved configuration/artifacts or remove this marker to tune against
final results. A failed final run needs investigation rather than automatic retry.

**Limitation:** earlier full-data analyses and CV already used these records.
This workflow separates future development from testing; it does not make the
reserved rows retroactively unseen for earlier model choices. An independent
new dataset would be needed to remove that historical exposure.

### Bounded corrected-library development study

Run the study in a new directory (`elcs_development_v1` contains the completed
study; see [findings](notes/matte_notes.md#development-study-progress)):

```bash
python src/critical_temperature/run_development_experiments.py --output-dir reports/holdout/elcs_development_v2
```

The driver records the plan before running matched unmodified baselines and a
corrected raw baseline at 1,000 iterations / population 100. It compares corrected
training-dedup eLCS at all four combinations of iterations 1,000 / 10,000 and
population 100 / 1,000. Highest mean development balanced accuracy selects the
ensemble budget; ties prefer fewer iterations, then smaller population.

Results go to `development_summary.csv`; selection and progress are recorded in
JSON. `study_complete.json` appears only when all eight experiments finish.
Existing directories are refused, and interrupted studies are not automatically
resumed. No final holdout is evaluated. Development selection does not establish
unbiased performance, statistical significance or convergence.

### Current Phase II results: completed exploratory revision

Start with the [reader-friendly results page](results.html) and the
[current technical record](notes/exploratory_revision.md). Supporting records are
under `reports/holdout/elcs_exploratory_revision_v2/`, including all six development
candidates, selection, descriptive CV/evaluation tables, fitted rules and completion
records. The page presents the revised single eLCS and ensemble alongside three
unchanged conventional-model references.

The revision selected 339,280 record updates per fit/member and population limit
1,000 using development CV. Its evaluation reused previously viewed records:
**these results are exploratory, not an untouched holdout or external validation**.
No new confirmatory significance test applies to the revised pair. Do not tune
against these evaluation scores or remove scoring start markers.

Reproducing the complete revision is an expensive, explicit training operation.
After installing dependencies and fetching the data, use a **new** directory
(the existing completed directory is refused):

```bash
python src/critical_temperature/run_exploratory_revision.py --output-dir reports/holdout/elcs_exploratory_revision_reproduction
```

This runs the entire fixed grid serially, selects using development CV, trains the
matched ensemble and scores the selected pair once on reused evaluation records.
It requires the committed conventional reference artifacts. Interrupted runs are
not automatically resumed. It does not replace the frozen original or revised
records, and does not establish independent generalization.

The HTML builder below only presents the committed `elcs_exploratory_revision_v2`
evidence; it does not automatically switch to a reproduction directory.

### Historical Phase II comparisons and rules

The [original technical interpretation](notes/phase_two_results.md) and
`reports/holdout/final_comparison/` describe the earlier seven-system comparison,
including its original statistical test. Those configurations used only 1,000
record presentations for selected eLCS training, not epochs. They remain historical
evidence, not the current page's results. The original significance test must not
be applied to the revised models. Original and revised scores are preserved.

`compare_models.py table` combines compatible completed experiments using
`--stage cv` or `--stage test`, `--experiments` and a new `--output` path. It checks
saved hashes, dataset identity, rows, labels and folds; it never trains models.
The `stats` command is restricted to the declared corrected deduplicated
single-versus-ensemble settings, 19,999 draws and seed 42. Its output is numeric;
the assumptions and significance interpretation are in the results note.

Final eLCS evaluations automatically export the exact fitted populations and
training-row provenance, including separate files for each ensemble member.
Validate and list existing exports without fitting or scoring again:

```bash
python src/critical_temperature/extract_elcs_rules.py --experiment reports/holdout/elcs_development_v1/corrected_ensemble_dedup_selected
```

The rule exporter's old all-data fitting command is no longer supported. Training
rule accuracy is not held-out accuracy. Run project tests with the unittest command
above; optional pytest users should use `python -m pytest tests -q`, not vendor-wide
discovery. The GitHub Actions workflow is configured to run the offline project
suite and `pip check` on Python 3.13 for Linux and Windows. It does not download the dataset or retrain the studies.
Historical fresh-install verification is recorded in the original results note;
see that note for its scope and limits.

To rebuild the HTML page from the saved evidence (Python standard library only):

```bash
python src/critical_temperature/build_results_page.py
```

This verifies saved artifact hashes and replaces only `results.html`. It does not
train models, evaluate the holdout or require the raw dataset. The checked-in page
is ready to view; rebuilding is only needed after an intentional presentation edit.

<details>
<summary>Historical experiments</summary>

### Running the initial Phase II baseline

The four baseline runners share the fold recipe in `model_data.model_folds()` and use
unchanged raw rows for evaluation. `run_elcs.py --data preprocessed` now removes
exact full-row duplicates from **training folds only**; it does not load the
globally cleaned CSV. These runs save to
`reports/elcs_training_dedup_<iterations>_iterations.csv`, leaving historical
preprocessed results untouched. Matching folds across runs require the same raw
data, row order, split settings and library version. See
[results and interpretation](notes/phase_two_results.md) for evaluation limitations.

From the repository root, after installing dependencies and fetching the raw data:

```bash
python src/critical_temperature/run_elcs.py
```

This legacy runner imports the installed `skeLCS` package, so its implementation
depends on what was installed. The historical CSVs were generated before the
fallback repair; do not overwrite them with corrected results. Prefer the
provenance-recording development workflow above for new experiments.

This evaluates eLCS on raw `train.csv` using all five
`StratifiedGroupKFold` folds. The loader creates the `critical_temp > 77` label,
removes target/identifier columns from the inputs, and groups identical feature
vectors. No scaling, feature selection or deduplication is applied. These groups
do not guarantee separation of repeated compositions or material families.

The initial configuration uses 100 learning iterations, a population-size limit
of 100, and random seed 42 for both splitting and each fresh fold model. This is
a small initial training budget, not a tuned or convergence-validated setup.

The runner defaults to 1,000 learning iterations. To test another budget while
keeping all other settings unchanged, pass a positive integer:

```bash
python src/critical_temperature/run_elcs.py --iterations 10000
```

Results are saved to `reports/elcs_raw_<iterations>_iterations.csv` (for example,
`reports/elcs_raw_10000_iterations.csv`). Repeating a budget overwrites that
budget's file; other budgets and the original `reports/elcs_raw_baseline.csv`
are preserved.
It records configuration, fold sizes, accuracy, balanced accuracy, precision,
recall and F1, followed by unweighted fold means and sample standard deviations.
New runs also record per-fold true negatives, false positives, false negatives
and true positives, with class `1` (above 77 K) as positive. These counts make
class-specific errors available for the recorded 81.7% / 18.3% class split;
count columns are left blank in the mean/std rows.
Precision, recall and F1 use class `1` (above 77 K) as positive and return zero
when undefined. Fold standard deviations are not confidence intervals or
significance tests. The runner tests use a stand-in estimator to stay fast and
are included in the unittest command above.

The historical increase from 100 to 1,000 iterations changed mean balanced
accuracy from 0.412 to 0.428 and recall from 0.447 to 0.439. These results predate
the fallback repair and do not establish convergence.

</details>

## Analytical objective

The original dataset supports regression using the numerical critical temperature. This project transforms the target into a binary classification label:

| Class | Definition | Interpretation |
| --- | --- | --- |
| `0` | `critical_temp <= 77` | Below the liquid-nitrogen threshold |
| `1` | `critical_temp > 77` | Potentially liquid-nitrogen-compatible |

The original `critical_temp` column is retained for exploratory analysis and label generation, but removed from the model inputs to prevent target leakage.

## Project stages

### 1. Data acquisition and reproducibility

- Download or programmatically retrieve the dataset from UCI.
- Record the source, licence, access date and file versions.
- Preserve the raw files unchanged.
- Provide reproducible instructions for rebuilding processed datasets.

### 2. Data inspection and validation

- Confirm record and feature counts.
- Inspect column names, data types and value ranges.
- Check for missing, blank, infinite and invalid values.
- Identify exact and feature-level duplicate records.
- Compare duplicated chemical formulas and their recorded temperatures.
- Identify constant and near-zero-variance features.
- Document assumptions used when interpreting the variables.

### 3. Data cleaning and transformation

- Resolve duplicates and invalid records using documented rules.
- Investigate unusual zero, negative and extreme values.
- Retain, transform, cap or remove outliers only with a justified rule.
- Generate the binary 77 K target.
- Separate predictors, target and identifying information.
- Apply scaling or transformation where required by the selected models.
- Fit all learned preprocessing operations using training data only.

### 4. Exploratory data analysis

- Summarise the distribution of critical temperature.
- Measure the balance of the two target classes.
- Examine feature distributions, skewness and outliers.
- Compare feature distributions between the two classes.
- Calculate Pearson and Spearman correlations.
- Identify highly correlated and potentially redundant features.
- Explore relationships between compositional complexity and temperature.
- Examine common elements and combinations using `unique_m.csv`.
- Use PCA or other dimensionality-reduction visualisations where appropriate.

### 5. Feature engineering and selection

- Remove identifiers and leakage-prone variables.
- Assess redundant weighted and unweighted feature pairs.
- Compare filter, embedded and model-based feature-selection methods.
- Consider scaling, power transformations and dimensionality reduction.
- Keep preprocessing inside the validation pipeline.

### 6. Model development

- Establish a simple baseline classifier.
- Implement the required Learning Classifier System.
- Compare it with at least three suitable non-deep-learning approaches.
- Candidate comparison models include logistic regression, support vector machines, random forests and gradient boosting.
- Tune models without using the held-out test set.

### 7. Evaluation

- Use stratified training, validation and test partitions.
- Apply cross-validation to model selection and tuning.
- Report the confusion matrix, precision, recall, F1 score and ROC-AUC.
- Include PR-AUC and balanced accuracy if the classes are imbalanced.
- Compare models using consistent folds and preprocessing.
- Examine false positives and false negatives in the context of material screening.

### 8. Explainability

- Compare global feature importance across models.
- Use coefficients, permutation importance or SHAP where appropriate.
- Examine individual predictions cautiously.
- Treat model explanations as statistical associations rather than causal physical relationships.

### 9. Limitations and responsible use

- The dataset contains known superconductors and cannot determine whether an arbitrary material is superconducting.
- A predicted class does not confirm physical performance or material feasibility.
- The 77 K boundary simplifies a continuous target and creates an abrupt distinction between otherwise similar materials.
- A material normally requires operating headroom below its measured `Tc`.
- Critical current and critical magnetic field are not captured by the classification target.
- Database selection effects, repeated material families and measurement variation may affect generalisation.
- Predictions require experimental validation before practical use.

## Project schedule and collaboration

- The Project Phase I deadline has been extended to **31 August 2026**.
- Project Assessment 1 group feedback meetings will take place in the Week 6 lab; all group members must attend the same lab session.
- The group must use one shared GitHub repository with every member added as a collaborator.
- Members must commit directly to the shared repository rather than use separate individual branches.
- Each student must make at least two commits per week; commit history will be used to monitor individual contribution and participation.
- Shared repository use does not remove the requirement for each student to complete their own assessed code, analysis, visualisations and report.

## Reproducibility principles

- Raw data remains unchanged.
- Cleaning decisions are implemented in code rather than manual spreadsheet edits.
- Random seeds and dataset splits are recorded.
- Training and test data are kept separate throughout preprocessing.
- Generated datasets and figures can be reproduced from the source files.
- Package versions are recorded in `requirements.txt` or an equivalent environment file.

## Attribution

Hamidieh, K. (2018). *Superconductivty Data* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C53P47](https://doi.org/10.24432/C53P47)

This repository is an academic project. All submitted analysis, interpretation and written assessment content must comply with the applicable AUT academic-integrity and assessment requirements.
