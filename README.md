# Critical Temperature Machine-Learning Classification

An end-to-end data engineering and machine-learning project using the UCI Superconductivity dataset to classify known superconductors according to whether their critical temperature exceeds 77 K—the approximate threshold for liquid-nitrogen cooling.

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

Create a virtual environment and install the recorded dependencies:

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

The files are excluded from Git because they can be reproduced by rerunning the script. The script overwrites existing copies with the files from the current UCI archive, and an internet connection is required. See [`data/README.md`](data/README.md) for data-directory details.

### Reproducing the Phase I pipeline

Run each stage from the repository root:

```bash
python src/critical_temperature/fetch_data.py
python src/critical_temperature/data_inspection.py
python src/critical_temperature/data_cleaning.py
python src/critical_temperature/data_transformation.py
python src/critical_temperature/data_analysis.py
```

Generated datasets, tables and figures are excluded from Git and recreated under `data/processed/` and `reports/`.

## Analytical objective

The original dataset supports regression using the numerical critical temperature. For this project, the target will be transformed into a binary classification label:

| Class | Definition | Interpretation |
| --- | --- | --- |
| `0` | `critical_temp <= 77` | Below the liquid-nitrogen threshold |
| `1` | `critical_temp > 77` | Potentially liquid-nitrogen-compatible |

The original `critical_temp` column will be retained for exploratory analysis and label generation, but removed from the model inputs to prevent target leakage.

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
- Compare it with at least two suitable machine-learning approaches.
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
