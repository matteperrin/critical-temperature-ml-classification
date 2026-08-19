# Project Phase I Requirements

**Course:** ENGE707  
**Due:** 31 August 2026 (extended from 23 August; submission time was not specified in the update)  
**Value:** 30 marks / 30%

## Schedule update

- Project Assessment 1 group feedback meetings will be held during the Week 6 lab instead of Week 5.
- All group members must attend the same Week 6 lab session for the feedback meeting.

Phase I covers the data-engineering and exploratory-analysis foundation of the project. Machine-learning model development and evaluation are not required during this phase.

## 1. Problem definition and dataset selection

- Define the real-world problem and analytical objective.
- Explain the motivation, stakeholders and practical relevance.
- Introduce the UCI Superconductivity dataset.
- Record the source, access date, licence and supporting documentation.
- Explain how the dataset satisfies the project requirements.
- Provide evidence of lecturer approval.

### Project objective

Investigate whether the elemental and compositional features of known superconductors can identify materials with a critical temperature above 77 K.

The 77 K threshold is approximately the operating temperature available through liquid-nitrogen cooling. Materials above this threshold may therefore be candidates for more accessible cryogenic operation than lower-temperature superconductors.

## 2. Data acquisition, inspection and documentation

All practical work must be completed using Python.

- Load `train.csv` and, where relevant, `unique_m.csv`.
- Confirm the number of records and features.
- Inspect column names, data types and variable meanings.
- Check value ranges and unique values.
- Check for missing, blank, infinite and invalid values.
- Identify exact and feature-level duplicate records.
- Examine duplicated chemical formulas and their recorded critical temperatures.
- Identify constant and near-zero-variance features.
- Document assumptions made when interpreting the variables.
- Ensure the data-acquisition process is reproducible.

## 3. Data cleaning and transformation

- Address missing, invalid or inconsistent values.
- Investigate and appropriately handle duplicate records.
- Correct inappropriate data types and formats.
- Examine unusual zero and negative values.
- Investigate potential outliers.
- Justify whether outliers are retained, transformed, capped or removed.
- Retain the original `critical_temp` variable for exploratory analysis.
- Create the binary classification target:

| Class | Definition | Interpretation |
| --- | --- | --- |
| `0` | `critical_temp <= 77` | Below the liquid-nitrogen threshold |
| `1` | `critical_temp > 77` | Potentially liquid-nitrogen-compatible |

- Produce a cleaned dataset or provide clear reproduction instructions.

## 4. Exploratory data analysis and visualisation

Analyse:

- The distribution of critical temperature.
- The number and proportion of records in each 77 K class.
- Feature distributions, skewness and outliers.
- Relationships between individual features and critical temperature.
- Differences in feature distributions between the two classes.
- Pearson and Spearman correlations where appropriate.
- Highly correlated and potentially redundant features.
- The relationship between compositional complexity and critical temperature.
- Common elements and compositions when using `unique_m.csv`.
- Missingness patterns and other data-quality concerns.

Potential visualisations include:

- Histograms
- Boxplots
- Scatter plots
- Class-distribution plots
- Correlation heatmaps
- Missing-value plots
- Pair plots for selected variables
- PCA visualisations where appropriate

## Deliverables

- Python source code or Jupyter Notebook
- Cleaned dataset or clear dataset-reproduction instructions
- Data-pipeline diagram
- Progress report of 2,500–3,000 words

## Not required in Phase I

- Learning Classifier System implementation
- Classification model training
- Model comparison
- Performance metrics
- Hyperparameter tuning
- Explainable AI
- Statistical hypothesis testing
- Conclusions about model performance

## GitHub project requirements

- Create only one GitHub repository/project for the group.
- Add every group member as a collaborator on the shared repository.
- Each member must commit directly to the shared repository; do not create separate branches for individual members.
- Each student must make at least two commits per week.
- Commit history will be used to monitor individual contributions and participation.
- All members must contribute actively and maintain a consistent record of their progress.

## Individual-work requirement

The shared-repository requirements do not replace the individual assessment requirements. All group members use the same approved dataset and may discuss the shared problem, analytical objective and possible methods. Each student must independently:

- Write and run their own Python code.
- Perform their own data cleaning and analysis.
- Create their own visualisations.
- Conduct and interpret their own statistical tests when required.
- Explain and justify their own decisions.
- Write their own report.

Completed code, written explanations, interpretations and assessment responses must not be copied or shared between group members.

## Academic-integrity requirement

Large language models may assist with Python suggestions, error explanations and debugging. They must not be used to write the project report, interpret results, produce the discussion, complete the critical reflection or generate the conclusion. All code must be understood, checked and run by the submitting student.

## JUST TO REMEMBER

No code in final report
