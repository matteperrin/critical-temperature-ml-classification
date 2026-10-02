## Personal Logbook to record work done for Phase 2

## 27 September 2026 - Session 1

- Reviewed the Phase II requirements and project responsibilities.
- Revisited the Phase I problem definition and dataset.
- Reviewed the existing eLCS baseline and preprocessing implementation.
- Identified the remaining work for Tasks 1 and 2.

## 28 September 2026 – Session 2

- Implemented a Random Forest baseline for the conventional machine-learning comparison.
- Used the same `StratifiedGroupKFold` five-fold validation approach as the eLCS baseline.
- Added evaluation metrics including:
  - accuracy
  - balanced accuracy
  - precision
  - recall
  - F1-score
  - ROC-AUC
  - PR-AUC
- Ran the Random Forest experiment on the raw superconductivity dataset.
- Recorded fold balanced accuracies of:
  - Fold 1: 0.904
  - Fold 2: 0.910
  - Fold 3: 0.909
  - Fold 4: 0.918
  - Fold 5: 0.907
- Recorded a mean balanced accuracy of approximately **0.909**.
- Saved the results to `reports/random_forest_raw.csv`.
- Created `tests/test_random_forest.py` to test the Random Forest experiment.
- Verified that:
  - five separate folds are created
  - training and testing groups do not overlap
  - expected evaluation metrics are saved
  - the results file contains the five folds plus mean and standard deviation
- Ran the full test suite and confirmed all five tests passed.
- Committed the Random Forest baseline, experiment results and validation tests to GitHub.

## 30 September 2026 – Session 3

- Pulled the latest group changes from GitHub.
- Reviewed the latest work completed by the group.
- Confirmed that the eLCS training budget had been increased to 1,000 iterations and made configurable.
- Confirmed that SVM work had also been started by another group member.
- Ran the complete updated test suite.
- Confirmed all seven tests passed, including:
  - dataset validation tests
  - eLCS fold tests
  - configurable iteration tests
  - Random Forest tests

## 1 October 2026 – Session 4

- Started work on my Task 7 responsibility: finding and exporting LCS rules.
- Reviewed the current `run_elcs.py` implementation.
- Created `extract_elcs_rules.py`.
- Added functionality to:
  - train an eLCS model
  - export the final learned rule population
  - save feature conditions and predicted classes
  - rank learned rules using accuracy, match count and fitness
  - save the top 20 rules for easier inspection
- Ran a 100-iteration rule extraction test.
- Exported 94 rules from the 100-iteration model.
- Ran the full 1,000-iteration rule extraction.
- Exported 98 rules from the 1,000-iteration model.
- Saved:
  - the full eLCS rule population
  - a ranked top-20 rule file for later interpretation
- Prepared the rule outputs for later use in the Explainable AI section, where at least three LCS rules will be analysed.
- Added personal logbook files for group members and prepared the latest rule outputs for GitHub.

## 2 October 2026

- Reviewed the latest group commits before continuing with the improved eLCS work.
- Confirmed the eLCS rule extraction pipeline and generated rule outputs for later interpretation.
- Attempted to pull the latest changes from the shared repository.
- The pull was blocked because my local `ayushman_notes.md` changes would have been overwritten, so these changes needed to be committed before syncing with the latest group work.