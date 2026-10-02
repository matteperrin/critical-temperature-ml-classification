# Phase 2 code notes

## Shared test folds

The main issue was that the raw and preprocessed eLCS runs were not testing on the same rows. Preprocessing removed rows before splitting, so using the same random seed did not keep the folds the same.

The raw dataset has 21,263 rows and 66 exact duplicates. Removing those duplicates and rebuilding the folds changed the test-row assignments in every fold.

The fix is kept small:

- All four model runners use `model_folds()` with the same raw data.
- Preprocessed eLCS removes duplicates from the training rows only. The test rows stay unchanged.
- The duplicate check includes `critical_temp`, so different temperature measurements are not removed just because they have the same binary label.
- `--data preprocessed` no longer loads the CSV produced by `preprocess.py`. The new results go to `reports/elcs_training_dedup_<iterations>_iterations.csv`, leaving the old preprocessed results alone.
- The new test checks that all models get the same test rows, cleaning only affects training rows, and scaling is fitted on training data only. It also checks that different temperatures in the same class are kept.

There is no saved fold file. The raw data, row order, split settings and library version need to stay the same between runs. The groups only cover identical feature vectors, not every repeated composition or material family.

## Code still to do

- **Improved eLCS:** set up a clearly separate improved configuration or pipeline. Keep the raw baseline, preprocessed baseline and improved system separate. Use training-side validation to choose changes, not the final evaluation folds.
- **Statistical test:** add a suitable test for comparing the models. Fold means and standard deviations alone do not cover this requirement.
- **Results:** make the outputs easier to compare. The conventional models save ROC-AUC and average precision, while eLCS saves confusion counts. Also, the current `pr_auc` column is average precision, not trapezoidal PR-AUC.
- **Tests:** add coverage for the standalone preprocessing script and rule export. The shared-fold test uses fake models and a test scaler, so it checks the pipeline rather than model performance or old result files.
- **Rule export:** let `extract_elcs_rules.py` use the final model configuration. It currently trains on all the raw data, so keep that separate from held-out evaluation.

There is no need to add another conventional model or write eLCS from scratch for these fixes. Keep any new experiment results separate from the old CSVs.
