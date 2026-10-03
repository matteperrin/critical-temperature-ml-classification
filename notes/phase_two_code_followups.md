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

## Development CV and final test

`run_holdout.py` adds a separate workflow without changing the older runners. The first of five grouped, stratified folds is reserved for final testing. Five-fold CV runs only within the remaining roughly 80%.

Use the `cv` command to compare configurations, then the separate `test` command once the choices are fixed. Each experiment saves its settings, row assignments and predictions. Test runs reject changed settings or data and do not overwrite an earlier final run. A failed final run also leaves a marker so it needs checking before any retry.

The earlier full-data experiments already used these records. Reserving them now separates future runs, but does not make them unseen for decisions already made. See the README for commands.

## Code still to do

- **Improved eLCS:** finish the interrupted development grid and preprocessed ensemble comparison. See [progress](matte_notes.md#development-study-progress) and the [study command](../README.md#bounded-corrected-library-development-study).
- **Statistical test:** add a suitable test for comparing the models. Fold means and standard deviations alone do not cover this requirement.
- **Results:** the eLCS-only study writes `development_summary.csv`, but the full six-system comparison is still needed. The conventional models save ROC-AUC and average precision, while eLCS saves confusion counts. The legacy `pr_auc` column is average precision, not trapezoidal PR-AUC.
- **Tests:** add coverage for the standalone preprocessing script and rule export. The shared-fold test uses fake models and a test scaler, so it checks the pipeline rather than model performance or old result files.
- **Rule export:** let `extract_elcs_rules.py` use the final model configuration. It currently trains on all the raw data, so keep that separate from held-out evaluation.

There is no need to add another conventional model or write eLCS from scratch for these fixes. Keep any new experiment results separate from the old CSVs.
