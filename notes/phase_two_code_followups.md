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

## Completed code and evaluations

The development grid, shared-fold conventional baselines, seven frozen final
evaluations, validated comparison reports, paired statistical test and
configuration-linked rule exports are complete. Standalone preprocessing and
rule-export tests were added. All 66 project tests passed, including in a fresh
isolated environment. See [results and interpretation](phase_two_results.md) for
findings, three sample rules, verification limits and artifact links.

The corrected ensemble was significantly worse than the matched single eLCS on
the declared final balanced-accuracy comparison. Do not call it a demonstrated
improvement. Other final comparisons are descriptive. No post-test tuning occurred.

## Remaining assessment work

- Complete the 5,000–6,500-word report: methods/pseudocode, references, results,
  limitations, contributions and reflections.
- Prepare GitHub evidence, the demonstration and required Turnitin/submission
  artifacts. Passing tests cannot establish completion of these deliverables.
- Do not reuse the now-scored final rows to select another configuration. Future
  tuning requires a separate evaluation plan and honest exposure disclosure.

No additional conventional model or eLCS reimplementation is needed for this
scope. Preserve historical results and their original/corrected attribution.
