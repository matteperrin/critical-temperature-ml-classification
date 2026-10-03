# Matte's notes

## eLCS majority-class fallback correction

The bundled scikit-eLCS 1.2.4 implementation calculated its fallback class with
`max(self.classCount)`. This selects the largest class label, not the class with
the most training examples. For our labels, it selected class `1` (above 77 K)
even though class `0` was more frequent.

Changed `third_party/scikit-eLCS/skeLCS/DataManagement.py` to use:

```python
self.majorityClass = max(self.classCount, key=self.classCount.get)
```

This is an upstream correctness repair, not a new LCS algorithm or a tuned
prediction threshold. It uses training labels only and does not hard-code class
`0`. If counts tie, the first observed class wins deterministically for a fixed
training-row order. The change affects the fallback used when the rules cannot
produce a useful decision; it does not change the learned rules.

### Investigation evidence

Before this source edit, all five raw-eLCS development folds reproduced their
saved predictions exactly. Correcting only the fallback in memory changed mean
balanced accuracy from **0.4741 to 0.5946** and mean accuracy from **0.3854 to
0.7883**. It changed 8,880 of 17,010 development predictions. These are diagnostic
results, not newly saved experiment artifacts or final-test results.

On the first development fold, only 48.1% of validation rows matched a rule, and
rules specified approximately 39 of 81 features on average. The incorrect
fallback therefore affected many predictions. Fixing it does not resolve poor
coverage or establish convergence. Lower rule specificity increased coverage in
single-fold experiments but did not improve balanced accuracy relative to the
corrected default configuration.

The 4,253 reserved holdout rows were not evaluated. Historical baseline results
must remain identified as using the unmodified library; future corrected runs
need separate result files and explicit attribution of this local library edit.
The previous uncommitted comparison work and development artifacts were removed
from the working tree and preserved in a Git stash named
`Backup before isolated eLCS majority-class fix`.

### Verification and setup

`tests/test_elcs_majority_class.py` loads the bundled source directly and covers
frequency-based selection, arbitrary numeric labels, deterministic count ties,
and no-match predictions. The regression tests failed before the correction
(exit code 1) and all three passed afterward (exit code 0). The full remaining
suite also passed: 19 tests, with `PYTHONPATH` pointing to `third_party/scikit-eLCS`
so model imports used the corrected bundled source (exit code 0).

Run the focused checks with:

```bash
python -m unittest discover -s tests -p test_elcs_majority_class.py -v
```

An existing non-editable installation of `skeLCS` will not automatically pick up
the source edit. The new development/holdout runner explicitly loads the bundled
source, so it does not require reinstalling `skeLCS`. Legacy runners still import
the installed package; reinstall before generating corrected legacy results:

```bash
python -m pip install --no-deps ./third_party/scikit-eLCS
```

## Development study progress

Added training-only deduplication for the three-member ensemble and a bounded
iteration/population study. Commands and the predeclared selection rule are in
the [README](../README.md#bounded-corrected-library-development-study); library
isolation and provenance are documented in the [integration note](elcs_integration.md#corrected-versus-supplied-implementation).

Six experiments completed under `reports/holdout/elcs_development_v1/`:

- **Bug repair:** mean development balanced accuracy increased from 47.4% to
  59.5% with the raw data and training settings unchanged.
- **Training deduplication:** the corrected model reached 61.9%, versus 59.5%
  without deduplication.
- **Larger budgets:** increasing population alone reached 56.2%; increasing
  iterations alone reached 60.9%. Neither beat the 61.9% setting.

Balanced accuracy averages recall for the two classes; it is not the proportion
of all rows classified correctly. These differences are descriptive development
results, not statistically tested improvements or final-test evidence.

The process exited during the 10,000-iteration / population-1,000 experiment.
That partial experiment and machine-local logs were archived outside the repo.
`status.json` now records the interruption. The ensemble comparison is still
pending; no configuration has been selected from the incomplete grid. Completed
experiment artifacts and historical CSVs were preserved. No reserved holdout
rows were evaluated.

Verification: **30 tests passed**, covering tiny real-model fits of both variants,
source provenance, training-only preprocessing and study selection boundaries.
