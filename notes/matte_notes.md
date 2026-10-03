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

All eight experiments completed under `reports/holdout/elcs_development_v1/`:

- **Bug repair:** mean development balanced accuracy increased from 47.4% to
  59.5% with the raw data and training settings unchanged.
- **Training deduplication:** the corrected model reached 61.9%, versus 59.5%
  without deduplication.
- **Larger budgets:** increasing population alone reached 56.2%; increasing
  iterations alone reached 60.9%; increasing both reached 59.5%. None beat the
  61.9% setting. The predeclared rule selected 1,000 iterations / population 100.
- **Ensemble:** three members at the selected budget reached 59.2%, versus 61.9%
  for the matched single model: a decrease of 2.7 percentage points. Positive-class
  recall fell from 33.2% to 23.0%, while precision rose from 44.4% to 54.1%. The
  ensemble was more selective but missed more above-77-K materials. It used three
  times the total iteration budget and did not improve the primary metric.

Balanced accuracy averages recall for the two classes; it is not the proportion
of all rows classified correctly. These differences are descriptive development
results, not statistically tested improvements or final-test evidence.

After the earlier interruption, the last grid setting was rerun from scratch
and the ensemble was fitted. `continuation.json` records the source check and
preserved artifacts; only the study driver's bookkeeping differed from the
original plan. Model, loader, split and bundled-library sources were unchanged.
The original six experiments and plan were preserved. `selection.json` records
the development-selected budget, and `study_complete.json` / `status.json` mark
completion. No reserved holdout rows were evaluated. The ensemble is a tested
candidate enhancement, not a demonstrated performance improvement.

Verification: **30 tests passed**, covering tiny real-model fits of both variants,
source provenance, training-only preprocessing and study selection boundaries.
