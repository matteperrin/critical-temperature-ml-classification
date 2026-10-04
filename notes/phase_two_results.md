# Phase II results and interpretation

**Historical results:** this note describes the original 1,000-update eLCS study,
not the current retrained models. The [exploratory revision](exploratory_revision.md)
is the current technical record; [results.html](../results.html) presents that
revision. The statistical test below applies only to the original model pair.

This is a technical results summary, not the final assessment report. This interpretation revision
makes the learning-budget limitation explicit and adds a derived majority-class
reference. Frozen configurations, predictions, scores and statistical results
are unchanged; no additional final evaluations or hypothesis tests were run.

## Main findings

- **Random Forest performed best among the evaluated configurations.** Its final
  balanced accuracy was **90.4%**, compared with 80.8% for Logistic Regression,
  79.0% for SVM and 63.1% for corrected, deduplicated single eLCS. These rankings
  are descriptive; no significance tests were conducted between conventional models.
  eLCS received less than one full pass of rule-learning updates, so this does not
  establish how adequately trained versions of the algorithms compare.
- **The library repair improved the tested eLCS configuration.** Final
  balanced accuracy was 47.5% for unmodified deduplicated eLCS and 63.1% for its
  corrected counterpart. The only difference was the majority-class repair.
- **The ensemble enhancement was unsuccessful on the primary metric.** It reached
  58.3%, versus 63.1% for the matched single model: **4.8 percentage points lower**.
  Higher ordinary accuracy does not overturn this result because the target is
  imbalanced and balanced accuracy was the declared primary metric.
- **More selective positive predictions missed more useful candidates.** Of 779
  above-77-K test records, single eLCS detected 282 and missed 497; the ensemble
  detected 167 and missed 612. Ensemble false positives fell from 344 to 167,
  but this came at the cost of recall falling from 36.2% to 21.4%.

For comparison, Random Forest detected 656 of the 779 positive records, missed
123, and produced 121 false positives. Its predictions remain screening signals,
not evidence of superconductivity, physical feasibility or commercial usefulness.

## Evaluation design and model changes

The same feature-group split was used for every system: 17,010 development rows
and 4,253 final rows. Development used five inner stratified group folds. Final
records comprise 3,031 distinct feature-vector groups, with 3,474 negatives and
779 positives. Identical feature vectors stay together, but related compositions
and material families are not guaranteed to be separated.

Original raw and deduplicated eLCS, corrected deduplicated eLCS, and the corrected
ensemble used 1,000 iterations and population limit 100. The single seed was 42;
ensemble seeds were 11, 42 and 73. The ensemble uses three times the total
iteration budget. It is not an equal-compute comparison.

Exact full-source-row deduplication occurred only in training partitions. Different
continuous-temperature measurements were retained. Logistic Regression and SVM
used training-fitted standard scaling. Conventional project defaults were unchanged:
Logistic Regression `max_iter=1000`, RBF SVM `C=1`, and Random Forest with 200 trees.

Development tested four single-model iteration/population combinations. The
predeclared highest-mean-balanced-accuracy rule selected 1,000 iterations /
population 100. Neither a larger budget nor the ensemble improved on that setting.
The bug repair, preprocessing and ensemble must be discussed as separate changes.

The [final plan](../reports/holdout/final_comparison/plan.json) was saved before any
final scores were viewed. It freezes configuration hashes, source hashes, runtime
versions and the statistical recipe. Seven final evaluations ran once, with no
post-test tuning. Each has predictions, confusion counts, a start marker and a
completion manifest. Completed eLCS rule exports came from those exact fitted
models, without a separate fit.

## Learning-budget interpretation

In the bundled implementation, one eLCS iteration processes one training record,
not one epoch. The data is shuffled before learning, then traversed sequentially;
only after reaching the end does traversal wrap. See the learning loop in
[`eLCS.py`](../third_party/scikit-eLCS/skeLCS/eLCS.py) and traversal in
[`OfflineEnvironment.py`](../third_party/scikit-eLCS/skeLCS/OfflineEnvironment.py).

The corrected, deduplicated final single model had 16,964 training records but
only 1,000 individual record presentations for rule learning: approximately
**5.9%** received direct rule-learning updates. All supplied training records
still informed feature characterization and class counts, including the majority
fallback. It would therefore be inaccurate to say the model used only 1,000
records altogether.

Development training partitions contained 13,566–13,578 deduplicated records.
At 1,000 iterations, approximately 7.4% received direct updates. Even the largest
tested budget of 10,000 iterations did not complete one full development-training
pass. Ensemble seeds changed the shuffled training prefixes as well as evolutionary
randomness. The three-member comparison is not purely a test of different
rule populations learned from complete passes through the same records.

These are **sub-epoch rule-learning budgets**, not evidence of adequate training
or convergence. The observed ranking and unsuccessful ensemble remain valid for
the recorded configurations. They do not establish eLCS's fully trained capability,
and they do not prove longer training would help. The correct response is to
qualify the interpretation, not silently replace scores or retune using the
already-viewed final records.

## Metrics and supporting results

Balanced accuracy averages negative-class and positive-class recall equally. It
is not the percentage of all rows classified correctly. Precision, recall and F1
refer to class 1 (`critical_temp > 77`). Zero-denominator metrics return zero.

### Majority-class reference

The final records contain 3,474 negatives and 779 positives. Always predicting
class 0 would therefore achieve **81.7% ordinary accuracy**, **50.0% balanced
accuracy** and **0% positive recall**. Positive precision is undefined with no
positive predictions; the project's zero-division convention would report zero.

This reference is derived from saved confusion counts, not an eighth fitted
experiment or an additional hypothesis test. It was added during interpretation
and is not represented as a predeclared benchmark.

The ensemble's ordinary accuracy is exactly equal to this reference because its
167 true positives equal its 167 false positives. Its predictions are nevertheless
different: it detects 167 positive records and reaches 58.3% balanced accuracy.
Thus it improves on the trivial reference on balanced accuracy, but still loses
to corrected single eLCS on the predeclared comparison. Do not describe the
ensemble as equivalent to always predicting the majority class.

### Aggregation and evidence

CV metrics are unweighted fold means with sample standard deviations, not
confidence intervals. CV confusion counts are pooled. Final metrics use all
reserved rows. ROC-AUC and average precision use conventional-model probability
scores; they are intentionally unavailable for these eLCS records. Average
precision is not trapezoidal PR-AUC. CSV scores are parsed with round-trip float
precision so tiny score distinctions do not silently change rank-based metrics.

- [Development comparison](../reports/holdout/final_comparison/development_comparison.csv)
- [Final comparison](../reports/holdout/final_comparison/test_comparison.csv)
- [Primary statistical result](../reports/holdout/final_comparison/statistical_test.json)
- [Completion record](../reports/holdout/final_comparison/evaluation_complete.json)

These detailed CSVs support the narrative above; the headline results should not
be replaced with an unexplained table of hyperparameter codes. Saved configurations
contain machine-local absolute data/provenance paths. On another checkout, create
new experiments through the documented runners rather than editing frozen records;
the original artifacts are historical evidence, not portable fitted models.

## Primary statistical comparison

The sole predeclared contrast was corrected deduplicated single eLCS versus the
corrected deduplicated ensemble. The metric was balanced accuracy, expressed as
ensemble minus single model. The alternative was two-sided, with alpha 0.05.

The group-paired permutation test swaps whole model prediction vectors within
feature-vector groups, rather than independently swapping duplicate records.
Rows retain class weights `1 / (2 * number of records in their true class)`.
Thus the statistic is record-weighted balanced accuracy, not an equal-group metric.

There were 328 groups with nonzero effects. The test used 19,999 seeded Monte Carlo
swaps (seed 42), with `(extreme + 1) / (draws + 1)` correction. Exact enumeration
is used by the implementation for at most 16 informative groups.

The observed difference was **-0.04834** and the Monte Carlo p-value was
**0.00005**, the smallest reportable value with this sampling budget. At alpha
0.05, reject the exchangeability null: the difference is statistically significant
under the stated assumptions, and its direction favors the **single model**, not
the ensemble. Do not describe this as a statistically significant improvement.

The test assumes independent groups and within-group exchangeability of the two
model prediction vectors under the null. Unknown material-family dependence and
historical data exposure limit this interpretation. It is not an assumption-free
or external-validation claim. Other comparisons remain descriptive.

## Three sample ensemble rules

These examples come from the actual final ensemble. They were selected by class
and training match count, not by their success on final test examples. CSV row
numbers include the header. Each rule is a conjunction of all its specified
conditions; `#`/unspecified features are wildcards. The excerpts below are **not
complete rules**. Full, unrounded conditions are in the linked population CSVs.
Continuous conditions use strict lower/upper comparisons; discrete conditions use
equality. Decimal bounds below are rounded for discussion only.

### Rule 1: member seed 11, CSV row 23, predicts above 77 K

[Full rule population](../reports/holdout/elcs_development_v1/corrected_ensemble_dedup_selected/rules/member_11/rules.csv)

This rule specifies 41 of 81 features. Example conditions include mean atomic mass
between approximately 13.1584 and 145.2273, weighted mean valence between 1.7865
and 3.7335, and valence range equal to 3. These restrict aggregate composition
properties; they do not identify a unique compound or a physical mechanism.

The rule was correct on 4 of 9 training matches (44.4%), with fitness about 0.01734
and numerosity 1. Its limited experience and modest accuracy do not support using
it as an independent screening recommendation.

### Rule 2: member seed 42, CSV row 76, predicts at or below 77 K

[Full rule population](../reports/holdout/elcs_development_v1/corrected_ensemble_dedup_selected/rules/member_42/rules.csv)

This rule specifies 45 features. Example conditions include atomic-mass range
between approximately 96.9095 and 148.9026, mean first ionization energy (`mean_fie`)
between 622.4621 and 928.3879, and weighted mean valence between 0.0827 and 4.4487.
The broad valence interval overlaps positive rules, showing why an isolated
feature condition cannot explain the decision; the complete conjunction matters.

It was correct on 3 of 5 training matches (60.0%), with fitness about 0.07776 and
numerosity 1. Five matches are weak evidence of generalization, even though this
rule is more heavily weighted than the two positive examples.

### Rule 3: member seed 73, CSV row 73, predicts above 77 K

[Full rule population](../reports/holdout/elcs_development_v1/corrected_ensemble_dedup_selected/rules/member_73/rules.csv)

This rule specifies 35 features. Example conditions include mean valence between
approximately 1.8525 and 3.3475, weighted mean atomic radius between 91.8782 and
139.0449, and mean atomic mass between 71.1485 and 141.1851. The interval overlap
with other rules illustrates alternative conjunctions learned by different seeds,
not a new causal explanation of superconductivity.

It was correct on 3 of 8 training matches (37.5%), with fitness about 0.007416 and
numerosity 1. It is another low-experience rule whose positive prediction should
not be trusted in isolation.

### What these rules do and do not explain

Within each member, all matching rules vote using fitness multiplied by numerosity.
A sample rule alone does not determine that member's output. The three member
predictions then use majority voting. When no useful rule decision exists, the
corrected library falls back to the majority training class. Such a fallback is
not a rule-based explanation.

Exported `Accuracy`, `Correct Count` and `Match Count` are training statistics,
not held-out rule performance. These highly specific, low-experience rules support
inspection of the learned representation but provide weak evidence for practical
trustworthiness. Covering can also produce intervals extending beyond physically
plausible values; those are model bounds, not proposed material properties.
Domain review and experimental validation remain necessary.

## Verification and limitations

Project checks passed: `python -m pytest tests -q` and the documented unittest
suite. A separate Python 3.13 virtual environment installed `requirements.txt`,
passed `pip check`, and passed all 66 project tests. No global environment was
changed. The fresh check covers installation and synthetic integration, not a
second execution of the irreversible final evaluation or a fresh network fetch.

An unscoped pytest run also collected bundled upstream tests: 31 failed because
they expect `test/DataSets/...` relative to a different working directory. Those
vendor fixtures were not repaired or weakened; use the project-scoped commands.
The legacy full-data rule exporter was exercised during a fail-first check before
its CLI was replaced; historical output bytes were unchanged, and no new final
predictions were produced by that check.

The reserved records had already appeared in earlier full-data analyses and rule
exports. The workflow prevents future test-driven tuning but cannot make them
retroactively unseen. Further limitations include unidentified material-family
relationships, class imbalance, limited LCS training budgets, no established
convergence, and binary simplification of a continuous temperature. Results apply
to known superconductors in this dataset, not arbitrary materials.

The ensemble did not deliver the hoped-for enhancement. Present that negative
result honestly. The remaining assessment work is the full report, references,
contribution/reflection evidence, Turnitin submission and demonstration; these
cannot be inferred from passing code tests.
