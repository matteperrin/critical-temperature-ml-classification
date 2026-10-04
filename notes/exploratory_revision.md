# Exploratory retraining revision

## Why we are revising the work

We made two consequential methodological mistakes:

1. We treated 1,000 eLCS iterations as a meaningful learning budget without first
   checking the implementation's units. An iteration presents one record, not an
   epoch. The selected final configuration directly updated on only about 5.9%
   of its 16,964 deduplicated training records. Even the largest original
   development budget did not complete one pass. Our initial interpretation gave
   this limitation insufficient prominence.
2. Records later designated for final evaluation had already appeared in earlier
   full-data analyses and rule exports. We cannot honestly describe them as a
   historically untouched holdout. We have also now seen their model scores.

The original predictions and scores are not fabricated or computationally wrong.
Their interpretation was too broad. We retain them as evidence of what was done,
not as evidence of adequately trained eLCS or independent external validation.
See [the original results and their corrected interpretation](phase_two_results.md).

The user approved retraining on the same dataset and reporting an explicitly
**exploratory revision**, including reused-evaluation-row scoring. This revision
can correct training exposure; it cannot undo prior data exposure.

## Approved training and selection procedure

Keep the original dataset, grouped partitions, training-only source-row
deduplication, corrected bundled library and other model parameters. Do not use
newly obtained evaluation scores to select or extend the configurations.

Compute the deduplicated final-development training count using the existing
runner's loading and splitting logic. The confirmed count is **16,964**. Use
constant record-update budgets derived from that count:

| Final-development pass equivalent | Updates per fit/member | Population limits |
| --- | ---: | --- |
| 1 | 16,964 | 100 and 1,000 |
| 5 | 84,820 | 100 and 1,000 |
| 20 | 339,280 | 100 and 1,000 |

The smaller development training folds have 13,566–13,578 deduplicated records,
so they receive approximately 1.25, 6.25 and 25 passes respectively. These are
constant-update comparisons, not exactly equal-pass CV/final fits. The vendor
shuffles once when fitting and traverses that order repeatedly without
reshuffling at each wrap. Population limit `N` counts micro-classifiers, including
numerosity, not necessarily distinct exported rules.

1. Save the revision plan, dataset/source fingerprints, actual training counts,
   fixed reference hashes and disclosures before fitting.
2. Run all six corrected single-model candidates with seed 42 on the existing
   five development folds.
3. Select the highest mean development balanced accuracy. Exact ties prefer
   fewer updates, then the smaller population limit.
4. Run the matched three-member ensemble on development folds using seeds
   11, 42 and 73. It receives three times the single model's aggregate update
   budget; it is not an equal-compute comparison.
5. Freeze the selected pair and reporting settings before scoring reused
   evaluation rows.
6. Fit the selected single model and ensemble on the development partition and
   score each once on the existing 4,253 evaluation records. Export the exact
   fitted rules before scoring, using the existing runner.
7. Report every candidate's development results and the selected pair's
   exploratory evaluation results, including unfavorable outcomes.

One or twenty passes is not a guarantee of convergence or adequate training.
The budget grid is finite and approved in advance of revision results. It must
not be extended just because the revised models fail to beat an existing score.

## References and statistical interpretation

Retain the logistic-regression, SVM and random-forest records as fixed references
on the same partitions. Validate their compatibility and hashes; do not retrain
or retune them merely to accompany this revision. The revised comparison table
contains these three references and the retrained single/ensemble pair. Summarize
the original eLCS findings in writing rather than making the new comparison
dependent on superseded eLCS artifacts. Cleanup remains subject to agreement and
must wait until the replacement has been verified.

The original primary significance test remains attached to its original
1,000-update configuration pair. **It does not apply to the revised models.**
The revision uses descriptive comparisons, not a new confirmatory hypothesis
test. Neither reused-row rankings nor development-based selection establish
performance on genuinely independent materials or material families.

## Execution and evidence

The revision uses a new output directory so that original configurations,
predictions, rule exports, completion manifests and start markers are preserved.
Original fitting and comparison modules are reused, rather than changing the
meaning of their existing iteration argument or loosening their protections.

Long fits run serially because the library seeds global random generators.
Elapsed times and active/completed experiment status are recorded. A failed or
interrupted grid is incomplete, not a successful selection from whichever cheap
candidates happened to finish. No automatic restart or deletion of scoring
start markers is permitted.

**Current status: completed.** All 11 recorded stages finished under
`reports/holdout/elcs_exploratory_revision_v2/`. The 60 completion-manifest artifact
hashes were verified, and all five evaluation comparison rows were revalidated
against saved predictions and experiment manifests. The root `results.html` now
presents the exploratory revision, not the superseded seven-system comparison.
No additional fits or scoring were performed to generate the page.

## Completed findings

Development CV selected 339,280 updates and population limit 1,000. Its mean
balanced accuracy was **67.10%**, versus **63.65%** for the matched ensemble.
Thus the ensemble did not improve development balanced accuracy.

On the reused evaluation records:

| Model | Balanced accuracy | Positive precision | Positive recall |
| --- | ---: | ---: | ---: |
| Revised single eLCS | 63.95% | 81.49% | 29.40% |
| Revised eLCS ensemble | 64.49% | 87.92% | 29.91% |
| Logistic regression, fixed reference | 80.76% | 67.93% | 68.81% |
| SVM, fixed reference | 78.96% | 67.06% | 65.08% |
| Random Forest, fixed reference | 90.36% | 84.43% | 84.21% |

The ensemble detected 233 of 779 positive records, versus 229 for the single
model, and produced 32 rather than 52 false positives. Its **0.54-percentage-point**
balanced-accuracy advantage is descriptive only. It is not a consistent ensemble
advantage across development and evaluation stages, nor an established
statistically significant improvement.

Relative to the original evaluation, single eLCS increased from 63.15% to 63.95%
(approximately 0.80 points), and the ensemble from 58.32% to 64.49%
(approximately 6.18 points). Both training budget and population changed, so
these gains cannot be attributed solely to longer training. Despite the much
greater training exposure, conventional references remain substantially ahead.
The results do not establish convergence or external generalization.

Fitted rule examples on the results page are selected within each seed by
illustrated class and greatest training match count, with fitness breaking ties.
They come from the retrained populations, not the original short-budget exports.
Repeated presentations can contribute multiple matches: these counts are not
numbers of distinct training records or independent rule-validation results.
