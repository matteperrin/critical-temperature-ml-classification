# Exploratory results page design

## Context
- Artifact: offline research results sheet for readers without ML tooling.
- Action: understand the revised comparison, then inspect metrics, rules and evidence.
- Direction: preserve the existing custom Swiss research sheet; no framework or visual overhaul.
- Dials: DESIGN_VARIANCE=4, MOTION_INTENSITY=1, VISUAL_DENSITY=5.
- Signature: full-scale ranked comparison paired with concrete single/ensemble prediction counts.
- The first viewport explicitly identifies the revision as exploratory and the evaluation records as reused.

## Typography
- Display: Bahnschrift, Arial Narrow, sans-serif. Body: Trebuchet MS, Arial, sans-serif.
- Local fonts only; no network requests.
- Scale: 14 / 16 / 20 / 24 / 32 / 48 px, approximately 1.25 ratio; fluid display size 32–48px.
- Base: 16px, line-height 1.6; tabular numerical figures.

## Color
| Role | OKLCH | sRGB fallback | Use |
| --- | --- | --- | --- |
| bg | oklch(0.978 0.004 240) | #f6f8fa | Cool research paper |
| surface | oklch(0.994 0.002 240) | #fcfdfe | Table surface |
| fg | oklch(0.245 0.022 240) | #17232c | Main text |
| muted | oklch(0.430 0.020 240) | #46545e | Secondary text and reference bars |
| border | oklch(0.840 0.012 240) | #c4cdd4 | Structural separators |
| track | oklch(0.920 0.008 240) | #e1e7eb | Full-scale chart track |
| accent | oklch(0.410 0.075 190) | #00584f | Links, focus and single-model series |
| accent-fg | oklch(0.994 0.002 240) | #fcfdfe | Selection contrast |
| success | oklch(0.410 0.075 190) | #00584f | Descriptive positive difference, without a significance claim |
| warning | oklch(0.480 0.120 35) | #963e28 | Contrasting ensemble-series identity, not an error flag |
| error | oklch(0.480 0.120 35) | #963e28 | Reserved; damaged evidence fails the build |

## Spacing and shape
- Base unit: 4px. Ramp: 4 / 8 / 12 / 16 / 24 / 32 / 48 / 64 / 80px.
- Page maximum width: 1152px; asymmetric 7:5 and 5:7 grids; single-column mobile.
- Radius: 0. Shadows: none. No icons, stock imagery, gradients or decorative badges.

## Interaction and scientific copy
- Preserve anchor navigation, skip link and native details/summary controls.
- Navigation/disclosure targets at least 44px; visible focus; no animated content.
- Link-color feedback: 120ms ease-out, disabled for reduced motion.
- Mobile metric table becomes labeled rows; headers remain accessible.
- Print expands disclosure bodies.
- Five models: revised single/ensemble and three fixed conventional references.
- No inherited p-value or untouched-holdout claim. Explicit budget units, historical errors and bounded conclusions.
- Rule excerpts belong to the actual retrained populations; training matches include repeated presentations.

## Verification
- Rendered and inspected in headless Chrome at 1920px and 375px.
- No horizontal overflow, external resource requests or browser runtime errors.
- Keyboard skip link and native rule disclosure verified; two accessible tables retained.
- Targets at least 44px; tested control/body text at least 14px.
- Contrast ratios: foreground 15.16:1, muted 7.53:1, accent 7.91:1, ensemble color 6.46:1.
- Reduced-motion and print-expanded disclosures passed; print PDF produced in temporary storage.
- Hard-no scan: no new decorative components, fabricated figures or network dependencies.
- This is Chrome evidence, not manual screen-reader testing or cross-browser certification.
- Rebuild: `python src/critical_temperature/build_results_page.py`.
- Generated page: `results.html`; numerical content comes from hash-verified saved evidence.
