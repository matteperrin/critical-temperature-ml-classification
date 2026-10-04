# Results page design

## Context
- Artifact: offline research results page for readers without ML/tooling experience.
- Action: understand the result first; optionally inspect metrics, rules and evidence.
- Direction: custom Swiss research sheet, not a dashboard or a marketing landing page.
- Dials: DESIGN_VARIANCE=4, MOTION_INTENSITY=1, VISUAL_DENSITY=5.
- Signature: a full-scale, ranked comparison plot coupled to the negative ensemble result.
- Real visuals are the measured data plot; stock laboratory imagery would imply evidence we do not have.

## Typography
- Display: Bahnschrift, Arial Narrow, sans-serif. Body: Trebuchet MS, Arial, sans-serif.
- Locally installed fonts only: no network font requests; fallbacks remain readable offline.
- Scale: 14 / 16 / 20 / 24 / 32 / 40 / 48 px, approximately 1.25 ratio.
- Base: 16px, line-height 1.6, body tracking 0; numerical values use tabular figures.

## Color
| Role | OKLCH | sRGB fallback | Use |
|---|---|---|---|
| bg | oklch(0.978 0.004 240) | #f6f8fa | Cool research paper |
| surface | oklch(0.994 0.002 240) | #fcfdfe | Reading and table surfaces |
| fg | oklch(0.245 0.022 240) | #17232c | Main text |
| muted | oklch(0.430 0.020 240) | #46545e | Secondary text, still AA |
| border | oklch(0.840 0.012 240) | #c4cdd4 | Structural separators |
| track | oklch(0.920 0.008 240) | #e1e7eb | Chart track |
| accent | oklch(0.410 0.075 190) | #00584f | Links, focus, single eLCS bar |
| accent-fg | oklch(0.994 0.002 240) | #fcfdfe | Reserved contrast role |
| success | oklch(0.410 0.075 190) | #00584f | No decorative success badges |
| warning | oklch(0.480 0.120 35) | #963e28 | Ensemble result and bar |
| error | oklch(0.480 0.120 35) | #963e28 | No UI error state: build rejects damaged evidence |

## Spacing and shape
- Base unit: 4px. Spacing: 4 / 8 / 12 / 16 / 24 / 32 / 48 / 64 / 80 px.
- Widths: page up to 1152px; content follows an asymmetric 7:5 grid; single-column mobile.
- Radius: 0 only. Shadow: none. Hairlines indicate structure, not decoration.
- No icons, external imagery, framework, runtime JavaScript or network dependencies.

## Interaction
- Native anchor navigation and details/summary disclosures, usable without JavaScript.
- All navigation and disclosure targets are at least 44px high.
- Links have distinct hover, active and focus-visible states; focus uses the accent.
- No animated content; only 120ms link-color feedback, disabled for reduced motion.
- Mobile comparison becomes labeled rows without page-level horizontal scrolling.
- Print mode removes navigation and includes collapsed explanation/rule bodies.

## Verification
- Rendered in headless Chrome at 1920px and 375px; screenshots inspected. No page-level horizontal overflow.
- Keyboard: skip link is first focus target; Enter opens native rule disclosures.
- Tables retain accessible table roles; mobile header is visually hidden and caption uses the full row width.
- Measured sRGB text/background contrast: foreground 15.16:1, secondary 7.53:1, accent 7.91:1, warning 6.46:1. All exceed 4.5:1.
- Navigation/disclosure targets at least 44px; no tested body or control text smaller than 14px.
- Reduced-motion feedback disabled; print exposes disclosure bodies and produces a PDF.
- Zero external resource requests or browser runtime errors. Saved local evidence links verified by project tests.
- Hard-no scan: no surviving P0; no gradients, icon-card grid, decorative imagery, default single-font treatment or fabricated values.
- Direction judged justified, coherent and subject-specific: the measured comparison is the visual, and the unsuccessful ensemble is not hidden.
- Verification covers Chrome, not manual screen-reader use or a cross-browser certification. Screenshots/PDF are temporary local checks, not repository dependencies.
- Rebuild: `python src/critical_temperature/build_results_page.py`.
- Generated file: repository-root `results.html`; do not hand-edit its numbers.
