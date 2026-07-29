# DriftWatch

**A model can keep predicting confidently after the world it learned from has moved. DriftWatch catches the moment it does.**

**Replaying a real logistic regression over 6,587 time-ordered UCI Online Shoppers sessions in 21 batches, DriftWatch shows accuracy fall from 0.8993 (June reference) to 0.4400 in batch 11 of 21 (Nov 03), the exact batch a synthetic drift is injected, and its Page-Hinkley detector signals that same batch. By the final batch, Dec 05, accuracy has settled at 0.5669 and AUC at 0.5694, both near coin-flip. On the untouched real stream, the same 21-batch replay still fires 229 drift alerts from ordinary seasonal movement alone, all computed client-side with no server, model download, or API key.**

![self-check](https://img.shields.io/badge/self--check-9%2F9%20passing-brightgreen)
![stream](https://img.shields.io/badge/stream-6%2C587%20rows%20%2F%2021%20batches-blue)
![features](https://img.shields.io/badge/features%20monitored-16-blue)
![python](https://img.shields.io/badge/python-pandas%20%2B%20scikit--learn%20pinned-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## The problem

A model can keep returning valid-looking predictions after its operating environment or the relationship it learned has changed. Nothing crashes, nothing errors, the numbers are just quietly wrong. DriftWatch makes that failure mode inspectable. It keeps input distribution movement (data drift) separate from error and performance movement (concept drift), then exposes the distributions, thresholds, and alert logic behind every status.

## How it works

DriftWatch is a static page with no server, model download, or API key. An offline Python builder fits a real classifier and writes three committed JSON artifacts (reference window, real stream, injected stream). The browser loads only those three files and recomputes every batch metric and every drift statistic client-side: PSI, KS, chi-square, Jensen-Shannon divergence, and Page-Hinkley on the sequential error stream. Moving the PSI alert threshold or KS alpha slider re-runs those computations live against the same committed predictions and features, so the counts on screen are never canned.

## Method

The offline builder fits a balanced scikit-learn logistic regression on the earliest usable months. June is an untouched reference window. Jul-Dec are split into 400-row temporal batches without shuffling.

For reference-bin proportions \(p_i\) and batch proportions \(q_i\):

\[
\mathrm{PSI} = \sum_i (q_i-p_i)\ln(q_i/p_i)
\]

Numeric features use fixed June decile bins, with underflow and overflow folded into the end bins. A small epsilon handles empty shares. The UI shows the requested traffic-light interpretation: below 0.10 is stable, 0.10-0.25 is moderate, and above 0.25 is significant. Research changed the presentation here: these cutoffs are common industry heuristics, but their error rates are not statistically calibrated and depend on binning and sample size. That caveat is visible in the UI, following the findings in [Yurdakul and Naranjo's PSI analysis](https://www.risk.net/journal-of-risk-model-validation/7725371/statistical-properties-of-the-population-stability-index). The user-adjustable PSI alert follows the same greater-than-or-equal threshold convention documented by [Evidently](https://docs.evidentlyai.com/metrics/customize_data_drift).

The two-sample Kolmogorov-Smirnov statistic is the maximum absolute gap between empirical CDFs:

\[
D_{n,m} = \sup_x |F_n(x)-G_m(x)|
\]

At user-selected \(\alpha\), the browser uses the asymptotic two-sided critical value:

\[
\sqrt{-\tfrac{1}{2}\ln(\alpha/2)}\sqrt{\frac{n+m}{nm}}
\]

At \(\alpha=0.05\), the leading constant is approximately 1.36. KS is computed from exported numeric samples, not histogram approximations. The statistic follows the [NIST two-sample KS definition](https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/ks2samp.htm). Categorical inputs use a two-sample chi-square homogeneity statistic with the 0.05 critical value.

Jensen-Shannon divergence is:

\[
\mathrm{JS}(P,Q) = \tfrac{1}{2}\mathrm{KL}(P\|M) + \tfrac{1}{2}\mathrm{KL}(Q\|M), \quad M=\tfrac{1}{2}(P+Q)
\]

Base-2 logarithms keep divergence in \([0,1]\). Research changed a common implementation shortcut: DriftWatch reports the divergence itself, not the square-root distance returned by some libraries. This follows [Lin's original formulation](https://doi.org/10.1109/18.61115).

Batch performance includes accuracy, rank-based ROC AUC with average ranks for ties, Brier score, and observed positive rate. Page-Hinkley reads the sequential 0/1 error stream:

\[
m_t = \alpha m_{t-1} + (e_t-\bar{e}_t-\delta)
\]

It signals when \(m_t-\min_{i\le t}m_i>\lambda\), after a minimum observation count. DriftWatch uses `min_instances=30`, `delta=0.005`, `lambda=50`, and `alpha=0.9999`, matching the established defaults documented by [River](https://riverml.xyz/dev/api/drift/PageHinkley/) and [scikit-multiflow](https://scikit-multiflow.readthedocs.io/en/stable/api/generated/skmultiflow.drift_detection.PageHinkley.html).

Input PSI, KS, and chi-square alerts are labeled **data drift**. Performance decay and Page-Hinkley are labeled **concept drift signals**, not confirmed causes. This wording is deliberate: feature drift is a proxy when ground truth is absent, while realized performance becomes available when labels arrive, as described in the [Evidently data drift](https://docs.evidentlyai.com/metrics/preset_data_drift) and [classification monitoring](https://docs.evidentlyai.com/metrics/preset_classification) guidance.

## Data

The source is the [UCI Online Shoppers Purchasing Intention dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset), DOI `10.24432/C5F88Q`. It contains 12,330 sessions, 17 predictor fields, a binary `Revenue` label, and a real `Month` field. UCI distributes it under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The original 1.07 MB CSV is cached at `data/raw/online_shoppers_intention.csv`.

Temporal use:

- Fit: Feb, Mar, May, 5,455 rows
- Reference: June, 288 rows
- Stream: Jul-Dec, 6,587 rows in 21 batches
- Model inputs: 10 numeric and 6 categorical fields. `Month` is the temporal coordinate, not an input.

`data/injected.json` is explicitly synthetic. Starting at Nov 03, seed `202707` multiplies `ProductRelated_Duration` by 1.6 and adds seeded noise, adds 0.035 to `ExitRates`, and flips 65% of returning-visitor labels. The generated artifact records 2,102 changed labels.

## Results

All values below are read directly off the running page at PSI alert `0.25` and KS alpha `0.05`. Nov 02 is the last batch shared by both streams; the injected stream diverges starting Nov 03.

| Batch | Stream | Accuracy | AUC | Brier | Note |
| --- | --- | ---: | ---: | ---: | --- |
| June (reference) | both | 0.8993 | 0.8659 | 0.0811 | n=288, untouched reference window |
| Nov 02 (batch 10 / 21) | both | 0.8350 | 0.8319 | 0.1307 | last batch before the injected split |
| Nov 03 (batch 11 / 21) | injected | 0.4400 | 0.4942 | 0.4836 | first injected batch; Page-Hinkley signals here |
| Dec 05 (batch 21 / 21) | injected, latest | 0.5669 | 0.5694 | 0.3804 | 272 alerts logged across the injected stream (230 data drift, 42 concept drift) |
| Dec 05 (batch 21 / 21) | real, latest | 0.8740 | 0.8491 | 0.1196 | 229 alerts logged across the real stream (214 data drift, 15 concept drift) |

The real stream's own Page-Hinkley detector still signals once, on Nov 04, from ordinary seasonal movement, with no injection at all. PSI threshold changes produce 300, 229, and 181 real-stream alerts at 0.10, 0.25, and 0.50. KS alpha changes produce 212, 229, and 234 alerts at 0.01, 0.05, and 0.10. Both controls were moved live on the running page to produce these counts, confirming the computation is client-side and not canned.

## Reproduce

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/build_data.py
node app.js --selftest
python3 -m http.server 8000
```

Open `http://localhost:8000/?selftest=1`. The Python builder prints its split, artifact size, injection count, and reference metrics, then exits nonzero if its schema or 5 MB data ceiling breaks. The browser query runs nine `console.assert` checks and prints a pass/fail summary.

Vercel can serve the repository root as a static site with no build command.

## Security/BYOK

There is no LLM, embedding model, API key, BYOK mode, database, cookie, local storage, or runtime third-party request. The page fetches only the three committed same-origin JSON artifacts. Dataset strings are inserted with `textContent`, never HTML. CSV export uses a local Blob URL and revokes it immediately.

## Limitations

- Month is the only temporal field. Within-month order is the source CSV order, not event time.
- June has 288 rows. Reference size and batch size affect metric variance and test power.
- PSI depends on fixed reference bins and epsilon handling. Its standard bands are heuristics.
- KS asymptotic critical values are approximate for finite samples and repeated values.
- Per-feature tests are not corrected for multiple comparisons, so some alerts will be false positives.
- Immediate labels are assumed. Real production systems often have delayed or missing outcomes.
- Page-Hinkley identifies error-stream change, not cause. Covariate shift can also change errors.
- Injected results prove detector behavior under one documented transformation, not universal sensitivity.
