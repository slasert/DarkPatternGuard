# Robustness and explainability

Exactly four phases remain: Phase 1 initial classifier (completed); robustness,
explainability and error analysis (current); Phase 3 cross-domain/data/model
optimization (future); Phase 4 final integration/validation/readiness (future).
Phases 3 and 4 are not implemented.

## Status and provenance

Original Phase 1 weights and src/models code were unavailable in both upstream
commits, local searches and GitHub releases. The user subsequently authorized
new reference training. These models are **not** recovered Phase 1 weights or
exact reproductions of historical scores. Original results/ and data/cleaned/
are unchanged.

Working models, genuine test/challenge predictions, controlled perturbations,
actual-model explanations and API/UI integration are implemented. Human review of the authored challenge is still pending:
28 pending, 8 ambiguous, 0 approved. Its supervised Robustness Macro-F1 is
**null**. This is a remaining human research task; model loading now works.

## Architecture and files

Existing cleaning, metrics, calibration, confusion plots, prediction CSVs and
bootstrap utilities are reused. eval.benchmark supplies shared error extraction.
Its seed reader moved to src/eval/config.py so reports and existing ensemble
algorithms do not import missing training code just to read configuration.

| File | Purpose |
| --- | --- |
| src/robustness/train.py; scripts/train_reference_models.py | Train reference models using train/validation only. |
| src/robustness/reference.py | JSON-only frozen TF-IDF classifiers and exact feature terms. |
| src/robustness/inference.py | Trained-model adapters, identity and probability validation. |
| src/robustness/challenge.py | Schema, review gates and contamination checks. |
| scripts/build_challenge.py | Reproduce authored candidates without overwriting reviews. |
| src/robustness/review.py; scripts/review_challenge.py | Blind human-review CSV exchange with source hashes. |
| src/robustness/evaluate.py; scripts/evaluate_robustness.py | Frozen original-test and challenge evaluation. |
| src/robustness/perturb.py; scripts/evaluate_perturbations.py | Controlled held-out case/space/typo probes. |
| src/robustness/explain.py; scripts/explain_prediction.py | Signed word sensitivity and actual model feature attributions. |
| src/robustness/visualize.py | Standalone escaped attribution HTML. |
| app/frontend/src/components/ModelExplanation.tsx | Existing Analyze page extension, without redesign. |
| tests/test_robustness.py, test_reference_models.py, test_challenge_review.py, test_attribution_visualization.py | Schema, training, review, attribution and integration checks. |

## Training and model selection

Python 3.11 and pinned requirements.txt are required. The unchanged split has
1,610 train, 200 validation and 219 test rows, grouped by source page.
TF-IDF learns only from training text: unigrams/bigrams, up to 20,000 features,
Unicode accent stripping, lowercase, sublinear TF, L2 normalization.
Logistic regression uses balanced class weights and seed 1729. Multinomial NB
uses empirical priors. Validation Macro-F1 selects C or alpha from 0.5, 1, 2;
ties prefer smaller parameters. No test or challenge data is read during fitting.

Artifacts contain vocabulary, IDFs and trained coefficients as plain JSON,
with canonical label order, split/feature hashes, dimensions, finite-value
checks and provenance. Loaded probabilities must match scikit-learn at 1e-10
tolerance. No executable pickle is loaded. Existing artifacts cannot be
overwritten. These are deliberately modest reference models, not Phase 3 tuning.

The registry prefers restored original transformer candidates when present;
otherwise it chooses a reference model by validation Macro-F1 only. The local
selection is Naive Bayes. DPG_REFERENCE_ARTIFACT can explicitly select another
reference JSON. /models distinguishes reference-retrained entries from
historical results. Robustness evaluation refuses heuristic fallbacks.

## Reproduction

Run from the repo root; evaluation output directories must be new. Weights and
transient experiment outputs are ignored.

```powershell
python -m pip install -r requirements.txt
python scripts/train_reference_models.py
python scripts/evaluate_robustness.py --audit-only --output experiment_runs/audit-new
python scripts/evaluate_robustness.py --model reference_logistic_regression --output experiment_runs/reference-logistic
python scripts/evaluate_robustness.py --model reference_naive_bayes --output experiment_runs/reference-naive-bayes
python scripts/evaluate_perturbations.py --model reference_logistic_regression --output experiment_runs/perturb-logistic
python scripts/evaluate_perturbations.py --model reference_naive_bayes --output experiment_runs/perturb-naive-bayes
python scripts/explain_prediction.py --model reference_logistic_regression --text "The checkout price reverts at midnight. Only two units remain." --output experiment_runs/explanation-logistic
python scripts/explain_prediction.py --model reference_naive_bayes --text "You do not need to hurry; the price and availability stay the same." --output experiment_runs/explanation-naive-bayes
```

Training needs no new dependencies. Models live in models_out/reference/. For repeat training,
use a new --models-dir rather than replacing frozen weights. Reports are
written to the selected experiment output directory.
configs/robustness.json fixes evaluation seed, similarity threshold, resamples,
histogram bins and standalone attribution word limit. CPU evaluation uses
deterministic Torch settings and a headless plotting backend.

## Measured results

| New reference model | Validation Macro-F1 | Original-test Macro-F1 | Typo flips / changed texts |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.9450 | 0.9423 | 6 / 196 |
| Multinomial Naive Bayes | 0.9591 | 0.9437 | 3 / 196 |

These are **not challenge scores** or historical Phase 1 reproductions.
Uppercase (219 changed texts) and expanded whitespace (208 changed texts)
produce zero flips. Their scores inherit original held-out labels under an
explicit surface-invariance assumption, not new human annotation. This is
expected from preprocessing and does not establish general robustness.
Single internal character transposition is deterministic; its flips are
descriptive because meaning preservation is not individually reviewed.

Reports contain original-test per-class precision/recall, F1, ECE, bootstrap
intervals, confusion plots and actual error rows. They record package versions,
seeds, checkpoint/vocabulary/challenge/split hashes. Challenge artifacts contain
real predictions and confidence for all 36 candidates plus paired changes.
Supervised challenge metrics, category errors and matrices are generated only
after approved clean labels exist. Paired variants are dependent: row-bootstrap
CIs are exploratory, not significance tests. No significant winner is claimed.

## Human annotation and contamination

Read data/challenge/README.md. A teammate must review candidates independently
of model outputs. The blind CSV is experiment_runs/human-review.csv.

```powershell
python scripts/review_challenge.py export --output experiment_runs/human-review-new.csv
python scripts/review_challenge.py import --decisions experiment_runs/human-review.csv --output data/challenge/challenge.reviewed.jsonl
python scripts/evaluate_robustness.py --model reference_naive_bayes --challenge data/challenge/challenge.reviewed.jsonl --output experiment_runs/naive-bayes-reviewed
```

The lexical screen checks train, validation and test separately with existing
clean_text, normalized exact matching and char 3–5-gram cosine threshold 0.85.
It currently excludes zero candidates. It can miss semantic paraphrases:
manual inspection remains necessary. Reviewer/rationale and source-hash checks
prevent inconsistent imports but cannot authenticate human identity or quality.
Mixed active signals remain ambiguous without defensible adjudication.
Authored challenge and natural test differ in sampling/class mixture; score
differences are not causal improvements or directly matched comparisons.

## Explanations and API

POST /explain returns the actual explained text, model name/seed/hash,
probabilities and signed word spans, with optional same-model perturbation.
It stores no submitted text. Up to 80 word spans bounds interactive cost.
The React section highlights positive/negative sensitivity, shows numeric
values and exact feature terms, supports emoji offsets and compares edits.

Deletion holds the original predicted class c fixed:
P(c | full text) minus P(c | word removed). Positive supports that probability;
negative opposes it. This is sensitivity, not causal evidence, SHAP or
Integrated Gradients. Deletion changes overlapping n-grams and renormalization.

Exact TF-IDF × class coefficient terms plus bias reconstruct the target
**logit**, not its probability. All terms are returned; UI shows at most 20
features and states that limit. NB terms reconstruct joint log-probability and
are generally negative; cross-class effects need a comparison of class logits.
Reference models use full input text; original transformer loading retains its
256-subword truncation. Empty legacy word hints do not prove text is safe.

References: [LogisticRegression](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html)
and [feature ablation](https://captum.ai/api/feature_ablation.html).

## Verification

```powershell
python -m pytest
npm --prefix app/frontend run build
npm --prefix app/frontend run test:e2e
```

Latest full run: **44 passed, 1 skipped**. The explicit skip is upstream's absent
optional models.llm module; no LLM benchmark is claimed. Tests run normally if
that implementation is restored. Three browser tests and production build pass.
Tests cover actual fitting/roundtrip, deterministic artifact bytes, no test read
during training, corrupt files, exact logit terms and API prediction/XAI identity.
Phase 1 split/result diffs remain empty.

On this Windows/OneDrive host, the latest browser verification used one worker
to avoid a parallel Vite cold-start timeout:
`npm --prefix app/frontend run test:e2e -- --workers=1 --timeout=60000`.
All three tests passed. A smoke call using the actual trained local Naive Bayes
artifact returned 200 from health/predict/explain, with matching prediction and
attribution identity; this check was not a mocked model.
