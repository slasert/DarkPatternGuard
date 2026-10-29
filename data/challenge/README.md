# Robustness evaluation challenge data

The project-authored challenge is covered by the root MIT LICENSE. Original
Phase 1 source-data provenance and upstream terms remain in data/DATA_CARD.md.

`challenge.jsonl` contains 36 project-authored synthetic candidates: 24 originals
(four per category), eight proposed surface variants and four proposed semantic
variants. They were not sampled from Phase 1 data. Construction source and
proposed reasoning are recorded per row. The reproducible authoring source is
`scripts/build_challenge.py` v1. This is a stress test, not natural data.

All labels are **proposals**: 28 pending and eight ambiguous rows, zero approved.
Never automatically copy `proposed_label` into `label`.

## Annotation protocol

Review originals and variants independently, without consulting model outputs.
Prefer two independent annotators and adjudicate disagreements. Record reviewer
identifiers and adjudication reasoning in `reviewer` and `rationale`, and version
the reviewed file. Set `review_status: approved` and a canonical `label` only
after human approval. Unresolved cases remain `ambiguous`, with `label: null`;
unsuitable cases become `rejected`. Reviewers may correct proposals. The software
enforces recorded approval, not reviewer honesty or inter-annotator agreement.
Report agreement separately if multiple annotators participate.

| Label | Rule |
| --- | --- |
| Urgency | A live time constraint encourages purchase or checkout. |
| Scarcity | Restricted quantity or availability encourages purchase. |
| Social Proof | Others' purchases, popularity or endorsements persuade. |
| Normal | Specifications, factual instructions, negated/quoted cues without active pressure. |

Label active meaning, not keywords. Negation cancels only cues within its scope.
Punctuation alone does not imply urgency; rare materials need not imply low
stock. Labels identify linguistic cues, not proof of deception or legal findings.
If multiple active classes coexist without a defensible dominant cue, keep the
case ambiguous. Context-dependent promotions stay ambiguous without context.

`parent_id` references an original. `invariant` proposes preserved meaning;
`label_change` proposes changed meaning; `uncertain` leaves the relation open.
Approved invariant pairs require equal labels; approved label-change pairs
require different labels. A prediction flip is not automatically a failure.

## Separation and contamination

Never use this directory for training, validation, tuning or calibration.
Audit against train, validation **and test**, using the existing `clean_text`
normalization, alphanumeric exact matching and character 3–5-gram TF-IDF cosine
similarity (threshold 0.85). All external matches are quarantined from metrics.
Undeclared internal near duplicates and literal duplicates are also quarantined;
declared nonidentical controlled edits are retained. All matches are reported.

The lexical detector can miss semantic paraphrases and flag shared phrasing.
Inspect `contamination.json` before academic claims. Do not tune thresholds on
model results. Revise/remove contaminated cases, retain a review record and
rerun; there is no silent override. Paired examples are statistically dependent.

Rebuild candidates into a **new** file (exclusive creation protects annotations):

```powershell
python scripts/build_challenge.py --output robustness_candidates_rebuilt.jsonl
```

The checked-in file is the unreviewed baseline. Keep reviewed revisions
separately or commit them with an explicit annotation record.

## Practical human-review exchange

A blind UTF-8 CSV has been prepared locally at `experiment_runs/human-review.csv`.
It hides synthetic proposed labels/reasons and contains no model predictions.
Fill `review_status`, `label`, `reviewer` and `rationale` yourself for each decision.
Use the canonical English class names above; leave `label` empty for pending,
ambiguous or rejected examples. A human must approve the records; this tool
does not substitute automated review for annotation.

```powershell
python scripts/review_challenge.py export --output experiment_runs/another-review.csv
python scripts/review_challenge.py import --decisions experiment_runs/human-review.csv --output data/challenge/challenge.reviewed.jsonl
python scripts/review_challenge.py status --source data/challenge/challenge.reviewed.jsonl
```

Import validates all IDs, source text/category, dataset SHA-256, reviewer,
rationale and reviewed pair semantics before writing a new JSONL. Truncated,
duplicate, stale or inconsistent label files cannot silently pass.
Existing files are never overwritten. Unedited templates import as pending,
not as approved. The software records an asserted human review; it cannot
authenticate reviewer identity or guarantee annotation quality.
