# Project scripts

Command-line entry points used by the Makefile belong here. Scripts should
remain deterministic and read shared settings from `configs/base.yaml`.

Robustness evaluation adds `build_challenge.py` (exclusive creation of unapproved
candidates) and `evaluate_robustness.py` (audit/frozen-model evaluation with
`configs/robustness.json`). See `docs/robustness.md` for commands and artifact requirements.

`review_challenge.py` exports blind human-review CSVs and validates explicit
annotation imports. `explain_prediction.py` exports actual-model JSON/HTML word
attributions without changing the serving model.

After explicit authorization to replace unavailable weights, `train_reference_models.py`
fits two new reference models on train/validation only. `evaluate_perturbations.py`
measures held-out metamorphic sensitivity.
See `docs/robustness.md` for the ordered reproduction commands and review limitation.
