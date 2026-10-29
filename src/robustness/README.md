# Robustness and explainability

`challenge.py`: schema, annotation gates and contamination screening.
`inference.py`: strict adapters to actual Phase 1 trained serving artifacts.
`explain.py`: signed word-deletion probability differences with a fixed target.
`evaluate.py`: existing metric/error utilities under separate test conditions.

`train.py` and `reference.py` implement the newly authorized, separate TF-IDF
reference training run and JSON-only frozen artifacts. `perturb.py` probes
original held-out texts; `visualize.py` exports actual attributions.

See `docs/robustness.md` and `data/challenge/README.md`.
Phases 3–4 are not implemented.
