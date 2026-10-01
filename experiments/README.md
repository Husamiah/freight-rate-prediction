# Experiments (decision log)

Run from the repo root, e.g. `python -m experiments.cv_round3`. Evaluation uses rolling-origin folds on the labeled data only.

| Script | Question | Finding |
|---|---|---|
| `quick_eval.py` | Which target / loss? (single Sep-Oct split, early stopping on it, so slightly optimistic) | log(rate/mile) with Huber beats log(rate), L1 and the median-$/mile baseline |
| `cv_experiments.py` | Feature sets, lane/city target encoding, Huber alpha | Dropping calendar features helps (MAE 121 -> 112); target encoding adds nothing; alpha 0.1 best |
| `cv_round2.py` | Pruning + LightGBM grid (leaves, min child samples) | Differences tiny; simpler trees (7 leaves) marginally best |
| `cv_round3.py` | Model families and blends | LightGBM + linear-tree + spline Ridge blend best on May-Aug folds |
| `cv_round4.py` | Same-day market means, day-of-week | Same-day market means help; day-of-week does not |
| `cv_round5_drift.py` | Why does Sep-Oct get worse? Which fix? | Models under-predict later months (drift); a linear trend in the Ridge component fixes most of it; recency weights / short windows are unstable |

Note: `cv_round3.py` and `cv_round4.py` predate the final model in `src/models.py` (they use 2 folds and the pre-trend blend). The final pipeline is `src/run_all.py`.
