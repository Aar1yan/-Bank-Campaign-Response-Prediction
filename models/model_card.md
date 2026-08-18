# Model Card — Bank Marketing Campaign Response Prediction

## Model name
`final_model.joblib` — Logistic Regression subscription-probability classifier.

## Problem
Predict whether a bank customer will subscribe to a term deposit, so that a
Portuguese bank's marketing team can prioritize who to contact.

## Prediction scenario
Immediately before a planned customer contact, using only customer and
campaign information already available at that moment — **not** during or
after the call itself.

## Dataset
UCI Bank Marketing Dataset (`bank-additional-full.csv`), 41,188 rows, direct
marketing phone campaigns run by a Portuguese bank, 2008–2010. Historical
data — it reflects that specific period, not necessarily current customer
behavior.

## Target
`y`: whether the client subscribed to a term deposit (`yes` / `no`), encoded
as 1 / 0 (`yes` = 1). Overall positive rate: 11.3%.

## Feature policy
Only information available **before** the specific planned contact is used
(notebook 03's leakage investigation). 22 features: 8 client/economic numeric
fields, `campaign`-derived and contact-history numeric fields, and 12
categorical fields (see `src/config.py` for the exact lists).

## Excluded features
- **`duration`** — the duration of the current/last call. Only known once
  the call has already happened; strongly predictive but unusable for a
  before-call prediction (notebook 03).
- **`pdays`** — technically available (historical, pre-campaign information),
  but excluded from the trained feature set for a separate, preprocessing
  reason: its `999` sentinel value dominates the column and destabilizes
  `StandardScaler` (notebook 05). Its useful signal is already captured by
  the engineered `was_previously_contacted` flag.

## Final model
Logistic Regression (scikit-learn), inside a single pipeline:
`ColumnTransformer(StandardScaler + OneHotEncoder) → LogisticRegression`.
Selected in notebook 05 as the strongest baseline candidate and confirmed
in notebook 06 — hyperparameter tuning found no configuration that beat
these defaults.

## Important hyperparameters
- `C = 1`
- `class_weight = None`
- `max_iter = 1000`
- `random_state = 42`

## Evaluation metrics (notebook 07, held-out chronological test set, n=8,236)
| Metric | Value |
|---|---|
| Accuracy | 0.645 |
| Precision | 0.456 |
| Recall | 0.793 |
| F1 | 0.579 |
| ROC-AUC | 0.747 |
| PR-AUC (Average Precision) | 0.534 |

## Selected threshold
**0.15** — chosen via out-of-fold cross-validation on training data only
(never on the test set), as the best-F1 threshold found there. At this
threshold on the test set, the model prioritizes ~54% of customers and
captures ~79% of actual subscribers in that sample.

## Known limitations
- **Train/test distribution shift**: the chronological split (necessary to
  avoid an unrealistic evaluation) means the training period's subscription
  rate (6.4%) and the test period's (30.8%) are genuinely different regimes.
  This is why the selected threshold ends up prioritizing a larger share of
  test customers (~54%) than a "prioritization" framing ideally implies.
- **Single train/test split**, not a rolling/walk-forward temporal
  validation.
- **Historical data** (2008–2010) — a specific economic period; current
  customer behavior and banking practices may differ.
- **No real cost or revenue data** — the threshold was chosen by F1 and
  general reasoning, not an actual cost-benefit calculation.
- **Modest model search** — 4 model families compared (notebook 05), 2
  tuned (notebook 06), each with a small hyperparameter search space.
- One coefficient (`euribor3m`) has a counterintuitive sign relative to its
  raw correlation with the target, traced to multicollinearity among the
  economic indicators (notebook 08) — a reason to trust the model's
  *predictions* more than any single coefficient's individual story.

## Intended use
A prioritization aid for deciding which customers to contact first with
limited marketing resources — **not** a guarantee of individual customer
behavior, and not validated for any use beyond this dataset's scenario
(e.g. not for credit decisions, pricing, or any use affecting a customer's
access to financial services).

## Important warning
**Model outputs are estimated probabilities, not guarantees.** A high
predicted probability does not mean a customer will definitely subscribe,
and a low one does not mean they definitely won't. Predictions describe
patterns the model learned from historical data, not causal explanations of
customer behavior (notebook 08).
