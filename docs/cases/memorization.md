# Checking held-out evaluation against memorization

## Question and prerequisites

Was model accuracy measured on **held-out records not used for fitting**?

A **training set** provides rows for fitting parameters, while an **evaluation set** assesses how well predictions generalize to unseen records. High-capacity learners can memorize arbitrary labels on training inputs. Reporting performance on fitted rows measures memorization rather than generalization.

## Synthetic fixture

The fixture consists of 100 synthetic rows with 20 independent normal features and random binary labels (no genuine relationship between features and labels). A stratified split assigns 70 rows to training and 30 rows to a held-out test set.

An unpruned scikit-learn `DecisionTreeClassifier` is fitted strictly on the 70 training rows.

## Misleading approach

The misleading approach evaluates accuracy on the very 70 rows used to fit the tree. Because the decision tree expands until all leaves are pure, the measured training accuracy is `1.0` (100%).

This creates the illusion of high predictive power on pure noise. The verification check audits the fit row IDs against the evaluated row IDs and rejects any overlap.

## Correction

The corrected approach evaluates the fitted tree exclusively on the 30 held-out test rows disjoint from training. On random noise, the held-out accuracy hovers near chance (for seed 42, `0.4333`), accurately revealing that the tree does not generalize.

Verifying that the set of fitted row IDs and evaluated row IDs share zero records ensures held-out evaluation integrity.

## Run and verify

```text
python -m ml_failure_lab run memorization
python -m ml_failure_lab run memorization --json
python -m ml_failure_lab verify memorization --approach bad
python -m ml_failure_lab verify memorization --approach fixed
```

The bad check prints `FAIL` and exits with code 1 due to overlapping record IDs. The fixed check prints `PASS` and exits with code 0. Library versions and seeds may alter exact held-out scores, but the verification requirement strictly enforces disjoint record IDs.

## Limits and references

This case demonstrates full memorization using an unpruned tree on pure synthetic noise. It tests whether evaluation records are strictly separated from fitting rows, not whether a specific accuracy threshold is reached.

See [scikit-learn's model evaluation guide](https://scikit-learn.org/stable/modules/model_evaluation.html) and [cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html) for standard evaluation methods.
