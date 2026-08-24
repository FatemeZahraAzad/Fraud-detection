# Experiments Report — Credit Card Fraud Detection

## Experiment 1: Effect of Feature Scaling on KNN

### Motivation

KNN classifies a new sample based on the majority class among its `k` nearest neighbors,
where "nearest" is measured with Euclidean distance:

    d(x_i, x_j) = sqrt( sum_k (x_ik - x_jk)^2 )

This distance treats all features as if they live on the same numeric scale. If one feature
has a much larger standard deviation than the others, it dominates the sum and effectively
becomes the only feature the model "sees."

In this dataset, `V1...V28` are PCA components with std roughly in the 1–2 range, while
`Amount` has std ≈ 250. The ratio of their squared contributions to the distance is on the
order of (250/2)^2 ≈ 15,000–16,000, meaning `Amount` alone can dominate the distance
calculation unless features are scaled.

### Setup

- Model: `KNeighborsClassifier(n_neighbors=5)`
- Evaluation: 5-Fold Stratified Cross-Validation on the training set
- Compared two pipelines: with `StandardScaler` and without

### Results

| Model | Scaling | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| KNN | Without | 0.800 | 0.013 | 0.026 |
| KNN | With | 0.916 | 0.765 | 0.832 |

### Analysis

**Why KNN is sensitive to scaling:**
Without scaling, the neighbors selected for any given transaction are essentially chosen by
how close their `Amount` value is, regardless of the pattern encoded in `V1...V28`. Because
fraudulent transactions make up only ~0.17% of the data and are spread across a wide range
of `Amount` values, an unscaled model finds almost no fraud cases among the 5 nearest
neighbors of most fraudulent points — collapsing Recall to nearly zero (0.013). Precision
stays relatively higher (0.8) because the very few predictions the model *does* make as
"Fraud" tend to be cases that happen to land close to other frauds in raw `Amount`, which
are more likely to be correct. After scaling, all features contribute comparably to the
distance, neighbors reflect the actual multivariate pattern of fraud, and both Precision and
Recall improve sharply.

**Why Decision Tree is expected to be insensitive to scaling:**
A Decision Tree does not compute distances between samples. At each split it evaluates a
single feature independently against a threshold (e.g. "is `V14 <= 0.5`?"). Any monotonic
(and in particular any positive linear) transformation of a feature — such as scaling —
preserves the *rank order* of its values. Since the optimal split point is chosen purely by
how well a threshold on the sorted values separates the two classes (by impurity, not by
absolute magnitude), scaling changes only the numeric value of the threshold, not which
samples end up on each side of the split. This will be verified empirically once the
Decision Tree model is trained.

---

*(Further sections — Model Comparison, Hyperparameter Experiment, Threshold Experiment,
Final Model Selection — to be appended as those phases are completed.)*