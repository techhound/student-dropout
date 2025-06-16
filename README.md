**Student Dropout Risk Prediction Using Random Forests**

**Author**: James Cochrane\
**Date**: June 16, 2025\
**Project Type**: Predictive Analytics for Education Equity

---

### Objective

To develop a predictive model that can identify students at risk of dropping out early in the academic term using a combination of academic performance, behavioral patterns, and demographic information. The ultimate goal is to enable early interventions in educational settings.

---

### Data Source

We used the **UCI Student Performance Dataset**, publicly available from the UCI Machine Learning Repository:

- [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance)
- It includes two CSV files: `student-mat.csv` and `student-por.csv`, each containing 395 and 649 records respectively.
- The datasets describe Portuguese secondary school students and include features such as:
  - First- and second-term grades (`G1`, `G2`), final grade (`G3`)
  - Study time, failures, absences
  - Family relationships, parental education, support programs
  - Aspirational variables like `higher` (intention to pursue higher education)

The datasets were merged vertically to create a combined set of 1,044 student records.

---

### Target Variable Construction

The original dataset does **not** include a dropout flag. We created a **proxy target** to indicate students at risk of dropout, using the following rule:

```python
df["dropped_out"] = ((df["G3"] < 10) | (df["absences"] > 30)).astype(int)
```

- A student is labeled as **1 (at risk)** if:
  - Their final grade (`G3`) is less than 10 (failing grade in the Portuguese system), **or**
  - They have more than 30 absences (suggestive of disengagement)
- All other students are labeled **0 (not at risk)**

This is not a perfect substitute for longitudinal dropout data, but it approximates the conditions under which dropout is likely.

---

### Leakage Prevention

We identified early that the model achieved **suspiciously perfect performance**, which indicated **label leakage**. Specifically:

- `G3` (final grade) was included both as a predictor and as part of the label, giving the model direct access to its target.
- `G2` was also a late-term indicator, and while it wasn’t part of the label, it strongly correlates with `G3`.

To resolve this:

- We **removed **``** and **`` from the feature set entirely.
- Retained only early-term academic and behavioral data:
  - `G1` (first period grade)
  - `absences`, `studytime`, `failures`
  - Demographic and aspirational categorical variables

---

### Feature Engineering

- **Numeric features** were standardized using `StandardScaler`.
- **Categorical features** were one-hot encoded using `OneHotEncoder(handle_unknown="ignore")`.
- All preprocessing steps were included in a `ColumnTransformer` within a `Pipeline`.

---

### Model Selection & Tuning

We selected **Random Forest Classifier** as our primary model due to its:

- Ability to model non-linear interactions
- Built-in handling of mixed data types
- Resistance to overfitting when tuned properly
- Output of feature importances

#### Grid Search Parameters

```python
param_grid = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [None, 12, 20]
}
```

- We also added `class_weight="balanced"` to handle mild class imbalance.
- A **GridSearchCV** with 5-fold cross-validation was used.

#### Threshold Tuning

- Rather than use the default 0.50 cutoff for classification, we adjusted the threshold to **0.35** to improve recall for the minority class (`dropped_out = 1`).

```python
y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.35).astype(int)
```

---

### Final Model Results

- **Test Set Size**: 209 records (20% of full dataset)

| Metric      | Class 0 (not at risk) | Class 1 (at risk) |
| ----------- | --------------------- | ----------------- |
| Precision   | 0.93                  | 0.61              |
| Recall      | 0.86                  | 0.77              |
| F1-Score    | 0.89                  | 0.68              |
| **ROC-AUC** | -                     | **0.914**         |

**Confusion Matrix:**

```
[[139  23]
 [ 11  36]]
```

- **False Negatives (missed at-risk students)**: 11
- **False Positives (flagged incorrectly)**: 23

---

### Feature Importances (Top 5)

| Feature     | Importance |
| ----------- | ---------- |
| `G1`        | 0.2995     |
| `failures`  | 0.0664     |
| `absences`  | 0.0521     |
| `studytime` | 0.0166     |
| `higher_no` | 0.0128     |

These results show that **early academic indicators**, as well as aspirational flags like `higher`, provide significant predictive signal.

---

### Key Takeaways

1. **Label leakage is a real risk** in educational data where late-stage grades overlap with dropout proxies.
2. Removing `G2` and `G3` ensures that the model learns from **early-term information**, making it actionable.
3. Adjusting class weights and the classification threshold improves the **recall of at-risk students**, which is crucial for early intervention systems.
4. `G1`, `absences`, and prior failures consistently show up as **top predictive features**.

---

### Next Steps

- Compare performance with **Logistic Regression** for interpretability.
- Use **SHAP values** to explain individual predictions.
- Train on longitudinal student data with real withdrawal flags.
- Conduct **fairness audits** to ensure no demographic subgroup is disproportionately misclassified.
- Package the model as a dashboard or API for counselor review and action planning.

---

This project demonstrates how early academic signals and engagement metrics can be turned into actionable insights to reduce dropout risk, especially for equity-focused organizations like AVID.

