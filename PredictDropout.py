"""
Predict student “drop-out risk” with a Random Forest
---------------------------------------------------
• Uses the UCI Student-Performance dataset (Math + Portuguese).
• Label = 1 if   (final grade G3 < 10)  OR  (absences > 30).
• G3 is **removed from the feature set** to avoid data leakage.
Author : <your-name>
Date   : 2025-06-16
"""

import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             roc_auc_score,
                             RocCurveDisplay)

# ---------------------------------------------------------------------
# 1. Load the data  (Math + Portuguese combined)
# ---------------------------------------------------------------------
DATA_DIR = pathlib.Path(".")
df = (
    pd.read_csv(DATA_DIR / "student-mat.csv", sep=";")
    .pipe(lambda d: pd.concat([d,
                               pd.read_csv(DATA_DIR / "student-por.csv", sep=";")],
                              ignore_index=True))
)

# ---------------------------------------------------------------------
# 2. Create the target label  (Boolean mask → int 0/1)
# ---------------------------------------------------------------------
df["dropped_out"] = ((df["G3"] < 10) | (df["absences"] > 30)).astype(int)

# ---------------------------------------------------------------------
# 3. Split X, y   —  **drop G3 to prevent leakage**
# ---------------------------------------------------------------------
# X = df.drop(columns=["dropped_out", "G3"])
X = df.drop(columns=["dropped_out", "G2", "G3"])  # double-safety
y = df["dropped_out"]

# ---------------------------------------------------------------------
# 4. Feature lists  (only “early-term” numeric predictors)
# ---------------------------------------------------------------------
# numeric_features = [
#     "G1", "G2",            # 1st & 2nd period grades
#     "absences", "studytime", "failures"
# ]
numeric_features = ["G1", "absences", "studytime", "failures"]

categorical_features = [c for c in X.columns if c not in numeric_features]

# ---------------------------------------------------------------------
# 5. Pre-processing pipeline
# ---------------------------------------------------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("scaler", StandardScaler())]), numeric_features),
        ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]),
         categorical_features),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

# ---------------------------------------------------------------------
# 6. Random Forest + quick hyper-param grid
# ---------------------------------------------------------------------
# rf = RandomForestClassifier(random_state=42, n_jobs=-1)

rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"      # NEW
)
pipe = Pipeline([("prep", preprocess), ("clf", rf)])

param_grid = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth":    [None, 12, 20],
}

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1,
)

# ---------------------------------------------------------------------
# 7. Train / Test split & fit
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.20, random_state=42
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# ---------------------------------------------------------------------
# 8. Evaluation
# ---------------------------------------------------------------------
# y_pred  = best_model.predict(X_test)
# y_prob  = best_model.predict_proba(X_test)[:, 1]

y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.35).astype(int)    # NEW threshold

auc     = roc_auc_score(y_test, y_prob)

print("\n=== Best hyper-parameters ===")
print(grid.best_params_)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print(f"ROC-AUC: {auc:.3f}")

# ---------------------------------------------------------------------
# 9. Top 15 feature importances (after one-hot)
# ---------------------------------------------------------------------
feature_names = best_model.named_steps["prep"].get_feature_names_out()
importances   = best_model.named_steps["clf"].feature_importances_
top_idx       = np.argsort(importances)[::-1][:15]

print("\n=== Top 15 Feature Importances ===")
for rank, i in enumerate(top_idx, start=1):
    print(f"{rank:2d}. {feature_names[i]:<25s}  {importances[i]:.4f}")

# Optional: show ROC curve (opens a window if running locally)
try:
    RocCurveDisplay.from_predictions(y_test, y_prob)
except Exception:
    pass
