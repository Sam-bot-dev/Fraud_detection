# train_baseline.py
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# 1) Load your labeled csv
df = pd.read_csv("transactions.csv")

y = df["label"].astype(int)
X = df.drop(columns=["label","txn_id"])

num_cols = X.select_dtypes(include=['number','bool']).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

pre = ColumnTransformer([
    ("num", "passthrough", num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cat_cols)
])

clf = XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    scale_pos_weight=max(1, (len(y)-y.sum())/max(1,y.sum()))  # handle imbalance
)

pipe = Pipeline([("pre", pre), ("smote", SMOTE(sampling_strategy=0.2)), ("clf", clf)])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pipe.fit(Xtr, ytr)

proba = pipe.predict_proba(Xte)[:,1]
ap = average_precision_score(yte, proba)
print("PR-AUC:", ap)

# choose threshold at ~1% FPR
prec, recall, thresh = precision_recall_curve(yte, proba)
# You can map to desired operating point later; for now:
import joblib; joblib.dump(pipe, "fraud_model.joblib")
