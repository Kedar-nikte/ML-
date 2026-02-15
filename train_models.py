import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# ------------------------
# Load dataset
# ------------------------
df = pd.read_csv(r"data\Dry_Bean_Dataset.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

le = LabelEncoder()
y = le.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------
# Models
# ------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric='mlogloss')
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, preds)
    pre = precision_score(y_test, preds, average='macro')
    rec = recall_score(y_test, preds, average='macro')
    f1 = f1_score(y_test, preds, average='macro')
    mcc = matthews_corrcoef(y_test, preds)
    auc = roc_auc_score(y_test, prob, multi_class='ovr')

    results.append([name, acc, auc, pre, rec, f1, mcc])

    joblib.dump(model, f"model/{name}.pkl")
    joblib.dump(scaler, r"model/scaler.pkl")
    joblib.dump(le, r"model/label_encoder.pkl")


pd.DataFrame(results,
             columns=["Model","Accuracy","AUC","Precision","Recall","F1","MCC"]
             ).to_csv(r"model/metrics.csv", index=False)

print("Training complete ✓")
