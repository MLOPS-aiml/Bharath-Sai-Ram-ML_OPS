
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime

np.random.seed(42)

# 1. Generate sample customer data
n = 1000

data = pd.DataFrame({
    "tenure": np.random.randint(1, 73, n),
    "monthly_charges": np.random.uniform(20, 150, n),
    "support_calls": np.random.randint(0, 10, n)
})

score = (
    -0.08 * data["tenure"]
    + 0.035 * (data["monthly_charges"] - 70)
    + 0.5 * data["support_calls"]
    + np.random.normal(0, 0.35, n)
)

data["churn"] = (score > 1.0).astype(int)

X = data[["tenure", "monthly_charges", "support_calls"]]
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 2. Train model
model = RandomForestClassifier(
    n_estimators=100, random_state=42
)
model.fit(X_train, y_train)

# 3. Measure baseline performance
baseline_predictions = model.predict(X_test)
baseline_accuracy = accuracy_score(
    y_test, baseline_predictions
)

print("Baseline Accuracy:", round(baseline_accuracy, 4))

# 4. Simulate data drift
drifted_X = X_test.copy()
drifted_X["monthly_charges"] += 40
drifted_X["support_calls"] = np.minimum(
    drifted_X["support_calls"] + 5, 20
)

# Labels from the original test set are retained
# to compare predictions against the same outcomes.
drift_predictions = model.predict(drifted_X)
drift_accuracy = accuracy_score(y_test, drift_predictions)

print("Accuracy After Drift:", round(drift_accuracy, 4))

# 5. Log predictions
log = pd.DataFrame({
    "timestamp": datetime.now().isoformat(),
    "actual_churn": y_test.to_numpy(),
    "baseline_prediction": baseline_predictions,
    "drift_prediction": drift_predictions
})

log.to_csv("prediction_logs.csv", index=False)
print("Prediction logs saved to prediction_logs.csv")

# 6. Plot comparison
plt.bar(
    ["Before Drift", "After Drift"],
    [baseline_accuracy, drift_accuracy]
)
plt.ylabel("Accuracy")
plt.title("Model Performance Before and After Data Drift")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("drift_comparison.png")
plt.close()

print("Graph saved as drift_comparison.png")
