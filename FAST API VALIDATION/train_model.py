
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

n = 1000
tenure = np.random.randint(1, 73, n)
monthly_charges = np.random.uniform(20, 120, n)
support_calls = np.random.randint(0, 8, n)

score = (
    -0.08 * tenure
    + 0.035 * (monthly_charges - 70)
    + 0.5 * support_calls
    + np.random.normal(0, 0.35, n)
)

y = (score > 1.0).astype(int)

X = np.column_stack(
    (tenure, monthly_charges, support_calls)
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25,
    random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

joblib.dump(model, "churn_model.pkl")

print("Model trained successfully!")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Model saved as churn_model.pkl")
