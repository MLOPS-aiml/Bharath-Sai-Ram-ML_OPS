# Model Serialization Experiment

This project demonstrates the process of training a machine learning model and serializing (saving and loading) it using two popular Python libraries: **Pickle** and **Joblib**. 

Model serialization is a crucial step in ML Ops, allowing you to persist trained models to disk so they can be deployed, shared, or reused without retraining.

---

## Project Structure

* **[_Model Serialization .ipynb](file:///D:/ML%20OPS/3%20exp/_Model%20Serialization%20.ipynb)**: The main Jupyter Notebook containing the training, evaluation, and serialization code.
* **`iris_model_pickle.pkl`** (generated after run): The serialized Decision Tree model using `pickle`.
* **`iris_model_joblib.pkl`** (generated after run): The serialized Decision Tree model using `joblib`.

---

## Workflow Overview

The notebook follows these steps:

1. **Load Dataset**: Imports the classic Iris flower dataset using `sklearn.datasets.load_iris`.
2. **Train/Test Split**: Splits the dataset into training (80%) and testing (20%) sets to validate the model's accuracy.
3. **Train Model**: Trains a `DecisionTreeClassifier` from `scikit-learn` on the training data.
4. **Evaluate**: Tests model predictions against the test dataset, printing the classification accuracy (e.g., `1.0` or 100% accuracy on this small split).
5. **Serialization & Verification**:
   * **Pickle**: Saves the model to `iris_model_pickle.pkl` using `pickle.dump()`, loads it back with `pickle.load()`, and makes a sample prediction to verify.
   * **Joblib**: Saves the model to `iris_model_joblib.pkl` using `joblib.dump()`, loads it back with `joblib.load()`, and makes a sample prediction to verify.

---

## Prerequisites & Installation

To run the notebook, ensure you have Python installed along with the required libraries. You can install the dependencies via pip:

```bash
pip install pandas scikit-learn joblib notebook
```

---

## How to Run

1. Open your terminal or command prompt in this directory.
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open `_Model Serialization .ipynb` and run all cells to train and serialize the model.

---

## Pickle vs. Joblib

| Feature | `pickle` | `joblib` |
| :--- | :--- | :--- |
| **Standard Library** | Yes (built-in Python module) | No (requires external installation) |
| **Use Case** | General-purpose Python object serialization | Optimized for large numerical/numpy arrays |
| **Efficiency** | Good for small/standard objects | Highly efficient for models containing large arrays (like Random Forests, SVMs, or Deep Neural Networks) |
