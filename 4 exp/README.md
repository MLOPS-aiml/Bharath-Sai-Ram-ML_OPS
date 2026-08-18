# Loan Approval Prediction (MLOps Experiment)

This repository contains a machine learning experiment for predicting loan approvals based on user demographics, financial history, and asset details. The project implements a Classification model and integrates **MLflow** for experiment tracking and model versioning.

---

## 📂 Project Structure

```text
D:\ML OPS\4 exp\
│
├── Loan Approval prediction.ipynb  # Jupyter Notebook containing the end-to-end ML pipeline
├── loan_approval_dataset.csv       # Dataset copy containing loan applications
└── README.md                       # Project documentation (this file)
```

> [!NOTE]
> The Jupyter Notebook references a dataset located at `D:\datasets\loan_approval_dataset.csv`. Make sure to place the dataset there or update the file path in the notebook to run it.

---

## 📊 Dataset Details

The dataset `loan_approval_dataset.csv` contains details of loan applications with the following key attributes:

### Target Feature
*   **`loan_status`**: The approval status of the loan application (`Approved` or `Rejected`).

### Input Features
*   **Demographics & Personal Details**:
    *   `no_of_dependents`: Number of dependents the applicant has.
    *   `education`: Education status (`Graduate` or `Not Graduate`).
    *   `self_employed`: Employment status (`Yes` or `No`).
*   **Financial Details**:
    *   `income_annum`: Annual income of the applicant.
    *   `loan_amount`: Total loan amount requested.
    *   `loan_term`: Repayment duration of the loan in years.
    *   `cibil_score`: Credit score (CIBIL) of the applicant.
*   **Assets Valuation**:
    *   `residential_assets_value`: Value of residential properties owned.
    *   `commercial_assets_value`: Value of commercial properties owned.
    *   `luxury_assets_value`: Value of luxury items owned.
    *   `bank_asset_value`: Value of bank assets.

---

## ⚙️ Model Pipeline & MLflow Integration

The experiment uses a **Random Forest Classifier** trained and tracked under an active MLflow run.

### Pipeline Steps:
1.  **Data Preprocessing**:
    *   Cleans column names by removing trailing/leading whitespaces.
    *   Converts categorical columns (`education`, `self_employed`) into numerical values using one-hot encoding (`pd.get_dummies(..., drop_first=True)`).
2.  **Train-Test Split**:
    *   Splits dataset into 80% training data and 20% test data (`random_state=42`).
3.  **Model Configuration**:
    *   Algorithm: `RandomForestClassifier`
    *   Hyperparameters: `n_estimators=100`, `max_depth=5`, `random_state=42`
4.  **Experiment Tracking**:
    *   Logs model hyperparameters (`n_estimators` and `max_depth`).
    *   Logs accuracy metrics.
    *   Registers and saves the model artifact (`random_forest_model`) using `mlflow.sklearn.log_model()`.

### Performance
*   **Model Accuracy**: `~96.49%` on the test set.

---

## 🚀 How to Run the Project

### 1. Install Dependencies
Make sure you have Python and Jupyter installed. Then, install the required packages:

```bash
pip install pandas scikit-learn mlflow
```

### 2. Run the Notebook
Open the notebook in your Jupyter environment:

```bash
jupyter notebook "Loan Approval prediction.ipynb"
```

Run all cells sequentially to trigger model training and start experiment tracking.

### 3. View MLflow Dashboard
To view parameters, metrics, and models saved during execution, run the MLflow UI in the project directory:

```bash
mlflow ui
```

Then open your browser and navigate to `http://localhost:5000` or the address printed in the console.
