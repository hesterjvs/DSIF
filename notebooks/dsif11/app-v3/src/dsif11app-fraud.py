import sys
import pickle
import numpy as np
import pandas as pd
import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import shap

# Add parent directory to sys.path if needed for config imports
sys.path.insert(0, '../')

# Define relative path and model ID
path_python_material = ".."  # REPLACE WITH YOUR PATH
model_id = "lr1"

# Load the pipeline
with open(f"{path_python_material}/models/{model_id}-pipeline.pkl", "rb") as f:
    loaded_pipeline = pickle.load(f)

# Create FastAPI app
app = FastAPI()

# Define Pydantic models
class Transaction(BaseModel):
    transaction_amount: float
    customer_age: int
    customer_balance: float

class TransactionBatch(BaseModel):
    transactions: List[Transaction]

# -----------------------
# Routes
# -----------------------

@app.get("/feature-importance/")
def get_feature_importance():
    """Return model feature importances from logistic regression."""
    importance = loaded_pipeline[1].coef_[0]
    feature_names = ["transaction_amount", "customer_age", "customer_balance"]
    feature_importance = dict(zip(feature_names, importance))
    return {"feature_importance": feature_importance}

@app.post("/predict/")
def predict_fraud(transaction: Transaction):
    """Predict fraud for a single transaction."""
    data_point = np.array([[
        transaction.transaction_amount,
        transaction.customer_age,
        transaction.customer_balance
    ]])

    # Make predictions
    prediction = loaded_pipeline.predict(data_point)
    probabilities = loaded_pipeline.predict_proba(data_point)
    confidence = probabilities[0].tolist()

    # SHAP values
    X_train_scaled_path = f"{path_python_material}/data/2-intermediate/dsif11-X_train_scaled.npy"
    X_train_scaled = np.load(X_train_scaled_path)
    explainer = shap.LinearExplainer(loaded_pipeline[1], X_train_scaled)
    shap_values = explainer.shap_values(data_point)

    return {
        "fraud_prediction": int(prediction[0]),
        "confidence": confidence,
        "shap_values": shap_values.tolist(),
        "features": ["transaction_amount", "customer_age", "customer_balance"]
    }

@app.post("/predict_batch/")
def predict_batch(batch: TransactionBatch):
    """Predict fraud for a batch of transactions."""
    data_points = [[
        t.transaction_amount,
        t.customer_age,
        t.customer_balance
    ] for t in batch.transactions]

    predictions = loaded_pipeline.predict(data_points)

    results = []
    for t, pred in zip(batch.transactions, predictions):
        results.append({
            "transaction_amount": t.transaction_amount,
            "customer_age": t.customer_age,
            "customer_balance": t.customer_balance,
            "fraud_prediction": int(pred)
        })

    return {"results": results}

@app.post("/predict_automation/")
def predict_automation(files_to_process: List[str]):
    """Batch prediction from files in specified config folders."""
    from conf.conf import landing_path_input_data, landing_path_output_data

    print(f"Files to process (beginning): {files_to_process}")
    if ".DS_Store" in files_to_process:
        files_to_process.remove(".DS_Store")

    input_data = pd.concat([
        pd.read_csv(f"{landing_path_input_data}/{f}")
        for f in files_to_process
    ], ignore_index=True, sort=False)

    input_data["pred_fraud"] = loaded_pipeline.predict(input_data)
    input_data["pred_proba_fraud"] = loaded_pipeline.predict_proba(
        input_data.drop(columns=["pred_fraud"])
    )[:, 1]
    input_data["pred_proba_fraud"] = input_data["pred_proba_fraud"].apply(lambda x: round(x, 5))

    now = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    output_file = f"{landing_path_output_data}/api_tagged_{now}.csv"
    input_data.to_csv(output_file, index=False)

    return {"message": f"Predictions saved in {output_file}"}
