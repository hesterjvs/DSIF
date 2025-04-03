import streamlit as st
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# API URL
api_url = "http://127.0.0.1:8502"

# Page title and header
st.title("Fraud Detection App")

# Display header image
image_path = "../images/dsif header 2.jpeg"
try:
    img = Image.open(image_path)
    st.image(img, use_column_width=True)
except FileNotFoundError:
    st.error(f"Image not found at {image_path}. Please check the file path.")

# ---------------------------
# SINGLE PREDICTION SECTION
# ---------------------------
st.header("Single Transaction Prediction")

transaction_amount = st.number_input("Transaction Amount")
customer_age = st.number_input("Customer Age")
customer_balance = st.number_input("Customer Balance")

data = {
    "transaction_amount": transaction_amount,
    "customer_age": customer_age,
    "customer_balance": customer_balance
}

col1, col2 = st.columns(2)

with col1:
    if st.button("Show Feature Importance"):
        response = requests.get(f"{api_url}/feature-importance/")
        if response.status_code == 200:
            feature_importance = response.json().get('feature_importance', {})
            features = list(feature_importance.keys())
            importance = list(feature_importance.values())

            fig, ax = plt.subplots()
            ax.barh(features, importance)
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance')
            st.pyplot(fig)
        else:
            st.error("Could not fetch feature importance.")

with col2:
    if st.button("Predict and Show SHAP Values"):
        response = requests.post(f"{api_url}/predict/", json=data)
        if response.status_code == 200:
            result = response.json()

            # Display prediction
            prediction = result['fraud_prediction']
            if prediction == 0:
                st.success("Prediction: Not Fraudulent")
            else:
                st.warning("Prediction: Fraudulent")

            # Display SHAP values
            shap_values = np.array(result['shap_values'])[0]
            features = result['features']

            st.subheader("SHAP Values (Impact on Prediction)")
            fig, ax = plt.subplots()
            ax.barh(features, shap_values)
            ax.set_xlabel('SHAP Value')
            st.pyplot(fig)
        else:
            st.error("Prediction failed.")

# Show confidence after SHAP to avoid overlap
if st.button("Show Prediction Confidence"):
    response = requests.post(f"{api_url}/predict/", json=data)
    if response.status_code == 200:
        result = response.json()
        confidence = result['confidence']
        labels = ['Not Fraudulent', 'Fraudulent']

        fig, ax = plt.subplots()
        ax.bar(labels, confidence, color=['green', 'red'])
        ax.set_ylabel('Confidence')
        ax.set_title('Prediction Confidence')
        st.pyplot(fig)
    else:
        st.error("Failed to get prediction confidence.")

# ---------------------------
# BATCH UPLOAD SECTION
# ---------------------------
st.header("Batch Prediction via CSV Upload")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    required_cols = {"transaction_amount", "customer_age", "customer_balance"}
    if not required_cols.issubset(df.columns):
        st.error(f"CSV must include the following columns: {', '.join(required_cols)}")
    else:
        transactions = df[["transaction_amount", "customer_age", "customer_balance"]].to_dict(orient="records")
        payload = {"transactions": transactions}

        with st.spinner("Getting predictions..."):
            try:
                response = requests.post(f"{api_url}/predict_batch/", json=payload)
                if response.status_code == 200:
                    # Handle new-style response
                    results = response.json().get("results", [])
                    predictions = [r["fraud_prediction"] for r in results]
                    df["fraud_prediction"] = predictions

                    st.success("Predictions complete.")
                    st.subheader("Results with Predictions")
                    st.dataframe(df)

                    # Download button
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name="fraud_predictions.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Failed to get batch predictions.")
            except Exception as e:
                st.error(f"Error calling API: {e}")
