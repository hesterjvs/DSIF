

import streamlit as st
import requests
import httpx


st.title("Fraud Detection App")

transaction_amount = st.number_input("Transaction Amount")
customer_age = st.number_input("Customer Age")
customer_balance = st.number_input("Customer Balance")


if st.button("Predict"):
    response = response = requests.post("http://127.0.0.1:8502/predict/",
                          json={
                              "transaction_amount": transaction_amount,
                              "customer_age": customer_age,
                              "customer_balance": customer_balance
                          })
    st.write(response.status_code, response.text)

