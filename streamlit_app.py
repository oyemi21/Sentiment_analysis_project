import streamlit as st
import requests
import pandas as pd
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Shopease Sentiment Analysis Dashboard", layout="wide")
st.title("Shopease Sentiment Analysis Dashboard")

st.header("single review prediction")

# add a unique key here
user_input = st.text_area("Enter customer review:", key="single_review_input")
if st.button("Predict Sentiment"):
    if user_input.strip() =="":
        st.warning("Please enter a review before predicting.")
    else:
        with st.spinner("Analyzing sentiment..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": user_input}
                )
                if response.status_code == 200:
                   result = response.json()
                   col1, col2 = st.columns(2)
                   col1.metric("sentiment", result['label'])
                   col2.metric("confidence", f"{result['confidence']:.2f}")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"connection error: {e}")


st.divider()

st.header("Batch prediction (csv upload)")
uploaded_file = st.file_uploader("Upload a CSV file with a 'review' column", type=["csv"], key = "batch_file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    if st.button("Run batch prediction", key ="batch_predict"):
        with st.spinner("processing batch predictions..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict/batch",
                    files= {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                )
                if response.status_code == 200:
                    results = pd.DataFrame(response.json())
                    st.success("Batch prediction completed successfully!")
                    st.dataframe(results)

                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label = "download results",
                        data = csv,
                        file_name = "sentiment_predictions.csv",
                        mime = "text/csv",
                        key = "download_results"
                    )
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                    st.error(f"error : {e}")

st.divider()

st.header("Model retraining section")
if st.button("retrain model"):
    try:
        response = requests.post(f"{API_URL}/train")
    
        if response.status_code == 200:
            st.success("Training triggered successfully!")
            st.write(response.json())
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"error: {e}")
