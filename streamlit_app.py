import streamlit as st
import pandas as pd
from io import StringIO
import plotly.express as px
import os
import openai

# Initialize the API key securely
api_key = os.getenv("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")
if not api_key:
    st.error("OpenAI API key is not set. Please configure your environment variable `OPENAI_API_KEY`.")
    st.stop()  # Stop the app if the API key is not configured

openai.api_key = api_key

# Attempt to import optional dependencies
try:
    from langchain.vectorstores import Chroma
except ImportError:
    Chroma = None
    st.warning("Chroma vectorstore is not available, continuing without this component.")

# Function to load data from uploaded CSV files
def load_data():
    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True, help="Upload one or more CSV files.")
    if not uploaded_files:
        return pd.DataFrame()
    data_frames = []
    for uploaded_file in uploaded_files:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        data_frames.append(pd.read_csv(stringio))
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

orders_df = load_data()

# Function to clean and prepare column names
def prepare_columns(dataframe):
    dataframe.columns = dataframe.columns.str.replace('[^0-9a-zA-Z]+', '_').str.lower().str.strip()
    return dataframe

# Check if data is loaded to proceed
if not orders_df.empty:
    orders_df = prepare_columns(orders_df)

    # Function to infer datetime format using OpenAI
    def get_datetime_format(sample_time):
        try:
            response = openai.Completion.create(
                model="gpt-3.5-turbo",
                prompt=f"Provide the strftime format for this datetime string: '{sample_time}'",
                max_tokens=50
            )
            return response.choices[0].text.strip()
        except Exception as e:
            st.error(f"Failed to retrieve datetime format: {str(e)}")
            return None

    # RFM analysis to compute metrics
    def perform_rfm_analysis(df, date_col, customer_id_col, monetary_col):
        date_format = get_datetime_format(df[date_col].dropna().iloc[0])
        if date_format:
            df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
        else:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')  # Fallback if format detection fails

        current_date = df[date_col].max()
        rfm = df.groupby(customer_id_col).agg({
            date_col: lambda x: (current_date - x.max()).dt.days,
            customer_id_col: 'size',
            monetary_col: 'sum'
        }).rename(columns={date_col: 'Recency', customer_id_col: 'Frequency', monetary_col: 'MonetaryValue'})
        return rfm

    # Select columns and perform analysis
    date_column = st.selectbox('Select Date Column for Recency:', orders_df.columns)
    customer_id_column = st.selectbox('Select Customer ID Column for Frequency:', orders_df.columns)
    monetary_value_column = st.selectbox('Select Monetary Value Column:', orders_df.columns)

    rfm_data = perform_rfm_analysis(orders_df, date_column, customer_id_column, monetary_value_column)
    if not rfm_data.empty:
        st.write("RFM Analysis Results", rfm_data)
        fig = px.scatter(rfm_data, x='Recency', y='Frequency', size='MonetaryValue', color='MonetaryValue')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("RFM analysis failed or resulted in empty data.")
else:
    st.info("Please upload data to begin analysis.")
