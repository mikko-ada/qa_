import streamlit as st
import pandas as pd
from io import StringIO
import plotly.express as px
import os
import openai

# Setup and validating the OpenAI API key
api_key = os.getenv("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")
if not api_key:
    st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
else:
    openai.api_key = api_key

# Attempting to import modules from langchain, handling any missing ones
try:
    from langchain.vectorstores import Chroma
except ImportError:
    Chroma = None
    st.warning("Chroma vectorstore is not available, continuing without it.")

try:
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
except ImportError as e:
    st.error(f"Import error: {e}. Some functionality may not be available.")

# Function to clean and convert dataset column names
def rename_dataset_columns(dataframe):
    dataframe.columns = dataframe.columns.str.replace('[#,@,&,$,%,(,)]', '')
    dataframe.columns = dataframe.columns.str.replace(' ', '_')
    dataframe.columns = dataframe.columns.str.lower()
    return dataframe

# Function to load and concatenate multiple CSV files
def load_data(files):
    if files:
        data_frames = []
        for uploaded_file in files:
            if uploaded_file is not None:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                df = pd.read_csv(stringio)
                data_frames.append(df)
        if data_frames:
            return pd.concat(data_frames, ignore_index=True)
    return pd.DataFrame()

uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
orders_df = load_data(uploaded_files)
if not orders_df.empty:
    orders_df = rename_dataset_columns(orders_df)

# Function to get the datetime format from a datetime string using OpenAI
def get_time_format(time_str):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": f"What is the strftime format for this datetime string? '{time_str}'"}],
            temperature=0
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"API call to OpenAI failed: {e}")
        return None

# RFM Analysis function
def rfm_analysis(df, date_col, customer_id_col, monetary_value_col):
    if df.empty:
        return None

    # Attempt to convert date column to datetime
    sample_date = df[date_col].dropna().iloc[0]
    date_format = get_time_format(sample_date)
    if date_format:
        df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors='coerce')

    current_date = df[date_col].max()
    rfm = df.groupby(customer_id_col).agg(
        Recency=(date_col, lambda x: (current_date - x.max()).days),
        Frequency=(customer_id_col, 'count'),
        MonetaryValue=(monetary_value_col, 'sum')
    ).reset_index()

    return rfm

if not orders_df.empty:
    col_for_r = st.selectbox('Select the column for Recency (date of purchase):', orders_df.columns)
    col_for_f = st.selectbox('Select the column for Frequency (customer ID):', orders_df.columns)
    col_for_m = st.selectbox('Select the column for Monetary value:', orders_df.columns)

    rfm_data = rfm_analysis(orders_df, col_for_r, col_for_f, col_for_m)
    if rfm_data is not None and not rfm_data.empty:
        st.write("RFM Analysis Results:", rfm_data)
        fig = px.scatter(rfm_data, x='Recency', y='Frequency', size='MonetaryValue', color='MonetaryValue', hover_name='customer_id', size_max=60)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Failed to perform RFM analysis or insufficient data.")
else:
    st.error("No data available or failed to load. Please upload a valid CSV file.")
