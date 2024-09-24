import streamlit as st
import pandas as pd
from io import StringIO
import plotly.express as px
import os
import openai

# Setup and check the OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
else:
    openai.api_key = api_key

# Function to load and clean CSV data
def load_data(files):
    if files:
        data_frames = []
        for uploaded_file in files:
            if uploaded_file is not None:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                df = pd.read_csv(stringio)
                data_frames.append(df)
        return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()
    return pd.DataFrame()

# File upload section
uploaded_files = st.file_uploader("Choose one or more CSV files", accept_multiple_files=True)
orders_df = load_data(uploaded_files)

# Clean column names
def clean_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace('[^a-zA-Z0-9_]', '', regex=True).str.replace(' ', '_')
    return df

if not orders_df.empty:
    orders_df = clean_columns(orders_df)

# Function to determine the datetime format using OpenAI API
def get_time_format(time_str):
    if api_key:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": f"What is the strftime format for this datetime string: {time_str}?"}],
                temperature=0
            )
            return response.choices[0].message['content']
        except Exception as e:
            st.error(f"Failed to get date format from OpenAI: {str(e)}")
            return None
    else:
        st.warning("OpenAI API key is not configured.")
        return None

# RFM Analysis Function
def rfm_analysis(df, date_col, customer_id_col, monetary_value_col):
    if df.empty:
        st.error("No data to analyze.")
        return pd.DataFrame()
    
    sample_date = df[date_col].dropna().iloc[0]
    date_format = get_time_format(sample_date)
    if date_format:
        df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    current_date = df[date_col].max()
    rfm = df.groupby(customer_id_col).agg(
        Recency=(date_col, lambda x: (current_date - x.max()).days),
        Frequency=(customer_id_col, 'count'),
        MonetaryValue=(monetary_value_col, 'sum')
    )
    return rfm

# User input for column names
if not orders_df.empty:
    date_col = st.selectbox('Select Date Column for Recency:', orders_df.columns, index=orders_df.columns.get_loc("date") if "date" in orders_df.columns else 0)
    customer_id_col = st.selectbox('Select Customer ID Column for Frequency:', orders_df.columns, index=orders_df.columns.get_loc("customer_id") if "customer_id" in orders_df.columns else 0)
    monetary_value_col = st.selectbox('Select Monetary Value Column:', orders_df.columns, index=orders_df.columns.get_loc("sales") if "sales" in orders_df.columns else 0)

    # Perform RFM analysis
    rfm_data = rfm_analysis(orders_df, date_col, customer_id_col, monetary_value_col)
    if not rfm_data.empty:
        st.write("RFM Analysis Results:", rfm_data)
        fig = px.scatter(rfm_data, x='Recency', y='Frequency', size='MonetaryValue', color='MonetaryValue', hover_name=rfm_data.index)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("RFM analysis failed or produced no output.")
else:
    st.warning("Upload CSV files to begin analysis.")

