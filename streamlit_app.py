import streamlit as st
from io import StringIO
import re
import plotly.express as px
import numpy as np
import pandas as pd
import os

# Setting up and validating the OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
else:
    import openai
    openai.api_key = api_key

# Attempting to import modules from langchain, handling any missing ones
try:
    from langchain.vectorstores import Chroma
except ImportError:
    Chroma = None
    st.warning("Chroma vectorstore is not available, continuing without it.")

try:
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    from langchain.document_loaders import UnstructuredHTMLLoader
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.agents import create_retriever_tool, OpenAIFunctionsAgent
    from langchain.memory import ConversationBufferMemory
    from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
    from langchain.schema.messages import SystemMessage
    from langchain.prompts import MessagesPlaceholder
    from langchain.chains import ConversationalRetrievalChain
except ImportError as e:
    st.error(f"Import error: {e}. Some functionality may not be available.")

# Cleaning and managing data columns
def rename_dataset_columns(dataframe):
    dataframe.columns = dataframe.columns.str.replace('[#,@,&,$,(,)]', '')
    dataframe.columns = [re.sub(r'%|_%', '_percentage', x) for x in dataframe.columns]
    dataframe.columns = dataframe.columns.str.replace(' ', '_')
    dataframe.columns = [x.lstrip('_') for x in dataframe.columns]
    dataframe.columns = [x.strip() for x in dataframe.columns]
    return dataframe

def convert_datatype(df):
    for c in df.columns[df.dtypes == 'object']:
        df[c] = pd.to_datetime(df[c], errors='coerce')
    df = df.convert_dtypes()
    return df

# File upload and data preparation
uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
def load_data(files):
    data_frames = []
    for uploaded_file in files:
        if uploaded_file is not None:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(stringio)
            data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

orders_df = load_data(uploaded_files)
orders_df = rename_dataset_columns(orders_df) if not orders_df.empty else orders_df

# Handle datetime parsing and formatting with robust error handling
def get_time_format(time_str):
    if api_key:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=[{"role": "system", "content": f"Return the strftime format for this datetime string: {time_str}."}],
                temperature=0
            )
            return response.choices[0].message['content']
        except Exception as e:
            st.error(f"Error during API call: {str(e)}")
            return None
    return None

def rfm_analysis(date_column, customer_id_column, monetary_value_column):
    if not orders_df.empty:
        data = orders_df.dropna(subset=[date_column, customer_id_column, monetary_value_column])
        if not data.empty:
            date_format = get_time_format(data[date_column].iloc[0]) or "%Y-%m-%d"
            data[date_column] = pd.to_datetime(data[date_column], format=date_format, errors='coerce')
        current_date = data[date_column].max()
        rfm = data.groupby(customer_id_column).agg({
            date_column: lambda x: (current_date - x.max()).days,
            customer_id_column: 'count',
            monetary_value_column: 'sum'
        }).rename(columns={date_column: 'Recency', customer_id_column: 'Frequency', monetary_value_column: 'MonetaryValue'})
        return rfm
    return None

if not orders_df.empty:
    col_for_r = st.selectbox('Select the column for Recency (date of purchase):', orders_df.columns)
    col_for_f = st.selectbox('Select the column for Frequency (customer ID):', orders_df.columns)
    col_for_m = st.selectbox('Select the column for Monetary value:', orders_df.columns)
    rfm_data = rfm_analysis(col_for_r, col_for_f, col_for_m)
    if rfm_data is not None and not rfm_data.empty:
        st.write("RFM Analysis Results:", rfm_data)
        fig = px.scatter(rfm_data, x='Recency', y='Frequency', size='MonetaryValue', color='MonetaryValue', title="RFM Scatter Plot")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Failed to perform RFM analysis or insufficient data.")
else:
    st.error("No data available or failed to load. Please upload a valid CSV file.")
