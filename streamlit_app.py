import openai
import streamlit as st
from io import StringIO
import re
import plotly.express as px
import numpy as np
import pandas as pd
import os

# Securely fetch the OpenAI API key from environment variables
api_key = os.getenv("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")
if not api_key:
    st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
else:
    openai.api_key = api_key

# Try importing Chroma; handle the ImportError if Chroma is not available
try:
    from langchain.vectorstores import Chroma
except ImportError:
    st.warning("Chroma vectorstore is not available, continuing without it.")
    Chroma = None

# Since 'langchain_community' is causing import errors, let's exclude it and handle other imports
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
    st.error(f"Import error: {e}. Check if the required modules are available in your langchain installation.")

# Function to clean column names
def rename_dataset_columns(dataframe):
    dataframe.columns = dataframe.columns.str.replace('[#,@,&,$,(,)]', '')
    dataframe.columns = [re.sub(r'%|_%', '_percentage', x) for x in dataframe.columns]
    dataframe.columns = dataframe.columns.str.replace(' ', '_')
    dataframe.columns = [x.lstrip('_') for x in dataframe.columns]
    dataframe.columns = [x.strip() for x in dataframe.columns]
    return dataframe

# Function to auto-convert column data types
def convert_datatype(df):
    for c in df.columns[df.dtypes == 'object']:
        try:
            df[c] = pd.to_datetime(df[c])
        except:
            pass
    df = df.convert_dtypes()
    return df

# File upload functionality
uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)

@st.cache
def load_data(files):
    data_frames = []
    for uploaded_file in files:
        if uploaded_file is not None:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(stringio)
            data_frames.append(df)
    if data_frames:
        combined_df = pd.concat(data_frames, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no files are uploaded

orders_df = load_data(uploaded_files)
orders_df = rename_dataset_columns(orders_df) if not orders_df.empty else orders_df

# Function to determine datetime format using GPT
def get_time_format(time):
    if api_key:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=[
                    {
                        "role": "system",
                        "content": f"If I had a datetime string like this: {time}, what is the strftime format of this string? Return the strftime format only. Do not return anything else."
                    },
                ],
                temperature=0
            )
            return response['choices'][0]['message']['content']
        except openai.error.OpenAIError as e:
            st.error(f"Failed to communicate with OpenAI's API: {e}")
            return None
    else:
        return "%Y-%m-%d"  # Default format if API key is not set

# RFM Analysis function
def rfm_analysis(date_column, customer_id_column, monetary_value_column):
    if not orders_df.empty:
        data = orders_df.dropna(subset=[date_column, customer_id_column, monetary_value_column])
        strfttime = get_time_format(data[date_column].iloc[0]) if not data.empty else "%Y-%m-%d"
        data[date_column] = pd.to_datetime(data[date_column], format=strfttime)
        data[customer_id_column] = data[customer_id_column].astype(str)
        current_date = data[date_column].max()

        rfm = data.groupby(customer_id_column).agg({
            date_column: lambda x: (current_date - x.max()).days,
            customer_id_column: 'count',
            monetary_value_column: 'sum'
        })

        rfm.rename(columns={
            date_column: 'Recency',
            customer_id_column: 'Frequency',
            monetary_value_column: 'MonetaryValue'
        }, inplace=True)
        return rfm
    else:
        return None

# Streamlit UI Components for column selection
if not orders_df.empty:
    col_for_r = st.selectbox('What is the column used for recency - date of purchase?', orders_df.columns)
    col_for_f = st.selectbox('What is the column used to identify a customer ID?', orders_df.columns)
    col_for_m = st.selectbox('What is the column used for order value?', orders_df.columns)

    # RFM Analysis Execution
    st.title("RFM Analysis Results")
    function_response = rfm_analysis(col_for_r, col_for_f, col_for_m)
    if function_response is not None and not function_response.empty:
        st.write("RFM Analysis:", function_response)

        # Displaying key metrics using Streamlit
        st.metric(label="Average spending per customer", value=function_response["MonetaryValue"].mean())
        st.metric(label="Average number of purchases per customer", value=function_response["Frequency"].mean())
        st.metric(label="Average days since last purchase", value=function_response["Recency"].mean())

        # Visualizations for RFM segments
        st.title("RFM Segments Visualization")
        fig = px.scatter(function_response, x='Recency', y='Frequency', size='MonetaryValue', color='MonetaryValue',
                         hover_name=function_response.index, size_max=60)
        st.plotly_chart(fig, use_container_width=True)

        # Histograms for Frequency and Recency
        fig_freq = px.histogram(function_response, x='Frequency', nbins=20, title='Frequency Distribution')
        st.plotly_chart(fig_freq, use_container_width=True)
        fig_rec = px.histogram(function_response, x='Recency', nbins=20, title='Recency Distribution')
        st.plotly_chart(fig_rec, use_container_width=True)
    else:
        st.error("RFM Analysis could not be performed. Please ensure data is correctly formatted and sufficient.")

else:
    st.error("No data available. Please upload CSV files to proceed.")
