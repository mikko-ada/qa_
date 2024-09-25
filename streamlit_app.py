import streamlit as st
import pandas as pd
from io import StringIO
import plotly.express as px
import os
import openai

# Securely fetching the OpenAI API key using Streamlit secrets
api_key = st.secrets["openai"]["api_key"] if "openai" in st.secrets else None
if not api_key:
    st.error("OpenAI API key is not set. Please ensure it is configured in the Streamlit Secrets.")
    st.stop()  # Halts further execution if the API key is not configured
else:
    st.success("API Key is configured correctly.")

# Set the API key for OpenAI
openai.api_key = api_key

# Attempt to import Chroma with a fallback
try:
    from langchain.vectorstores import Chroma
except ImportError:
    Chroma = None
    st.warning("Chroma vectorstore is not available, continuing without it.")

# Importing other necessary modules from langchain
try:
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    from langchain.document_loaders import UnstructuredHTMLLoader
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.agents import create_retriever_tool
    from langchain.memory import ConversationBufferMemory
    from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
    from langchain.agents import OpenAIFunctionsAgent
    from langchain.schema.messages import SystemMessage
    from langchain.prompts import MessagesPlaceholder
    from langchain.chains import ConversationalRetrievalChain
except ImportError as e:
    st.error(f"Import error: {e}. Ensure you have the correct version of `langchain` installed. Check documentation for updated import paths.")

# Function to clean and convert column names
def rename_dataset_columns(dataframe):
    dataframe.columns = dataframe.columns.str.replace('[#,@,&,$,(,)]', '').str.replace(' ', '_').str.strip().lower()
    return dataframe

# Function to auto-convert column data types
def convert_datatype(df):
    for c in df.columns[df.dtypes == 'object']:
        try:
            df[c] = pd.to_datetime(df[c])
        except:
            pass
    return df.convert_dtypes()

# File upload functionality moved outside of the cached function
uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)

# Cached function to process data
@st.cache(allow_output_mutation=True)
def process_data(data_frames):
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

# Processing uploaded data if available
if uploaded_files:
    data_frames = [pd.read_csv(StringIO(file.getvalue().decode("utf-8"))) for file in uploaded_files]
    orders_df = process_data(data_frames)
    if not orders_df.empty:
        orders_df = rename_dataset_columns(orders_df)
        orders_df = convert_datatype(orders_df)

        # Function to determine datetime format using GPT
        def get_time_format(time):
            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=[{
                    "role": "system",
                    "content": f"If I had a datetime string like this: {time}, what is the strftime format of this string? Return the strftime format only. Do not return anything else."
                }],
                temperature=0
            )
            return response['choices'][0]['message']['content']

        # RFM Analysis function
        def rfm_analysis(date_column, customer_id_column, monetary_value_column):
            data = orders_df.dropna(subset=[date_column, customer_id_column, monetary_value_column])
            strftime_format = get_time_format(data[date_column].iloc[0])
            data[date_column] = pd.to_datetime(data[date_column], format=strftime_format)
            current_date = data[date_column].max()
            
            rfm = data.groupby(customer_id_column).agg({
                date_column: lambda x: (current_date - x.max()).days,
                customer_id_column: 'count',
                monetary_value_column: 'sum'
            }).rename(columns={
                date_column: 'Recency',
                customer_id_column: 'Frequency',
                monetary_value_column: 'MonetaryValue'
            })
            return rfm

        col_for_r = st.selectbox('What is the column used for recency - date of purchase?', orders_df.columns)
        col_for_f = st.selectbox('What is the column used to identify a customer ID?', orders_df.columns)
        col_for_m = st.selectbox('What is the column used for order value?', orders_df.columns)

        st.title("Key metrics")
        function_response = rfm_analysis(col_for_r, col_for_f, col_for_m)
        if function_response is not None:
            st.metric(label="Average spending per customer", value=function_response["MonetaryValue"].mean())
            st.metric(label="Average number of purchases per customer", value=function_response["Frequency"].mean())
            st.metric(label="Average order value", value=function_response[col_for_m].mean())
            
            st.title("Buying frequency")
            st.write(function_response[["Frequency", "Recency"]])
            fig = px.histogram(function_response, x="Frequency")
            st.plotly_chart(fig)

            st.title("RFM Analysis")
            st.write(function_response)
            st.scatter_chart(data=function_response, x="Recency", y="Frequency", color="MonetaryValue")
else:
    st.warning("Please upload a CSV file to proceed with the analysis.")
