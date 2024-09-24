import streamlit as st
import pandas as pd
from io import StringIO
import plotly.express as px
import os
import openai

# Securely fetching the OpenAI API key
api_key = os.getenv(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)  # Corrected to use the correct environment variable name
if not api_key:
    st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable in your system.")
    st.stop()  # Halts further execution if the API key is not configured

# Importing the OpenAI library only after confirming the API key is available
openai.api_key = api_key

# Handle the Chroma import with a fallback
try:
    from langchain.vectorstores import Chroma
except ImportError:
    st.warning("Chroma vectorstore is not available, continuing without it.")
    Chroma = None  # Fallback in case Chroma isn't essential for your app

# Try importing other potentially needed modules
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

# File upload functionality and data loading
@st.cache(allow_output_mutation=True)
def load_data():
    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
    if not uploaded_files:
        return pd.DataFrame()
    data_frames = []
    for uploaded_file in uploaded_files:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(stringio)
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

orders_df = load_data()
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
        }).rename(columns={date_column: 'Recency', customer_id_column: 'Frequency', monetary_value_column: 'MonetaryValue'})
        return rfm

    # User input for selecting columns
    col_for_r = st.selectbox('What is the column used for recency - date of purchase?', orders_df.columns)
    col_for_f = st.selectbox('What is the column used to identify a customer ID?', orders_df.columns)
    col_for_m = st.selectbox('What is the column used for order value?', orders_df.columns)

    # Display results
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
