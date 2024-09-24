# First
import openai
import streamlit as st
from io import StringIO
import re
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import json
import math

# Handle the Chroma import with a fallback
try:
    from langchain.vectorstores import Chroma
except ImportError:
    st.warning("Chroma vectorstore is not available, continuing without it.")
    Chroma = None  # Fallback in case Chroma isn't essential for your app

from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.chains import ConversationalRetrievalChain

# Ensure your OpenAI API key is securely handled
openai.api_key = "your-api-key-here"

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

@st.cache_data
def load_data(files):
    for uploaded_file in files:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        orders_df = pd.read_csv(stringio)
        return orders_df

orders_df = load_data(uploaded_files)
orders_df = rename_dataset_columns(orders_df)

# Function to determine datetime format using GPT
def get_time_format(time):
    return openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": f"If I had a datetime string like this: {time}, what is the strftime format of this string? Return the strftime format only. Do not return anything else."
            },
        ],
        temperature=0
    )

# RFM Analysis function
def rfm_analysis(date_column, customer_id_column, monetary_value_column):
    data = orders_df
    data = data.dropna(subset=[date_column, customer_id_column, monetary_value_column])
    # Ensure the date column is in datetime format
    strfttime = get_time_format(data[date_column].iloc[0]).choices[0]["message"]["content"]
    data[date_column] = pd.to_datetime(data[date_column], format=strfttime)
    data[customer_id_column] = data[customer_id_column].astype(str)
    current_date = data[date_column].max()

    rfm = data.groupby(customer_id_column).agg({
        date_column: lambda x: (x.max() - current_date).days,
        customer_id_column: 'count',
        monetary_value_column: 'sum'
    })

    rfm.rename(columns={
        date_column: 'Recency',
        customer_id_column: 'Frequency',
        monetary_value_column: 'MonetaryValue'
    }, inplace=True)
    return rfm

# Function to calculate custom quantiles
def custom_quantiles(rfm, r_bins, f_bins):
    r_quantiles_list = [(x + 1) / (r_bins) for x in range(0, r_bins)]
    f_quantiles_list = [(x + 1) / (f_bins) for x in range(0, f_bins)]
    r_quantiles = rfm.quantile(q=r_quantiles_list)
    f_quantiles = rfm.quantile(q=f_quantiles_list)

    def rfm_scoring(input, bins, quantile_list, parameter, quantiles):
        value = ""
        if parameter == "Recency":
            for q in reversed(range(len(quantile_list))):
                if input <= quantiles[parameter][quantile_list[q]]:
                    value = q + 1
                else:
                    break
            return value
        elif parameter == "Frequency":
            for q in reversed(range(0, len(quantile_list))):
                if input <= quantiles[parameter][quantile_list[q]]:
                    value = q + 1
                else:
                    break
            return value

    rfm['R'] = rfm['Recency'].apply(lambda input: rfm_scoring(input, r_bins, r_quantiles_list, 'Recency', r_quantiles))
    rfm['F'] = rfm['Frequency'].apply(lambda input: rfm_scoring(input, f_bins, f_quantiles_list, 'Frequency', f_quantiles))

    return rfm

# App Logic for selecting columns
col_for_r = st.selectbox('What is the column used for recency - date of purchase?', orders_df.columns)
col_for_f = st.selectbox('What is the column used to identify a customer ID?', orders_df.columns)
col_for_m = st.selectbox('What is the column used for order value?', orders_df.columns)

st.title("Key metrics")
function_response = rfm_analysis(
    date_column=col_for_r,
    customer_id_column=col_for_f,
    monetary_value_column=col_for_m
)

if function_response is not None:
    st.metric(label="Average spending per customer", value=function_response["MonetaryValue"].mean())
    st.metric(label="Average number of purchases per customer", value=function_response["Frequency"].mean())
    st.metric(label="Average order value", value=orders_df[col_for_m].mean())

st.title("Buying frequency")
if function_response is not None:
    st.write(function_response[["Frequency", "Recency"]])
    fig = px.histogram(function_response, x="Frequency")
    st.write(fig)

st.title("RFM Analysis")
company_desc = st.text_input('Description of company', 'an ecommerce company selling goods')

# Plot RFM results
if function_response is not None:
    st.write(function_response)
    st.scatter_chart(data=function_response, x="Recency", y="Frequency", color="MonetaryValue")
    
    r_iqr = abs(function_response["Recency"].quantile(0.75) - function_response["Recency"].quantile(0.25))
    r_bin_width = math.ceil(r_iqr / 2 + 0.00001)
    f_iqr = function_response["Frequency"].quantile(0.75) - function_response["Frequency"].quantile(0.25)
    f_bin_width = math.ceil(f_iqr / 2 + 0.00001)

    r_slider = st.slider('What is the bin width you would like for recency?', 1, 300, r_bin_width)
    f_slider = st.slider('What is the bin width you would like for frequency?', 1, 20, f_bin_width)

    r_max_suggest = function_response["Recency"].quantile(0.25) - 1.5 * r_iqr
    f_max_suggest = function_response["Frequency"].quantile(0.75) + 1.5 * f_iqr
    r_max_slider = st.slider('What is the min you would like for recency?', -500, 0, math.ceil(r_max_suggest + 0.00001))
    f_max_slider = st.slider('What is the max you would like for frequency?', 1, 20, math.ceil(f_max_suggest + 0.00001))

    fig = go.Figure(go.Histogram2d(
        x=function_response["Recency"],
        y=function_response["Frequency"],
        histfunc="count",
        xbins=dict(start=r_max_slider, end=0, size=r_slider),
        ybins=dict(start=0, end=f_max_slider, size=f_slider),
        texttemplate="%{z}"
    ))

    fig.update_layout(
        xaxis_title='Recency',
        yaxis_title='Frequency',
        bargap=0.1,
    )
    st.plotly_chart(fig, use_container_width=True)

    # AOV (Average Order Value) plot
    fig2 = go.Figure(go.Histogram2d(
        x=function_response["Recency"],
        y=function_response["Frequency"],
        z=function_response["MonetaryValue"],
        histfunc="avg",
        xbins=dict(start=r_max_slider, end=0, size=r_slider),
        ybins=dict(start=0, end=f_max_slider, size=f_slider),
        texttemplate="%{z}"
    ))

    fig2.update_layout(
        xaxis_title='Recency',
        yaxis_title='Frequency',
        bargap=0.1,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Segmentation logic based on RFM
    recency_bins = np.arange(start=r_max_slider, stop=r_slider, step=r_slider)
    frequency_bins = np.arange(start=0, stop=f_max_slider + f_slider, step=f_slider)

    hist, xedges, yedges = np.histogram2d(
        x=function_response['Frequency'], y=function_response['Recency'], bins=[frequency_bins, recency_bins]
    )

    hist_df = pd.DataFrame(hist, index=xedges[:-1], columns=yedges[:-1])
    hist_df = hist_df.loc[:, (hist_df != 0).any(axis=0)].loc[(hist_df != 0).any(axis=1)]
    st.write(hist_df)

    get_gpt_segments = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": f"If I had a dataframe below like this: \n {hist_df} \n where the headers represent the number of days since the customer last purchased, and the index represents the number of times the customers purchased (frequency), come up with distinct and non-overlapping segmentation conditions."
            },
        ],
        temperature=0.7
    )

    st.title("GPT Suggested Segments")
    for x in json.loads(get_gpt_segments["choices"][0]["message"]["content"])["segments"]:
        try:
            st.subheader(f"{x['name']} \n {x['r_condition']}, {x['f_condition']}")
            st.write(f"Description of segment: {x['desc']}")
            st.write("Number of contacts: ", len(function_response.query(f"({x['r_condition']}) & ({x['f_condition']})")))
            st.write(function_response.query(f"({x['r_condition']}) & ({x['f_condition']})"))
        except:
            pass
