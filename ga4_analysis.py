import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import csv

st.title("Customer Journey Pathway")

st.title("Purchase Funnel")
purchase_funnel_file = st.file_uploader("Upload your purchase funnel CSV file")

if purchase_funnel_file:

    lines = [line.decode('utf-8') for line in purchase_funnel_file.readlines()]

    processed_lines = [line.split(',') for line in lines]  # Customize this as needed
    #df = pd.DataFrame(processed_lines)
    for index, line in enumerate(processed_lines):
        if "Start date:" in line[0]:
            st.subheader(line[0])
            st.subheader(processed_lines[index+1][0])
            for i, x in enumerate(processed_lines[index:]):
                if len(x[0].strip()) == 0:
                    df = pd.DataFrame(processed_lines[index + 2: index + i])
                    #new_header = df.iloc[0]  # Grab the first row for the header
                    df = df[1:]  # Take the data less the header row
                    #df.columns = new_header  # Set the header row as the df header
                    df = df.set_index(df.columns[0])
                    df.index.name = 'category'  # Rename the index
                    df = df.iloc[:, 0::2]
                    df.columns = [f"Step {num}" for num in list(range(1, df.shape[1]+1))]
                    df = df.apply(pd.to_numeric, errors='coerce')
                    df1 = df.iloc[:, 0:-1]
                    df1.columns = [f"Drop off {num}" for num in list(range(1, df1.shape[1]+1))]
                    df2 = df.iloc[:, 1:]
                    df2.columns = [f"Drop off {num}" for num in list(range(1, df2.shape[1]+1))]
                    differences_df = df1 - df2
                    diff_p_df = ((df1 - df2) / df1 * 100).round(2)
                    fig = go.Figure(data=go.Heatmap(
                        z=diff_p_df,
                        x=diff_p_df.columns,
                        y=diff_p_df.index,
                        text="Total drop off: " + differences_df.astype(str) + "<br>" + "% drop off: " + diff_p_df.astype(str) + "%",
                        texttemplate="%{text}",
                        ))
                    st.write(df)
                    #st.write(differences_df)
                    #st.write(diff_p_df)
                    st.write(fig)
                    break
                elif (index + i == len(processed_lines) - 1):
                    df = pd.DataFrame(processed_lines[index + 2: index + i + 1])
                    # new_header = df.iloc[0]  # Grab the first row for the header
                    df = df[1:]  # Take the data less the header row
                    # df.columns = new_header  # Set the header row as the df header
                    df = df.set_index(df.columns[0])
                    df.index.name = 'category'  # Rename the index
                    df = df.iloc[:, 0::2]
                    df.columns = [f"Step {num}" for num in list(range(1, df.shape[1] + 1))]
                    df = df.apply(pd.to_numeric, errors='coerce')
                    df1 = df.iloc[:, 0:-1]
                    df1.columns = [f"Drop off {num}" for num in list(range(1, df1.shape[1] + 1))]
                    df2 = df.iloc[:, 1:]
                    df2.columns = [f"Drop off {num}" for num in list(range(1, df2.shape[1] + 1))]
                    differences_df = df1 - df2
                    diff_p_df = ((df1 - df2) / df1 * 100).round(2)
                    fig = go.Figure(data=go.Heatmap(
                        z=diff_p_df,
                        x=diff_p_df.columns,
                        y=diff_p_df.index,
                        text="Total drop off: " + differences_df.astype(str) + "<br>" + "% drop off: " + diff_p_df.astype(
                            str) + "%",
                        texttemplate="%{text}",
                    ))
                    st.write(df)
                    # st.write(differences_df)
                    # st.write(diff_p_df)
                    st.write(fig)

#purchase_funnel_df = pd.read_csv(purchase_funnel_file)

st.title("Traffic")
traffic_file = st.file_uploader("Upload your traffic CSV file")

if traffic_file:

    traffic_lines = [line.decode('utf-8') for line in traffic_file.readlines()]
    traffic_reader = csv.reader(traffic_lines)
    traffic_processed_lines = list(traffic_reader)

    for index, line in enumerate(traffic_processed_lines):
        if "Start date:" in line[0]:
            st.subheader(line[0])
            st.subheader(traffic_processed_lines[index+1][0])
            for i, x in enumerate(traffic_processed_lines[index:]):
                if len(x[0].strip()) == 0:
                    traffic_df = pd.DataFrame(traffic_processed_lines[index + 2: index + i])
                    new_header = traffic_df.iloc[0]  # Grab the first row for the header
                    traffic_df = traffic_df[1:]  # Take the data less the header row
                    traffic_df.columns = new_header  # Set the header row as the df header
                    traffic_df = traffic_df.apply(pd.to_numeric, errors='ignore')
                    #st.write(traffic_df)
                    traffic_pt = pd.pivot_table(data=traffic_df, index="First user source / medium", values=["Engaged sessions", "Conversions"], aggfunc="sum")
                    traffic_pt["CR Ratio"] = (traffic_pt["Conversions"] / traffic_pt["Engaged sessions"]).round(2)
                    filtered_pt = traffic_pt[traffic_pt["Engaged sessions"] > 100].sort_values(by="Engaged sessions", ascending=False)
                    st.write(traffic_df)
                    st.write(filtered_pt)
                    break
                elif (index + i == len(traffic_processed_lines) - 1):
                    traffic_df_2 = pd.DataFrame(traffic_processed_lines[index + 2: index + i])
                    new_header = traffic_df_2.iloc[0]  # Grab the first row for the header
                    traffic_df_2 = traffic_df_2[1:]  # Take the data less the header row
                    traffic_df_2.columns = new_header  # Set the header row as the df header
                    traffic_df_2 = traffic_df_2.apply(pd.to_numeric, errors='ignore')
                    traffic_pt_2 = pd.pivot_table(data=traffic_df_2, index="First user source / medium",
                                                values=["Engaged sessions", "Conversions"], aggfunc="sum")
                    traffic_pt_2["CR Ratio"] = (traffic_pt_2["Conversions"] / traffic_pt_2["Engaged sessions"]).round(2)
                    filtered_pt_2 = traffic_pt_2[traffic_pt["Engaged sessions"] > 100].sort_values(by="Engaged sessions", ascending=False)
                    st.write(traffic_df_2)
                    st.write(filtered_pt_2)
                    st.subheader("Comparison of source/medium between time periods (Absolute)")
                    traffic_difference_pt = filtered_pt - filtered_pt_2
                    st.write(traffic_difference_pt.sort_values(by="Engaged sessions", ascending=False))
                    sourcemed_fig = go.Figure(
                        data=[
                            go.Scatter(
                                x=filtered_pt_2["Engaged sessions"],
                                y=filtered_pt_2.index,
                                mode="markers",
                                name="before",
                                marker=dict(
                                    color="green",
                                    size=10
                                )

                            ),
                            go.Scatter(
                                x=filtered_pt["Engaged sessions"],
                                y=filtered_pt.index,
                                mode="markers",
                                name="after",
                                marker=dict(
                                    color="blue",
                                    size=10
                                )
                            ),
                        ]
                    )
                    st.write(sourcemed_fig)
                    st.subheader("Comparison of source/medium between time periods (%)")
                    traffic_difference_p_pt = (filtered_pt - filtered_pt_2)/filtered_pt_2
                    st.write(traffic_difference_p_pt.sort_values(by="Engaged sessions", ascending=False))
                    st.subheader("Comparison of campaigns between time periods")
                    sourcemed_option = st.selectbox(
                        'Select source/medium', filtered_pt.index)
                    campaign_pt = pd.pivot_table(traffic_df[traffic_df["First user source / medium"] == sourcemed_option],
                                                 index="Session campaign",
                                                 values=["Engaged sessions", "Conversions"],
                                                 aggfunc="sum"
                                                 )
                    campaign_pt["CR Ratio"] = campaign_pt["Conversions"] / campaign_pt["Engaged sessions"]
                    campaign_pt_2 = pd.pivot_table(traffic_df_2[traffic_df_2["First user source / medium"] == sourcemed_option],
                                                 index="Session campaign",
                                                 values=["Engaged sessions", "Conversions"],
                                                 aggfunc="sum"
                                                 )
                    campaign_pt_2["CR Ratio"] = campaign_pt_2["Conversions"] / campaign_pt_2["Engaged sessions"]
                    st.write(campaign_pt - campaign_pt_2)


st.title("SKU")
sku_file = st.file_uploader("Upload your SKU CSV file")

if sku_file:

    sku_lines = [line.decode('utf-8') for line in sku_file.readlines()]
    sku_reader = csv.reader(sku_lines)
    sku_processed_lines = list(sku_reader)

    for index, line in enumerate(sku_processed_lines):
        if "Start date:" in line[0]:
            st.subheader(line[0])
            st.subheader(sku_processed_lines[index+1][0])
            for i, x in enumerate(sku_processed_lines[index:]):
                if len(x[0].strip()) == 0:
                    sku_df = pd.DataFrame(sku_processed_lines[index + 2: index + i])
                    new_header = sku_df.iloc[0]  # Grab the first row for the header
                    sku_df = sku_df[1:]  # Take the data less the header row
                    sku_df.columns = new_header  # Set the header row as the df header
                    sku_df = sku_df.apply(pd.to_numeric, errors='ignore')
                    #st.write(sku_df)
                    sku_pt = pd.pivot_table(data=sku_df, index="Item name", values=["Items viewed", "Items purchased"], aggfunc="sum")
                    sku_pt["CR Ratio"] = (sku_pt["Items purchased"] / sku_pt["Items viewed"]).round(2)
                    sku_filtered_pt = sku_pt[sku_pt["Items viewed"] > 100].sort_values(by="Items viewed", ascending=False)
                    st.write(sku_df)
                    st.write(sku_filtered_pt)
                    break
                elif (index + i == len(sku_processed_lines) - 1):
                    sku_df_2 = pd.DataFrame(sku_processed_lines[index + 2: index + i])
                    new_header = sku_df_2.iloc[0]  # Grab the first row for the header
                    sku_df_2 = sku_df_2[1:]  # Take the data less the header row
                    sku_df_2.columns = new_header  # Set the header row as the df header
                    sku_df_2 = sku_df_2.apply(pd.to_numeric, errors='ignore')
                    sku_pt_2 = pd.pivot_table(data=sku_df_2, index="Item name",
                                                values=["Items viewed", "Items purchased"], aggfunc="sum")
                    sku_pt_2["CR Ratio"] = (sku_pt_2["Items purchased"] / sku_pt_2["Items viewed"]).round(2)
                    sku_filtered_pt_2 = sku_pt_2[sku_pt["Items viewed"] > 100].sort_values(by="Items viewed", ascending=False)
                    st.write(sku_df_2)
                    st.write(sku_filtered_pt_2)
                    st.subheader("Comparison of Items between time periods (Absolute)")
                    sku_difference_pt = sku_filtered_pt - sku_filtered_pt_2
                    st.write(sku_difference_pt.sort_values(by="Items viewed", ascending=False))
                    sku_sourcemed_fig = go.Figure(
                        data=[
                            go.Scatter(
                                x=sku_filtered_pt_2["Items viewed"],
                                y=sku_filtered_pt_2.index,
                                mode="markers",
                                name="before",
                                marker=dict(
                                    color="green",
                                    size=10
                                )

                            ),
                            go.Scatter(
                                x=sku_filtered_pt["Items viewed"],
                                y=sku_filtered_pt.index,
                                mode="markers",
                                name="after",
                                marker=dict(
                                    color="blue",
                                    size=10
                                )
                            ),
                        ]
                    )
                    st.write(sku_sourcemed_fig)
                    st.subheader("Comparison of items viewed between time periods (%)")
                    sku_difference_p_pt = (sku_filtered_pt - sku_filtered_pt_2)/sku_filtered_pt_2
                    st.write(sku_difference_p_pt.sort_values(by="Items viewed", ascending=False))
                    st.subheader("Comparison of source/medium between time periods")
                    sku_sourcemed_option = st.selectbox(
                        'Select items', sku_filtered_pt.index)
                    sku_campaign_pt = pd.pivot_table(sku_df[sku_df["Item name"] == sku_sourcemed_option],
                                                 index="Session source / medium",
                                                 values=["Items viewed", "Items purchased"],
                                                 aggfunc="sum"
                                                 )
                    sku_campaign_pt["CR Ratio"] = sku_campaign_pt["Items purchased"] / sku_campaign_pt["Items viewed"]
                    sku_campaign_pt_2 = pd.pivot_table(sku_df_2[sku_df_2["Item name"] == sku_sourcemed_option],
                                                       index="Session source / medium",
                                                       values=["Items viewed", "Items purchased"],
                                                       aggfunc="sum"
                                                 )
                    sku_campaign_pt_2["CR Ratio"] = sku_campaign_pt_2["Items purchased"] / sku_campaign_pt_2["Items viewed"]
                    st.write(sku_campaign_pt - sku_campaign_pt_2)
