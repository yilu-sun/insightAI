import openai
import streamlit as st
import os
import pandas as pd
import datetime

from langchain.chat_models import ChatOpenAI
from pandasai import SmartDataframe
import matplotlib.pyplot as plt

from pandasai.llm import OpenAI

# Load environment variables
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

openai.api_key = os.getenv('OPENAI_API_KEY')
llm_model = "gpt-3.5-turbo-1106"
randomness = 0


### Functions             -------------------------------------------------
class FaultTolerantSmartDataframe:
    def __init__(self, df, config):
        self.df = df
        self.sdf = SmartDataframe(df, config=config)

    def __call__(self, prompt):
        try:
            return self.sdf.chat(prompt)
        except Exception as e:
            return f'Error: {e}'


def get_sdf(df, model_name=llm_model):
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=randomness,
    )
    sdf = FaultTolerantSmartDataframe(df, config={"llm": llm})
    return sdf


def get_completion(system, prompt, model=llm_model):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=randomness,
    )
    return response.choices[0].message["content"]


def aggregate_data(sdf, time_category):
    aggregated_data_by_time = sdf(f"Aggregate the number of login_ids by ```{time_category}```")
    return aggregated_data_by_time


def get_insight_freestyle(aggregated_data_by_time, system):
    prompt_freestyle = f""" Data: ```{aggregated_data_by_time}```
    Provide the most interesting insights on usage trends.
    """
    insight_freestyle = get_completion(system, prompt_freestyle)

    return insight_freestyle


def get_insight_by_time(aggregated_data_by_time, system, time_category):
    prompt_by_time = f""" Data: ```{aggregated_data_by_time}```
    Call out the total number of logins, the ```{time_category}``` with most usage, and the ```{time_category}``` with least usage.
    Find the following patterns in the usage data: gradual increase, gradual decrease, sudden increase, sudden decrease, sudden spike, increase and decrease, periodicity.
    Be specific about percentage increase or decrease. 
    Rank the above patterns based on their significance with explanation.
    Provide the most interesting insights on usage trends based on the patterns.
    """
    insight_by_time = get_completion(system, prompt_by_time)

    return insight_by_time


def get_plot(sdf, plot_type, time_category):
    sdf(f"Plot the ```{plot_type}``` of the number of login_id by ```{time_category}```")
    st.image("temp_chart.png")


### Streamlit app code starts here     -----------------------------------------------------

st.set_page_config(
    page_title="Usage Insights",
    layout="wide"
)
st.header("Learning Insights from Raw Data")

input_data = None
user_selected_data = None

with st.sidebar:
    uploaded_files = st.file_uploader("Upload a CSV file", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        input_data = pd.read_csv(uploaded_file)
        st.write("filename:", uploaded_file.name)

    if input_data is not None:
        input_data['date'] = pd.to_datetime(input_data['date'])
        start_date = min(input_data['date']).strftime("%Y-%m-%d")
        end_date = max(input_data['date']).strftime("%Y-%m-%d")
        date_range_user = st.date_input(
            "Select the start and end time",
            (datetime.date.fromisoformat(start_date),
             datetime.date.fromisoformat(end_date)),
            datetime.date.fromisoformat(start_date),
            datetime.date.fromisoformat(end_date)
        )

        if len(date_range_user) == 2:
            if (date_range_user[0] < date_range_user[1]):
                user_selected_data = input_data[(input_data['date'] >= date_range_user[0].strftime("%Y-%m-%d")) & (
                        input_data['date'] <= date_range_user[1].strftime("%Y-%m-%d"))]

if user_selected_data is not None:
    print("true")
    sdf = get_sdf(user_selected_data, llm_model)

    system = f"""
    You are an expert data scientist providing insights on learner's usage data over a time period.
    """

    aggregated_data_by_week = aggregate_data(sdf, "week with year")
    aggregated_data_by_month = aggregate_data(sdf, "month with year")
    aggregated_data_by_quarter = aggregate_data(sdf, "quarter with year")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Top 3 Insights", "CoT Zero Shot Reasoning (Patterns Defined)", "CoT Zero Shot (Freestyle)",
         "Ask Data Questions"])

    with tab2:
        st.markdown("Chain of thought reasoning step by step on each of the defined patterns")
        get_plot(sdf, "bar plot", "week with year")
        output_week_zs = get_insight_by_time(aggregated_data_by_week, system, "week of the year")
        st.markdown(output_week_zs)

        get_plot(sdf, "bar plot", "month with year")
        output_month_zs = get_insight_by_time(aggregated_data_by_month, system, "month of the year")
        st.markdown(output_month_zs)

        get_plot(sdf, "bar plot", "quarter with year")
        output_quarter_zs = get_insight_by_time(aggregated_data_by_quarter, system, "quarter of the year")
        st.markdown(output_quarter_zs)

    with tab1:
        st.markdown("Top 3 insights after comparing weekly, monthly, quarterly results")
        col1_1, col2_1, col3_1 = st.columns(3)
        with st.container():
            with col1_1:
                get_plot(sdf, "histogram", "week with year")
            with col2_1:
                get_plot(sdf, "histogram", "month with year")
            with col3_1:
                get_plot(sdf, "histogram", "quarter with year")

        with st.container():
            prompt_comparison = f"""
            Compare the interesting insights from the weekly insights: ```{output_week_zs}```, monthly insights: ```{output_month_zs}```, and quarterly insights```{output_quarter_zs}``` and pick the most important 3 insights to display.
            """
            output_compare = get_completion(system, prompt_comparison)
            st.markdown(output_compare)

    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.header("Weekly")
            get_plot(sdf, "line plot", "week with year")
            st.markdown(get_insight_freestyle(aggregated_data_by_week, system))
        with col2:
            st.header("Monthly")
            get_plot(sdf, "line plot", "month with year")
            st.markdown(get_insight_freestyle(aggregated_data_by_month, system))
        with col3:
            st.header("Quarterly")
            get_plot(sdf, "line plot", "quarter with year")
            st.markdown(get_insight_freestyle(aggregated_data_by_quarter, system))
