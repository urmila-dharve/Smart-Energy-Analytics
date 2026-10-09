import os

import streamlit as st
import pandas as pd
import joblib

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Energy Analytics",
    page_icon="⚡",
    layout="wide"
)


# =========================================================
# POSTGRESQL CONNECTION
# =========================================================

load_dotenv()

DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:"
    f"{os.getenv('DB_PORT')}/"
    f"{os.getenv('DB_NAME')}"
)

engine = create_engine(DATABASE_URL)


# =========================================================
# LOAD ENERGY DATA FROM POSTGRESQL
# =========================================================

@st.cache_data
def load_energy_data():

    query = text("""
        SELECT
            datetime,
            total_consumption
        FROM energy_demand
        ORDER BY datetime;
    """)

    with engine.connect() as conn:
        data = pd.read_sql(query, conn)

    data["datetime"] = pd.to_datetime(data["datetime"])

    return data


# =========================================================
# SQL ANALYTICS FUNCTIONS
# =========================================================

@st.cache_data
def get_top_peak_days(limit=10):

    query = text("""
        SELECT
            DATE(datetime) AS demand_date,

            ROUND(
                AVG(total_consumption)::numeric,
                2
            ) AS average_demand,

            ROUND(
                MAX(total_consumption)::numeric,
                2
            ) AS peak_demand

        FROM energy_demand

        GROUP BY DATE(datetime)

        ORDER BY peak_demand DESC

        LIMIT :limit;
    """)

    with engine.connect() as conn:
        data = pd.read_sql(
            query,
            conn,
            params={"limit": limit}
        )

    return data


@st.cache_data
def get_top_demand_months(limit=10):

    query = text("""
        SELECT
            DATE_TRUNC(
                'month',
                datetime
            )::date AS month,

            ROUND(
                AVG(total_consumption)::numeric,
                2
            ) AS average_demand

        FROM energy_demand

        GROUP BY month

        ORDER BY average_demand DESC

        LIMIT :limit;
    """)

    with engine.connect() as conn:
        data = pd.read_sql(
            query,
            conn,
            params={"limit": limit}
        )

    return data


@st.cache_data
def get_hourly_sql_demand():

    query = text("""
        SELECT
            EXTRACT(HOUR FROM datetime)::int AS hour,

            ROUND(
                AVG(total_consumption)::numeric,
                2
            ) AS average_demand

        FROM energy_demand

        GROUP BY hour

        ORDER BY hour;
    """)

    with engine.connect() as conn:
        data = pd.read_sql(query, conn)

    return data


@st.cache_data
def get_weekday_weekend_sql():

    query = text("""
        SELECT

            CASE
                WHEN EXTRACT(DOW FROM datetime) IN (0, 6)
                THEN 'Weekend'
                ELSE 'Weekday'
            END AS day_type,

            ROUND(
                AVG(total_consumption)::numeric,
                2
            ) AS average_demand

        FROM energy_demand

        GROUP BY day_type

        ORDER BY average_demand DESC;
    """)

    with engine.connect() as conn:
        data = pd.read_sql(query, conn)

    return data


@st.cache_data
def get_monthly_sql_demand():

    query = text("""
        SELECT
            DATE_TRUNC(
                'month',
                datetime
            )::date AS month,

            ROUND(
                AVG(total_consumption)::numeric,
                2
            ) AS average_demand

        FROM energy_demand

        GROUP BY month

        ORDER BY month;
    """)

    with engine.connect() as conn:
        data = pd.read_sql(query, conn)

    return data


# =========================================================
# LOAD ML DATA
# =========================================================

@st.cache_data
def load_ml_data():

    data = pd.read_csv(
        "data/processed/ml_ready_energy.csv"
    )

    data["DateTime"] = pd.to_datetime(
        data["DateTime"]
    )

    return data


# =========================================================
# LOAD ANOMALY DATA
# =========================================================

@st.cache_data
def load_anomaly_data():

    data = pd.read_csv(
        "data/processed/anomaly_results.csv"
    )

    data["DateTime"] = pd.to_datetime(
        data["DateTime"]
    )

    return data


# =========================================================
# LOAD ML MODEL
# =========================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "models/demand_model.pkl"
    )


# =========================================================
# LOAD EVERYTHING
# =========================================================

try:

    df = load_energy_data()

    ml_df = load_ml_data()

    anomaly_df = load_anomaly_data()

    model = load_model()

except Exception as e:

    st.error(
        f"Error loading project data: {e}"
    )

    st.stop()


# =========================================================
# TITLE
# =========================================================

st.title(
    "⚡ Smart Energy Demand Analytics"
)

st.write(
    "Electricity Demand Forecasting & "
    "Anomaly Detection Platform"
)

st.success(
    "PostgreSQL database and ML model "
    "loaded successfully!"
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "📅 Select Date Range"
)

min_date = df["datetime"].min().date()

max_date = df["datetime"].max().date()


start_date = st.sidebar.date_input(
    "Start Date",
    min_value=min_date,
    max_value=max_date,
    value=min_date,
    key="start_date_v3"
)


end_date = st.sidebar.date_input(
    "End Date",
    min_value=min_date,
    max_value=max_date,
    value=max_date,
    key="end_date_v3"
)


if start_date > end_date:

    st.error(
        "Start Date must be before End Date."
    )

    st.stop()


# =========================================================
# FILTER DATA
# =========================================================

filtered_df = df[
    (df["datetime"].dt.date >= start_date)
    &
    (df["datetime"].dt.date <= end_date)
].copy()


filtered_ml = ml_df[
    (ml_df["DateTime"].dt.date >= start_date)
    &
    (ml_df["DateTime"].dt.date <= end_date)
].copy()


# Anomaly threshold
ANOMALY_THRESHOLD = 23217.14


filtered_anomalies = anomaly_df[
    (anomaly_df["DateTime"].dt.date >= start_date)
    &
    (anomaly_df["DateTime"].dt.date <= end_date)
    &
    (
        anomaly_df["Absolute_Error"]
        > ANOMALY_THRESHOLD
    )
].copy()


if len(filtered_df) == 0:

    st.warning(
        "No data available for selected dates."
    )

    st.stop()


# =========================================================
# ENERGY OVERVIEW
# =========================================================

st.subheader(
    "📊 Energy Overview"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Records",
        f"{len(filtered_df):,}"
    )


with col2:

    st.metric(
        "Average Demand",
        f"{filtered_df['total_consumption'].mean():,.0f}"
    )


with col3:

    st.metric(
        "Peak Demand",
        f"{filtered_df['total_consumption'].max():,.0f}"
    )


with col4:

    st.metric(
        "Minimum Demand",
        f"{filtered_df['total_consumption'].min():,.0f}"
    )


# =========================================================
# DEMAND TREND
# =========================================================

st.subheader(
    "📈 Electricity Demand Trend"
)


chart_data = filtered_df.copy()


if len(chart_data) > 5000:

    chart_data = (
        chart_data
        .set_index("datetime")[
            "total_consumption"
        ]
        .resample("D")
        .mean()
        .to_frame()
    )

else:

    chart_data = (
        chart_data
        .set_index("datetime")[
            ["total_consumption"]
        ]
    )


chart_data.columns = [
    "Total Consumption"
]


st.line_chart(
    chart_data
)


# =========================================================
# PEAK DEMAND ANALYSIS
# =========================================================

st.subheader(
    "⏰ Peak Demand Analysis"
)


hourly_demand = (
    filtered_df
    .assign(
        Hour=filtered_df["datetime"].dt.hour
    )
    .groupby("Hour")[
        "total_consumption"
    ]
    .mean()
)


peak_hour = hourly_demand.idxmax()

peak_hour_demand = hourly_demand.max()

lowest_hour = hourly_demand.idxmin()


peak_col1, peak_col2, peak_col3 = st.columns(3)


with peak_col1:

    st.metric(
        "Peak Demand Hour",
        f"{peak_hour:02d}:00"
    )


with peak_col2:

    st.metric(
        "Average Peak-Hour Demand",
        f"{peak_hour_demand:,.0f}"
    )


with peak_col3:

    st.metric(
        "Lowest Demand Hour",
        f"{lowest_hour:02d}:00"
    )


st.write(
    "Average electricity demand by hour:"
)


st.line_chart(
    hourly_demand.rename(
        "Average Demand"
    )
)


# =========================================================
# WEEKDAY VS WEEKEND
# =========================================================

st.subheader(
    "📅 Weekday vs Weekend Analysis"
)


filtered_df["Day_Type"] = (
    filtered_df["datetime"]
    .dt.dayofweek
    .apply(
        lambda x:
        "Weekend"
        if x >= 5
        else "Weekday"
    )
)


day_type_demand = (
    filtered_df
    .groupby("Day_Type")[
        "total_consumption"
    ]
    .mean()
)


weekday_demand = day_type_demand.get(
    "Weekday",
    0
)


weekend_demand = day_type_demand.get(
    "Weekend",
    0
)


day_col1, day_col2 = st.columns(2)


with day_col1:

    st.metric(
        "Weekday Average Demand",
        f"{weekday_demand:,.0f}"
    )


with day_col2:

    st.metric(
        "Weekend Average Demand",
        f"{weekend_demand:,.0f}"
    )


st.bar_chart(
    day_type_demand
)


# =========================================================
# POSTGRESQL SQL ANALYTICS
# =========================================================

st.header(
    "🗄️ PostgreSQL SQL Analytics"
)

st.write(
    "Advanced demand analysis performed "
    "directly using PostgreSQL queries."
)


# =========================================================
# SQL - TOP PEAK DAYS
# =========================================================

st.subheader(
    "🔥 Top 10 Peak-Demand Days"
)


top_peak_days = get_top_peak_days(10)


st.dataframe(
    top_peak_days,
    width="stretch"
)


# =========================================================
# SQL - TOP HIGH DEMAND MONTHS
# =========================================================

st.subheader(
    "📅 Top 10 High-Demand Months"
)


top_months = get_top_demand_months(10)


st.dataframe(
    top_months,
    width="stretch"
)


# =========================================================
# SQL - HOURLY DEMAND
# =========================================================

st.subheader(
    "🕐 Hourly Demand Analysis — PostgreSQL"
)


sql_hourly = get_hourly_sql_demand()


sql_hourly_chart = sql_hourly.copy()

sql_hourly_chart["hour"] = (
    sql_hourly_chart["hour"]
    .apply(
        lambda x:
        f"{x:02d}:00"
    )
)

sql_hourly_chart = (
    sql_hourly_chart
    .set_index("hour")
)


st.line_chart(
    sql_hourly_chart[
        "average_demand"
    ]
)


# =========================================================
# SQL - WEEKDAY VS WEEKEND
# =========================================================

st.subheader(
    "📊 Weekday vs Weekend — PostgreSQL"
)


sql_day_type = (
    get_weekday_weekend_sql()
)


st.dataframe(
    sql_day_type,
    width="stretch"
)


st.bar_chart(
    sql_day_type.set_index(
        "day_type"
    )["average_demand"]
)


# =========================================================
# SQL - MONTHLY DEMAND
# =========================================================

st.subheader(
    "📈 Monthly Demand Trend — PostgreSQL"
)


sql_monthly = (
    get_monthly_sql_demand()
)


sql_monthly_chart = (
    sql_monthly
    .set_index("month")
)


st.line_chart(
    sql_monthly_chart[
        "average_demand"
    ]
)


# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

st.subheader(
    "🔮 Actual vs Predicted Demand"
)


prediction_days = st.sidebar.selectbox(
    "Prediction Period",
    [1, 3, 7, 14, 30],
    index=2
)


features = [

    "Hour",

    "Day",

    "Day_of_Week",

    "Month",

    "Is_Weekend",

    "Lag_1",

    "Lag_4",

    "Lag_96",

    "Rolling_Mean_96"

]


if len(filtered_ml) > 0:

    max_rows = (
        prediction_days * 96
    )

    prediction_data = (
        filtered_ml
        .head(max_rows)
        .copy()
    )

    X = prediction_data[
        features
    ]

    prediction_data[
        "Predicted_Demand"
    ] = model.predict(X)


    prediction_chart = (
        prediction_data[
            [
                "DateTime",
                "Total_Consumption",
                "Predicted_Demand"
            ]
        ]
        .set_index("DateTime")
    )


    prediction_chart.columns = [

        "Actual Demand",

        "Predicted Demand"

    ]


    st.line_chart(
        prediction_chart
    )


    st.info(
        f"Showing Actual vs Predicted Demand "
        f"for approximately "
        f"{prediction_days} day(s)."
    )


else:

    st.warning(
        "No prediction data available."
    )


# =========================================================
# ANOMALIES
# =========================================================

st.subheader(
    "🚨 Potential Anomalies"
)


anomaly_count = len(
    filtered_anomalies
)


high_count = len(
    filtered_anomalies[
        filtered_anomalies[
            "Anomaly_Type"
        ]
        ==
        "High Demand Anomaly"
    ]
)


low_count = len(
    filtered_anomalies[
        filtered_anomalies[
            "Anomaly_Type"
        ]
        ==
        "Low Demand Anomaly"
    ]
)


anomaly_col1, anomaly_col2, anomaly_col3 = (
    st.columns(3)
)


with anomaly_col1:

    st.metric(
        "Potential Anomalies",
        f"{anomaly_count:,}"
    )


with anomaly_col2:

    st.metric(
        "High Demand",
        f"{high_count:,}"
    )


with anomaly_col3:

    st.metric(
        "Low Demand",
        f"{low_count:,}"
    )


if anomaly_count > 0:

    display_columns = [

        "DateTime",

        "Actual",

        "Expected",

        "Absolute_Error",

        "Anomaly_Type"

    ]


    available_columns = [

        col

        for col in display_columns

        if col in filtered_anomalies.columns

    ]


    st.dataframe(

        filtered_anomalies[
            available_columns
        ]
        .sort_values(
            "Absolute_Error",
            ascending=False
        )
        .head(20),

        width="stretch"

    )


    st.warning(
        "⚠️ These are potential anomalies "
        "based on unusually large deviations "
        "from model expectations. They are "
        "not confirmed faults."
    )


else:

    st.success(
        "✅ No potential anomalies found "
        "for the selected period."
    )


# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

st.subheader(
    "🤖 Machine Learning Model"
)


model_col1, model_col2, model_col3, model_col4 = (
    st.columns(4)
)


with model_col1:

    st.metric(
        "Algorithm",
        "Random Forest"
    )


with model_col2:

    st.metric(
        "Trees",
        "100"
    )


with model_col3:

    st.metric(
        "MAE",
        "4,326.97"
    )


with model_col4:

    st.metric(
        "R² Score",
        "0.9938"
    )


st.info(
    "The Random Forest model was evaluated "
    "against a persistence baseline. "
    "R² = 0.9938 should not be interpreted "
    "as 99.38% prediction accuracy."
)


# =========================================================
# RECENT DATA
# =========================================================

st.subheader(
    "📋 Recent Energy Data"
)


st.dataframe(
    filtered_df.tail(20),
    width="stretch"
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Smart Energy Demand Forecasting & "
    "Anomaly Detection Platform | "
    "Python • PostgreSQL • SQLAlchemy • "
    "Scikit-learn • Streamlit"
)