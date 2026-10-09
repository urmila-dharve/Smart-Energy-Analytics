import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# POSTGRESQL DATABASE CONFIGURATION
# =========================================================

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
# 1. GET COMPLETE ENERGY DATA
# =========================================================

def get_energy_data():

    query = text("""
        SELECT
            datetime,
            total_consumption
        FROM energy_demand
        ORDER BY datetime;
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    return df


# =========================================================
# 2. GET AVERAGE DEMAND
# =========================================================

def get_average_demand():

    query = text("""
        SELECT AVG(total_consumption)
        FROM energy_demand;
    """)

    with engine.connect() as conn:
        result = conn.execute(query).scalar()

    return result


# =========================================================
# 3. GET PEAK DEMAND
# =========================================================

def get_peak_demand():

    query = text("""
        SELECT
            datetime,
            total_consumption
        FROM energy_demand
        ORDER BY total_consumption DESC
        LIMIT 1;
    """)

    with engine.connect() as conn:
        result = conn.execute(query).fetchone()

    return result


# =========================================================
# 4. GET MINIMUM DEMAND
# =========================================================

def get_minimum_demand():

    query = text("""
        SELECT MIN(total_consumption)
        FROM energy_demand;
    """)

    with engine.connect() as conn:
        result = conn.execute(query).scalar()

    return result


# =========================================================
# 5. TOP PEAK-DEMAND DAYS
# =========================================================

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
        df = pd.read_sql(
            query,
            conn,
            params={"limit": limit}
        )

    return df


# =========================================================
# 6. MONTHLY AVERAGE DEMAND
# =========================================================

def get_monthly_demand():

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
        df = pd.read_sql(query, conn)

    return df


# =========================================================
# 7. TOP HIGH-DEMAND MONTHS
# =========================================================

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
        df = pd.read_sql(
            query,
            conn,
            params={"limit": limit}
        )

    return df


# =========================================================
# 8. HOURLY AVERAGE DEMAND
# =========================================================

def get_hourly_demand():

    query = text("""
        SELECT
            EXTRACT(
                HOUR FROM datetime
            )::int AS hour,

            ROUND(
                AVG(total_consumption)::numeric,
                2
            ) AS average_demand

        FROM energy_demand

        GROUP BY hour

        ORDER BY hour;
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    return df


# =========================================================
# 9. WEEKDAY VS WEEKEND
# =========================================================

def get_weekday_weekend_demand():

    query = text("""
        SELECT
            CASE
                WHEN EXTRACT(
                    DOW FROM datetime
                ) IN (0, 6)
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
        df = pd.read_sql(query, conn)

    return df


# =========================================================
# 10. DAILY DEMAND
# =========================================================

def get_daily_demand():

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

        ORDER BY demand_date;
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    return df


# =========================================================
# TEST DATABASE
# =========================================================

if __name__ == "__main__":

    try:

        print("=" * 60)
        print("SMART ENERGY ANALYTICS")
        print("PostgreSQL Database Test")
        print("=" * 60)

        print("\n1. BASIC STATISTICS")

        average = get_average_demand()
        minimum = get_minimum_demand()
        peak = get_peak_demand()

        print(
            f"Average Demand : {average:.2f}"
        )

        print(
            f"Minimum Demand : {minimum:.2f}"
        )

        print(
            f"Peak DateTime  : {peak[0]}"
        )

        print(
            f"Peak Demand    : {peak[1]:.2f}"
        )

        data = get_energy_data()

        print(
            f"\nTotal Records  : {len(data)}"
        )

        print("\n2. TOP 10 PEAK-DEMAND DAYS")

        print(
            get_top_peak_days(10)
            .to_string(index=False)
        )

        print("\n3. TOP 10 HIGH-DEMAND MONTHS")

        print(
            get_top_demand_months(10)
            .to_string(index=False)
        )

        print("\n4. HOURLY DEMAND")

        print(
            get_hourly_demand()
            .to_string(index=False)
        )

        print("\n5. WEEKDAY VS WEEKEND")

        print(
            get_weekday_weekend_demand()
            .to_string(index=False)
        )

        print("\n6. MONTHLY DEMAND")

        print(
            get_monthly_demand()
            .head(12)
            .to_string(index=False)
        )

        print("\n7. DAILY DEMAND")

        print(
            get_daily_demand()
            .head(10)
            .to_string(index=False)
        )

        print("\n" + "=" * 60)
        print("DATABASE TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:

        print("\nDATABASE ERROR:")
        print(e)