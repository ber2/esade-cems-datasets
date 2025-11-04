"""
Session 5: Data in Python
CEMS Data Analytics - 2025-11-05

This marimo notebook accompanies the lecture slides.
Students will work through data cleaning and statistical inference.
"""

import marimo

__generated_with = "0.17.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    from scipy import stats
    return mo, pd, px, stats


@app.cell
def _(mo):
    mo.md("""
    # Session 5: Data in Python

    ## Learning Goals

    1. Load and explore data with pandas
    2. Identify and fix data quality issues
    3. Perform statistical tests in Python
    4. Connect statistical theory to business questions
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ber2/esade-cems-datasets/refs/heads/main/hotel_bookings.csv"
    )
    return (df,)


@app.cell
def _(df):
    df.groupby("hotel")["adr"].mean()
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 1: First Look at the Data

    Let's start by loading our hotel booking data.
    """)
    return


@app.cell
def _(df, mo):
    mo.md(f"""
    We have loaded **{len(df):,} rows** of hotel booking data.

    Let's take a first look:
    """)
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(mo):
    mo.md("""
    ### Initial Observations

    Before diving into analysis, let's check the data quality and structure.
    """)
    return


@app.cell
def _(df):
    # Basic information about the dataset
    df.info()
    return


@app.cell
def _(df):
    # Statistical summary
    df.describe()
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 2: Investigating Data Quality Issues

    Let's systematically check for common problems.
    """)
    return


@app.cell
def _(df, mo):
    # Check for duplicates
    n_duplicates = df.duplicated().sum()

    mo.md(f"""
    ### Duplicates?

    Found **{n_duplicates}** rows with identical values across all columns.

    **Important question**: Should we remove these?

    Without a unique reservation ID, identical rows might represent:
    - True duplicates (data entry errors)
    - Different reservations that happen to be identical

    **Best practice**: Only remove duplicates if you have a unique identifier or domain knowledge
    that these are truly errors.
    """)
    return


@app.cell
def _(df):
    # Show which columns have missing values
    missing = df.isna().sum()
    missing[missing > 0]
    return


@app.cell
def _(df, mo, pd):
    # Check for problematic values
    negative_prices = df[df["adr"] < 0]

    # Calculate total stays if not present
    if (
        "stays_in_weekend_nights" in df.columns
        and "stays_in_week_nights" in df.columns
    ):
        total_stays = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
        zero_nights = df[(total_stays == 0)]
    else:
        zero_nights = pd.DataFrame()

    mo.md(f"""
    ### Checking for Problematic Values

    - **{len(negative_prices)}** bookings with negative prices
    - **{len(zero_nights)}** bookings with zero-night stays
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 3: Making Data Cleaning Decisions

    Let's think through what actually needs fixing.
    """)
    return


@app.cell
def _(df, mo):
    # Step 1: Keep all data (no duplicate removal without unique ID)
    df_clean = df.copy()

    mo.md(f"""
    ### Step 1: Handle Duplicates

    **Decision**: We'll keep all rows because we lack a unique reservation ID.

    In a real scenario, you would:
    - Check with the data provider about unique identifiers
    - Investigate whether duplicates are expected
    - Only remove if you have strong evidence they're errors
    """)
    return (df_clean,)


@app.cell
def _(df_clean, mo):
    # Step 2: Investigate missing values
    missing_summary = df_clean.isna().sum()
    missing_summary = missing_summary[missing_summary > 0]

    mo.md(f"""
    ### Step 2: Handle Missing Values

    Columns with missing values:
    """)
    return (missing_summary,)


@app.cell
def _(missing_summary):
    missing_summary
    return


@app.cell
def _(mo):
    # Examine what columns have missing values and why
    mo.md("""
    **Important question**: Are these missing values a problem?

    Missing values can be:
    - **Legitimate**: Customer didn't provide information (e.g., company, agent)
    - **Problematic**: Critical fields like price or hotel type

    **Decision**: We'll keep the data as-is. Missing values in optional fields are expected.

    If we needed to handle them, we could:
    - `df.dropna(subset=['critical_column'])` - drop rows missing critical values
    - `df['column'].fillna(value)` - fill with a default value
    - Keep them as NaN if our analysis can handle it
    """)
    return


@app.cell
def _(df_clean):
    # Keep the data without dropping NaNs
    df_clean2 = df_clean.copy()
    df_clean2
    return (df_clean2,)


@app.cell
def _(df_clean2, mo):
    # Step 3: Handle impossible values and create useful features
    df_clean3 = df_clean2.copy()

    # Add total stays column if not present
    if (
        "stays_in_weekend_nights" in df_clean3.columns
        and "stays_in_week_nights" in df_clean3.columns
    ):
        df_clean3["stays_total"] = (
            df_clean3["stays_in_weekend_nights"] + df_clean3["stays_in_week_nights"]
        )

    # Check for impossible values
    n_negative = len(df_clean3[df_clean3["adr"] < 0])
    n_zero_nights = len(df_clean3[df_clean3["stays_total"] == 0])

    mo.md(f"""
    ### Step 3: Handle Impossible Values

    Found:
    - **{n_negative}** bookings with negative prices
    - **{n_zero_nights}** bookings with zero-night stays

    **Decision**: These ARE problematic - they represent data errors.

    We'll remove them:
    """)
    return (df_clean3,)


@app.cell
def _(df_clean3):
    # Filter out impossible values
    df_final = df_clean3[
        (df_clean3["adr"] >= 0) & (df_clean3["stays_total"] > 0)
    ].copy()

    df_final
    return (df_final,)


@app.cell
def _(df, df_final, mo):
    mo.md(f"""
    ### Final Dataset Summary

    - Started with: **{len(df):,} rows**
    - Final dataset: **{len(df_final):,} rows**
    - Reduction: **{100 * (1 - len(df_final) / len(df)):.1f}%**

    **What we did**:
    - ✓ Kept rows that look identical (no unique ID to verify duplicates)
    - ✓ Kept missing values in optional fields (legitimate)
    - ✓ Removed impossible values (negative prices, zero-night stays)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 4: Statistical Inference in Python

    Now let's use our clean data to answer business questions!
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Business Question 1: Hotel Type vs Price

    **Is the average price different between City Hotel and Resort Hotel?**
    """)
    return


@app.cell
def _(df_final):
    # Calculate mean prices by hotel type
    price_by_hotel = (
        df_final.groupby("hotel")["adr"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    price_by_hotel
    return


@app.cell
def _(df_final, px):
    # Visualize the price distribution by hotel
    fig1 = px.box(
        df_final,  # .sample(50000),
        x="hotel",
        y="adr",
        title="Average Daily Rate by Hotel Type",
        labels={"adr": "Average Daily Rate (€)", "hotel": "Hotel Type"},
    )
    fig1
    return


@app.cell
def _(df_final, mo, stats):
    # Perform t-test
    city_prices = df_final[df_final["hotel"] == "City Hotel"]["adr"]
    resort_prices = df_final[df_final["hotel"] == "Resort Hotel"]["adr"]

    t_stat, p_value_ttest = stats.ttest_ind(city_prices, resort_prices)

    mo.md(f"""
    ### T-Test Results

    - **t-statistic**: {t_stat:.3f}
    - **p-value**: {p_value_ttest:.4e}
    - **Significance level**: α = 0.05

    **Interpretation**:
    {
        "The prices are **significantly different** between hotel types (p < 0.05)."
        if p_value_ttest < 0.05
        else "No significant difference in prices between hotel types (p >= 0.05)."
    }
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Business Question 2: Customer Type vs Cancellation

    **Is cancellation rate related to customer type?**
    """)
    return


@app.cell
def _(df_final, pd):
    # Create a crosstab
    cancel_by_customer = (
        pd.crosstab(
            df_final["customer_type"],
            df_final["is_canceled"].apply(
                lambda x: "Cancelled" if x == 1 else "Not Cancelled"
            ),
            normalize="index",
        )
        * 100
    )  # Convert to percentages

    cancel_by_customer.round(1)
    return


@app.cell
def _(df_final, px):
    # Visualize cancellation by customer type
    fig2 = px.histogram(
        df_final,
        x="customer_type",
        color="is_canceled",
        barmode="group",
        title="Cancellation Status by Customer Type",
        labels={"customer_type": "Customer Type", "is_canceled": "Canceled"},
    )
    fig2
    return


@app.cell
def _(df_final, mo, pd, stats):
    # Perform chi-square test
    contingency_table = pd.crosstab(
        df_final["customer_type"], df_final["is_canceled"]
    )

    chi2_stat, p_value_chi2, dof, expected_freq = stats.chi2_contingency(
        contingency_table
    )

    mo.md(f"""
    ### Chi-Square Test Results

    - **Chi-square statistic**: {chi2_stat:.3f}
    - **p-value**: {p_value_chi2:.4e}
    - **Degrees of freedom**: {dof}
    - **Significance level**: α = 0.05

    **Interpretation**:
    {
        "Customer type and cancellation **are related** (p < 0.05)."
        if p_value_chi2 < 0.05
        else "No significant relationship between customer type and cancellation (p >= 0.05)."
    }
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Summary

    Today we learned:

    1. **Loading data** with pandas
    2. **Exploring data** to identify quality issues
    3. **Cleaning data** systematically (duplicates, missing values, impossible values)
    4. **Statistical inference** in Python (t-test and chi-square test)
    5. **Connecting** statistical results to business questions

    ### Pattern to Remember

    - **Explore** → Look at your data first
    - **Test** → Apply the appropriate statistical test
    - **Interpret** → What does this mean for business decisions?

    ### Next Session

    **Data Visualization in Python** - We'll learn how to create compelling visualizations
    to communicate insights from data.
    """)
    return


if __name__ == "__main__":
    app.run()
