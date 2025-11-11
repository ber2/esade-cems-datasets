"""
Session 6: Data Visualization in Python
CEMS Data Analytics - 2025-11-12

This marimo notebook accompanies the lecture slides.
Students will explore data visualization techniques with the hotel dataset.
"""

import marimo

__generated_with = "0.17.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import numpy as np
    return mo, pd, px


@app.cell
def _(mo):
    mo.md("""
    # Session 6: Data Visualization in Python

    ## Learning Goals

    1. Understand why visualization reveals what statistics hide
    2. Explore distributions with histograms, box plots, and KDE
    3. Find relationships with scatter plots
    4. Compare categories with bar charts and line charts
    5. Choose the right visualization for your question
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 1: The Power of Looking at Data

    ### Anscombe's Quartet

    Four datasets with **identical** statistical properties:
    - Same mean for X and Y
    - Same variance
    - Same correlation coefficient
    - Same linear regression line

    But are they really the same?
    """)
    return


@app.cell
def _(pd):
    # Anscombe's Quartet data
    anscombe = pd.DataFrame(
        {
            "dataset": ["I"] * 11 + ["II"] * 11 + ["III"] * 11 + ["IV"] * 11,
            "x": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5] * 3 + [8] * 11,
            "y": [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
            + [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
            + [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
            + [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89],
        }
    )
    return (anscombe,)


@app.cell
def _(anscombe):
    anscombe
    return


@app.cell
def _(anscombe):
    # Statistical summary by dataset
    anscombe.groupby("dataset").agg(
        {"x": ["mean", "std"], "y": ["mean", "std"]}
    ).round(2)
    return


@app.cell
def _(anscombe, px):
    # Now let's visualize them
    fig_anscombe = px.scatter(
        anscombe,
        x="x",
        y="y",
        facet_col="dataset",
        title="Anscombe's Quartet: Same Statistics, Different Patterns",
        trendline="ols",
        height=400,
    )
    fig_anscombe
    return


@app.cell
def _(mo):
    mo.md("""
    **Key Insight**: Statistics can hide what visualization reveals!

    - Dataset I: Linear relationship
    - Dataset II: Non-linear (quadratic) relationship
    - Dataset III: Linear with one outlier
    - Dataset IV: No relationship except one influential point

    This is why we **always visualize our data** before analyzing it.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 2: Exploring Hotel Prices Visually

    Let's load our hotel booking dataset and explore price distributions.
    """)
    return


@app.cell
def _(pd):
    # Load hotel data
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ber2/esade-cems-datasets/refs/heads/main/hotel_bookings.csv"
    )

    # Clean the data (same as Session 5)
    if (
        "stays_in_weekend_nights" in df.columns
        and "stays_in_week_nights" in df.columns
    ):
        df["stays_total"] = (
            df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
        )

    df_clean = df[(df["adr"] >= 0) & (df["stays_total"] > 0)].copy()
    return (df_clean,)


@app.cell
def _(df_clean, mo):
    mo.md(f"""
    Dataset loaded: **{len(df_clean):,} bookings**

    Let's explore how prices (ADR - Average Daily Rate) are distributed.
    """)
    return


@app.cell
def _(df_clean, px):
    # Histogram of prices
    fig_hist = px.histogram(
        df_clean,  # [df_clean.adr < 5000],
        x="adr",
        nbins=50,
        title="Distribution of Hotel Prices",
        labels={"adr": "Average Daily Rate (€)"},
        marginal="box",
    )
    fig_hist
    return


@app.cell
def _(mo):
    mo.md("""
    ### What do we observe?

    - **Right-skewed**: Most prices are moderate, with a long tail of high prices
    - **Mode**: Most common prices around €100-€150
    - **Outliers**: Some very expensive bookings (€400+)

    This pattern is common in price data - most transactions at typical prices,
    few luxury/premium transactions.
    """)
    return


@app.cell
def _(df_clean, px):
    # Box plot of prices
    fig_box = px.box(
        df_clean,  # [df_clean.adr < 5000],
        y="adr",
        title="Box Plot of Hotel Prices",
        labels={"adr": "Average Daily Rate (€)"},
    )
    fig_box
    return


@app.cell
def _(mo):
    mo.md("""
    ### Anatomy of a Box Plot

    - **Box**: Contains middle 50% of data (Interquartile Range)
    - **Line inside box**: Median (50th percentile)
    - **Whiskers**: Extend to typical range (1.5 × IQR)
    - **Dots**: Potential outliers beyond whiskers

    Box plots are excellent for spotting outliers and comparing distributions.
    """)
    return


@app.cell
def _(df_clean, px):
    # Comparing distributions: City vs Resort
    fig_compare = px.box(
        df_clean[df_clean.adr < 5000].sample(5000),
        x="hotel",
        y="adr",
        color="hotel",
        title="Price Comparison: City Hotel vs Resort Hotel",
        labels={"adr": "Average Daily Rate (€)", "hotel": "Hotel Type"},
    )
    fig_compare
    return


@app.cell
def _(df_clean):
    # Summary statistics by hotel type
    df_clean.groupby("hotel")["adr"].agg(["count", "mean", "median", "std"]).round(
        2
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Business Insights

    **Question**: What do these distributions tell us about pricing strategies?

    - Different median prices between hotel types
    - Different variability (spread) in pricing
    - Different outlier patterns

    This could indicate:
    - Different target customer segments
    - Different room types/amenities
    - Different seasonal pricing strategies
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 3: Finding Relationships

    Scatter plots help us discover relationships between variables.
    """)
    return


@app.cell
def _(df_clean, px):
    # Scatter plot: stays vs price
    fig_scatter1 = px.scatter(
        df_clean.sample(min(10000, len(df_clean))),  # Sample for performance
        x="stays_total",
        y="adr",
        title="Relationship: Length of Stay vs Price",
        labels={"stays_total": "Total Nights", "adr": "Average Daily Rate (€)"},
        opacity=0.5,
    )
    fig_scatter1
    return


@app.cell
def _(mo):
    mo.md("""
    **What pattern do you see?**

    The relationship isn't immediately clear. Let's add more information...
    """)
    return


@app.cell
def _(df_clean, px):
    # Add color encoding
    fig_scatter2 = px.scatter(
        df_clean.sample(1000),
        x="stays_total",
        y="adr",
        color="hotel",
        title="Length of Stay vs Price by Hotel Type",
        labels={
            "stays_total": "Total Nights",
            "adr": "Average Daily Rate (€)",
            "hotel": "Hotel Type",
        },
        opacity=0.6,
    )
    fig_scatter2
    return


@app.cell
def _(mo):
    mo.md("""
    ### Visual Encoding

    We just used **color** to encode a third variable (hotel type).

    Common visual encoding dimensions:
    - **Position** (x, y coordinates)
    - **Color** (categorical or continuous)
    - **Size** (magnitude)
    - **Shape** (categorical)

    Each adds information but also adds complexity - use wisely!
    """)
    return


@app.cell
def _(df_clean, px):
    # Add size encoding too
    fig_scatter3 = px.scatter(
        df_clean.sample(1000),
        x="stays_total",
        y="adr",
        color="hotel",
        size="adults",
        title="Stay Length vs Price (colored by hotel, sized by number of adults)",
        labels={
            "stays_total": "Total Nights",
            "adr": "Average Daily Rate (€)",
            "hotel": "Hotel Type",
            "adults": "Number of Adults",
        },
        opacity=0.5,
    )
    fig_scatter3
    return


@app.cell
def _(mo):
    mo.md("""
    **Now we're showing 4 variables at once!**

    But is it too much? There's a balance between information and clarity.
    """)
    return


@app.cell
def _(df_clean, px):
    # Exploring another relationship: lead time vs cancellation
    fig_scatter4 = px.scatter(
        df_clean.sample(1000),
        x="lead_time",
        y="adr",
        color="is_canceled",
        title="Lead Time vs Price (colored by cancellation status)",
        labels={
            "lead_time": "Days Between Booking and Arrival",
            "adr": "Average Daily Rate (€)",
            "is_canceled": "Canceled",
        },
        opacity=0.4,
    )
    fig_scatter4
    return


@app.cell
def _(mo):
    mo.md("""
    ### Exploration Activity

    Try creating scatter plots for different variable pairs:
    - Price vs number of previous bookings
    - Lead time vs number of special requests
    - What other relationships might be interesting?

    **Remember**: Not every scatter plot will show a clear pattern - and that's okay!
    Sometimes "no relationship" is an important finding.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 4: Comparing Across Categories

    Different visualization types work better for different questions.
    """)
    return


@app.cell
def _(df_clean, pd):
    # Average price by arrival month
    monthly_avg = df_clean.groupby("arrival_date_month")["adr"].mean().reset_index()

    # Order months correctly
    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    monthly_avg["arrival_date_month"] = pd.Categorical(
        monthly_avg["arrival_date_month"], categories=month_order, ordered=True
    )
    monthly_avg = monthly_avg.sort_values("arrival_date_month")
    return month_order, monthly_avg


@app.cell
def _(monthly_avg):
    monthly_avg
    return


@app.cell
def _(monthly_avg, px):
    # Line chart for time series
    fig_line1 = px.line(
        monthly_avg,
        x="arrival_date_month",
        y="adr",
        title="Average Hotel Price by Month",
        labels={"arrival_date_month": "Month", "adr": "Average Daily Rate (€)"},
        markers=True,
        # range_y=[0, monthly_avg.adr.max() * 1.1 ],
    )
    fig_line1
    return


@app.cell
def _(mo):
    mo.md("""
    ### Seasonal Patterns

    Line charts are perfect for showing trends over time.

    **What do we observe?**
    - Peak prices in certain months (summer/holidays?)
    - Lower prices in off-season
    - Gradual transitions between seasons
    """)
    return


@app.cell
def _(df_clean, month_order, pd, px):
    # Comparing both hotels over time
    monthly_by_hotel = (
        df_clean.groupby(["arrival_date_month", "hotel"])["adr"]
        .mean()
        .reset_index()
    )
    monthly_by_hotel["arrival_date_month"] = pd.Categorical(
        monthly_by_hotel["arrival_date_month"], categories=month_order, ordered=True
    )
    monthly_by_hotel = monthly_by_hotel.sort_values("arrival_date_month")

    fig_line2 = px.line(
        monthly_by_hotel,
        x="arrival_date_month",
        y="adr",
        color="hotel",
        title="Average Price by Month and Hotel Type",
        labels={
            "arrival_date_month": "Month",
            "adr": "Average Daily Rate (€)",
            "hotel": "Hotel Type",
        },
        markers=True,
        # range_y=[0, monthly_by_hotel.adr.max() * 1.1 ],
    )
    fig_line2
    return


@app.cell
def _(mo):
    mo.md("""
    **Now we can compare seasonal patterns between hotel types**

    Do both hotels have the same seasonal pricing strategy?
    """)
    return


@app.cell
def _(df_clean, px):
    # Bar chart for categorical comparisons
    customer_avg = df_clean.groupby("customer_type")["adr"].mean().reset_index()

    fig_bar1 = px.bar(
        customer_avg,
        x="customer_type",
        y="adr",
        title="Average Price by Customer Type",
        labels={"customer_type": "Customer Type", "adr": "Average Daily Rate (€)"},
        color="customer_type",
    )
    fig_bar1
    return


@app.cell
def _(mo):
    mo.md("""
    ### When to Use Bar Charts vs Line Charts?

    - **Bar charts**: Comparing distinct categories (no inherent order)
    - **Line charts**: Showing trends over continuous variable (time, age, etc.)

    Using the wrong type can mislead!
    """)
    return


@app.cell
def _(df_clean, px):
    # Grouped bar chart - can get crowded
    hotel_customer_avg = (
        df_clean.groupby(["hotel", "customer_type"])["adr"].mean().reset_index()
    )

    fig_bar2 = px.bar(
        hotel_customer_avg,
        x="customer_type",
        y="adr",
        color="hotel",
        barmode="group",
        title="Average Price by Customer Type and Hotel",
        labels={
            "customer_type": "Customer Type",
            "adr": "Average Daily Rate (€)",
            "hotel": "Hotel Type",
        },
    )
    fig_bar2
    return


@app.cell
def _(mo):
    mo.md("""
    This is getting crowded. What if we add more categories?
    """)
    return


@app.cell
def _(df_clean, px):
    # Faceting / Small multiples
    hotel_customer_avg_facet = (
        df_clean.groupby(["hotel", "customer_type"])["adr"].mean().reset_index()
    )

    fig_facet = px.bar(
        hotel_customer_avg_facet,
        x="customer_type",
        y="adr",
        color="customer_type",
        facet_col="hotel",
        title="Average Price by Customer Type (separated by hotel)",
        labels={
            "customer_type": "Customer Type",
            "adr": "Average Daily Rate (€)",
            "hotel": "Hotel Type",
        },
    )
    fig_facet
    return


@app.cell
def _(mo):
    mo.md("""
    ### Faceting / Small Multiples

    When a single plot gets too crowded, **split into multiple panels**.

    **Trade-off**:
    - ✅ Easier to read each panel
    - ✅ Clearer patterns within categories
    - ⚠️ Harder to make direct comparisons across panels
    - ⚠️ Takes more space

    Choose based on your question!
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Summary: Choosing the Right Visualization

    | Question | Plot Type |
    |----------|-----------|
    | How is a variable distributed? | Histogram, box plot, violin |
    | What's the relationship between two numeric variables? | Scatter plot |
    | How do categories compare? | Bar chart |
    | How does something change over time? | Line chart |
    | Too many categories/groups? | Consider faceting |

    ### The Visualization Workflow

    1. **Start with a question**
    2. **Choose an appropriate plot type**
    3. **Create the visualization**
    4. **Observe patterns**
    5. **Adjust or try different approaches**
    6. **Interpret for your business context**

    **Remember**: Visualization is iterative. You often need to try multiple
    approaches before finding the clearest way to communicate your insight.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Practice Exercises

    Try exploring these questions with visualizations:

    1. **Distribution**: How are lead times distributed? Do they differ by hotel type?

    2. **Relationship**: Is there a relationship between number of previous cancellations
       and the likelihood of canceling this booking?

    3. **Comparison**: How do average prices compare across different market segments?

    4. **Trend**: How do cancellation rates vary by month?

    5. **Complex**: Create a visualization showing price by month, separated by hotel type,
       with different colors for customer types. Is it too complex? How could you simplify it?

    ### Next Session

    **Predicting from Past Data I**

    We'll use visualization skills to:
    - Explore data before building models
    - Visualize model predictions
    - Evaluate model performance
    - Communicate results effectively
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
