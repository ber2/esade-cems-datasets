"""
Session 7: Predicting from Past Data I
CEMS Data Analytics - 2025-11-19

This marimo notebook accompanies the lecture slides.
Students will learn linear regression with the Ames Housing dataset.
"""

import marimo

__generated_with = "0.17.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return (
        LinearRegression,
        go,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        pd,
        px,
        r2_score,
    )


@app.cell
def _(mo):
    mo.md("""
    # Session 7: Predicting from Past Data I

    ## Learning Goals

    1. Understand the concept of prediction from patterns
    2. Build simple linear regression models
    3. Evaluate model performance (MAE, RMSE, R²)
    4. Interpret regression coefficients
    5. Extend to multiple regression
    6. Recognize Simpson's paradox
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 1: Can We Predict House Prices?

    **Scenario**: You're a real estate agent. A client asks: "I have a 2,000 square foot house. What should I list it for?"

    Let's see if we can answer this using data!
    """)
    return


@app.cell
def _(pd):
    # Load Ames Housing dataset
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ber2/esade-cems-datasets/refs/heads/main/AmesHousing.csv"
    )
    return (df,)


@app.cell
def _(df, mo):
    mo.md(f"""
    Dataset loaded: **{len(df):,} house sales** in Ames, Iowa

    Key variables:
    - `Gr_Liv_Area`: Above grade living area (square feet)
    - `Sale_Price`: Sale price in dollars
    - `Bedroom_AbvGr`: Bedrooms above grade
    - `Full_Bath`: Full bathrooms
    - And many more...
    """)
    return


@app.cell
def _(df):
    # First look at the data
    df[["Gr Liv Area", "Bedroom AbvGr", "Full Bath", "SalePrice"]]
    return


@app.cell
def _(mo):
    mo.md("""
    ### Let's Visualize: Size vs Price
    """)
    return


@app.cell
def _(df, px):
    # Scatter plot: square footage vs price
    fig_scatter = px.scatter(
        df,
        x="Gr Liv Area",
        y="SalePrice",
        title="House Price vs Living Area",
        labels={
            "Gr Liv Area": "Living Area (sq ft)",
            "SalePrice": "Sale Price ($)",
        },
        opacity=0.6,
    )
    fig_scatter
    return


@app.cell
def _(mo):
    mo.md("""
    **What do we observe?**

    - Clear positive relationship: bigger houses cost more
    - The relationship looks roughly **linear**
    - But there's variability - not all 2000 sq ft houses have the same price

    **Key insight**: We can use this pattern to make predictions!
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 2: Simple Linear Regression

    Let's build our first predictive model!
    """)
    return


@app.cell
def _(LinearRegression, df):
    # Prepare data for modeling
    X_simple = df[["Gr Liv Area"]].values  # Features (must be 2D array)
    y = df["SalePrice"]  # Target

    # Create and fit the model
    model_simple = LinearRegression()
    _ = model_simple.fit(X_simple, y)
    return X_simple, model_simple, y


@app.cell
def _(mo, model_simple):
    mo.md(f"""
    ### The Model Parameters

    Our model found the "best" line through the data:

    - **Intercept**: ${model_simple.intercept_:,.2f}
    - **Coefficient (slope)**: ${model_simple.coef_[0]:,.2f} per sq ft

    **Interpretation**:
    - Base price (when size = 0): ${model_simple.intercept_:,.2f}
    - Each additional square foot adds: ${model_simple.coef_[0]:,.2f} to the price
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Making Predictions

    Now we can predict the price for **any** house size!
    """)
    return


@app.cell
def _(mo, model_simple):
    # Predict for a 2000 sq ft house
    predicted_price_2000 = model_simple.predict([[2000]])[0]

    mo.md(f"""
    **Example**: A 2,000 sq ft house

    Predicted price: **${predicted_price_2000:,.2f}**

    Calculation:
    ```
    Price = Intercept + (Coefficient × Square Feet)
          = ${model_simple.intercept_:,.2f} + (${model_simple.coef_[0]:,.2f} × 2000)
          = ${predicted_price_2000:,.2f}
    ```
    """)
    return


@app.cell
def _(df, go, model_simple, np, px):
    # Visualize the regression line
    fig_regression = px.scatter(
        df,
        x="Gr Liv Area",
        y="SalePrice",
        title="Linear Regression: House Price vs Living Area",
        labels={
            "Gr Liv Area": "Living Area (sq ft)",
            "SalePrice": "Sale Price ($)",
        },
        opacity=0.5,
    )

    # Add regression line
    x_range = np.linspace(df["Gr Liv Area"].min(), df["Gr Liv Area"].max(), 100)
    y_pred_line = model_simple.predict(x_range.reshape(-1, 1))

    fig_regression.add_trace(
        go.Scatter(
            x=x_range,
            y=y_pred_line,
            mode="lines",
            name="Regression Line",
            line=dict(color="red", width=3),
        )
    )

    fig_regression
    return


@app.cell
def _(mo):
    mo.md("""
    The red line represents our model's predictions.
    Points close to the line = good predictions.
    Points far from the line = poor predictions.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 3: Evaluating Our Predictions

    How good is our model? Let's measure!
    """)
    return


@app.cell
def _(X_simple, model_simple):
    # Make predictions for all houses in our dataset
    y_pred_simple = model_simple.predict(X_simple)
    return (y_pred_simple,)


@app.cell
def _(go, px, y, y_pred_simple):
    # Predicted vs Actual plot
    fig_pred_vs_actual = px.scatter(
        x=y,
        y=y_pred_simple,
        title="Predicted vs Actual Prices",
        labels={"x": "Actual Price ($)", "y": "Predicted Price ($)"},
        opacity=0.5,
    )

    # Add diagonal line (perfect predictions)
    fig_pred_vs_actual.add_trace(
        go.Scatter(
            x=[y.min(), y.max()],
            y=[y.min(), y.max()],
            mode="lines",
            name="Perfect Predictions",
            line=dict(color="red", dash="dash"),
        )
    )

    fig_pred_vs_actual
    return


@app.cell
def _(mo):
    mo.md("""
    **If all predictions were perfect**, all points would fall on the red diagonal line.

    Points above the line = we over-predicted
    Points below the line = we under-predicted
    """)
    return


@app.cell
def _(y, y_pred_simple):
    # Calculate residuals (errors)
    residuals_simple = y - y_pred_simple
    return (residuals_simple,)


@app.cell
def _(px, residuals_simple, y_pred_simple):
    # Residual plot
    fig_residuals = px.scatter(
        x=y_pred_simple,
        y=residuals_simple,
        title="Residual Plot",
        labels={
            "x": "Predicted Price ($)",
            "y": "Residual (Actual - Predicted) ($)",
        },
        opacity=0.5,
    )

    # Add horizontal line at y=0
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")

    fig_residuals
    return


@app.cell
def _(mo):
    mo.md("""
    **Good residual plot**: Random scatter around zero (no patterns)

    **Bad residual plot**: Clear patterns (curved, funnel-shaped, etc.) indicate model problems
    """)
    return


@app.cell
def _(
    mean_absolute_error,
    mean_squared_error,
    mo,
    np,
    r2_score,
    y,
    y_pred_simple,
):
    # Calculate evaluation metrics
    mae_simple = mean_absolute_error(y, y_pred_simple)
    rmse_simple = np.sqrt(mean_squared_error(y, y_pred_simple))
    r2_simple = r2_score(y, y_pred_simple)

    mo.md(f"""
    ### Evaluation Metrics

    **Mean Absolute Error (MAE)**: ${mae_simple:,.2f}
    - On average, our predictions are off by ${mae_simple:,.2f}
    - Easy to interpret: "typical error"

    **Root Mean Squared Error (RMSE)**: ${rmse_simple:,.2f}
    - Penalizes large errors more heavily
    - Also in dollars

    **R-squared (R²)**: {r2_simple:.3f}
    - We explain {r2_simple * 100:.1f}% of the variation in house prices
    - Using just square footage alone!
    """)
    return mae_simple, r2_simple, rmse_simple


@app.cell
def _(mo):
    mo.md("""
    ### Would You Trust This Model?

    **Questions to consider**:
    - Is a $30,000 average error acceptable?
    - For what use cases would this be good enough?
    - When would we need better predictions?

    **Context matters!**
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### ⚠️ Simpson's Paradox

    **Warning**: A trend can **reverse** when data is grouped!

    Let's see an example with our housing data.
    """)
    return


@app.cell
def _(df):
    # Create a simplified example: look at two neighborhoods
    # Filter to just two neighborhoods for clarity
    neighborhoods = df["Neighborhood"].value_counts().head(2).index
    df_two_neighborhoods = df[df["Neighborhood"].isin(neighborhoods)].copy()
    return (df_two_neighborhoods,)


@app.cell
def _(df, px):
    # Overall trend
    fig_overall = px.scatter(
        df,
        x="Year Built",
        y="SalePrice",
        title="Overall Trend: Year Built vs Price",
        labels={"Year Built": "Year Built", "SalePrice": "Sale Price ($)"},
        opacity=0.5,
        trendline="ols",
    )
    fig_overall
    return


@app.cell
def _(df_two_neighborhoods, px):
    # By neighborhood
    fig_by_neighborhood = px.scatter(
        df_two_neighborhoods,
        x="Year Built",
        y="SalePrice",
        color="Neighborhood",
        title="By Neighborhood: Year Built vs Price",
        labels={"Year Built": "Year Built", "SalePrice": "Sale Price ($)"},
        opacity=0.6,
        trendline="ols",
    )
    fig_by_neighborhood
    return


@app.cell
def _(mo):
    mo.md("""
    **Simpson's Paradox Lesson**:

    The overall trend might be positive, but within specific neighborhoods,
    the trend could be different or even reversed!

    **Hidden variables** (like neighborhood) can completely change the story.

    **Always**:
    - Check for lurking variables
    - Visualize subgroups
    - Don't trust overall patterns blindly
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 4: Multiple Regression

    Can we improve predictions by using more than just square footage?
    """)
    return


@app.cell
def _(mo, r2_simple):
    mo.md(f"""
    **Current performance**: R² = {r2_simple:.3f}

    We explain {r2_simple * 100:.1f}% of price variation using only living area.

    **What other features might matter?**
    - Number of bedrooms
    - Number of bathrooms
    - Age of house
    - Garage size
    - Neighborhood
    - And more...

    Let's try adding bedrooms and bathrooms!
    """)
    return


@app.cell
def _(LinearRegression, df):
    # Multiple regression with 3 features
    X_multiple = df[["Gr Liv Area", "Bedroom AbvGr", "Full Bath"]].values
    y_multi = df["SalePrice"]

    model_multiple = LinearRegression()
    _ = model_multiple.fit(X_multiple, y_multi)
    return X_multiple, model_multiple, y_multi


@app.cell
def _(mo, model_multiple):
    mo.md(f"""
    ### Multiple Regression Coefficients

    - **Intercept**: ${model_multiple.intercept_:,.2f}
    - **Living Area**: ${model_multiple.coef_[0]:,.2f} per sq ft
    - **Bedrooms**: ${model_multiple.coef_[1]:,.2f} per bedroom
    - **Bathrooms**: ${model_multiple.coef_[2]:,.2f} per bathroom

    **Interpretation** (holding other variables constant):
    - Each additional square foot adds ${model_multiple.coef_[0]:,.2f}
    - Each additional bedroom adds ${model_multiple.coef_[1]:,.2f}
    - Each additional bathroom adds ${model_multiple.coef_[2]:,.2f}
    """)
    return


@app.cell
def _(mo, model_multiple):
    # Example prediction with multiple features
    # 2000 sq ft, 3 bedrooms, 2 bathrooms
    example_house = [[2000, 3, 2]]
    predicted_price_multi = model_multiple.predict(example_house)[0]

    mo.md(f"""
    ### Example Prediction

    **House**: 2,000 sq ft, 3 bedrooms, 2 bathrooms

    **Predicted Price**: ${predicted_price_multi:,.2f}

    **Calculation**:
    ```
    Price = {model_multiple.intercept_:,.2f}
          + ({model_multiple.coef_[0]:,.2f} × 2000)
          + ({model_multiple.coef_[1]:,.2f} × 3)
          + ({model_multiple.coef_[2]:,.2f} × 2)\n
          = ${predicted_price_multi:,.2f}
    ```
    """)
    return


@app.cell
def _(X_multiple, model_multiple):
    # Predictions with multiple regression
    y_pred_multiple = model_multiple.predict(X_multiple)
    return (y_pred_multiple,)


@app.cell
def _(
    mean_absolute_error,
    mean_squared_error,
    mo,
    np,
    r2_score,
    y_multi,
    y_pred_multiple,
):
    # Evaluate multiple regression
    mae_multiple = mean_absolute_error(y_multi, y_pred_multiple)
    rmse_multiple = np.sqrt(mean_squared_error(y_multi, y_pred_multiple))
    r2_multiple = r2_score(y_multi, y_pred_multiple)

    mo.md(f"""
    ### Multiple Regression Performance

    **Mean Absolute Error (MAE)**: ${mae_multiple:,.2f}\n
    **Root Mean Squared Error (RMSE)**: ${rmse_multiple:,.2f}\n
    **R-squared (R²)**: {r2_multiple:.3f}

    We now explain **{r2_multiple * 100:.1f}%** of price variation.
    """)
    return mae_multiple, r2_multiple, rmse_multiple


@app.cell
def _(
    mae_multiple,
    mae_simple,
    mo,
    r2_multiple,
    r2_simple,
    rmse_multiple,
    rmse_simple,
):
    mo.md(f"""
    ### Comparison: Simple vs Multiple Regression

    | Metric | Simple (area only) | Multiple (area + bed + bath) | Improvement |
    |--------|-------------------|------------------------------|-------------|
    | MAE | ${mae_simple:,.2f} | ${mae_multiple:,.2f} | ${mae_simple - mae_multiple:,.2f} |
    | RMSE | ${rmse_simple:,.2f} | ${rmse_multiple:,.2f} | ${rmse_simple - rmse_multiple:,.2f} |
    | R² | {r2_simple:.3f} | {r2_multiple:.3f} | +{(r2_multiple - r2_simple) * 100:.1f}% |

    **Adding bedrooms and bathrooms improved our predictions!**
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Complications to Consider

    **1. Units Matter**
    - $110/sq ft vs $5,000/bedroom: different scales
    - Hard to compare "importance" directly
    - Solution: standardization (advanced topic)

    **2. Correlated Features**
    - Bigger houses tend to have more bedrooms
    - Living area and bedrooms are **correlated**
    - Makes it hard to separate their individual effects
    - This is called **multicollinearity**

    **3. More Features ≠ Always Better**
    - Adding features always increases R² on training data
    - But might **overfit** to our specific dataset
    - Could perform worse on new houses
    - **More on this next session!**
    """)
    return


@app.cell
def _(df):
    # Correlation between features
    df[["Gr Liv Area", "Bedroom AbvGr", "Full Bath"]].corr().round(3)
    return


@app.cell
def _(mo):
    mo.md("""
    Notice the correlation between living area and bedrooms (0.5+).
    They're moderately correlated - bigger houses have more bedrooms.
    """)
    return


@app.cell
def _(df, px):
    px.imshow(
        df[["Gr Liv Area", "Bedroom AbvGr", "Full Bath"]].corr().round(3),
        text_auto=True,
        template="plotly",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Summary

    ### What We Learned Today

    **1. Simple Linear Regression**
    - Find the best line through data
    - Make predictions from the line
    - Interpret slope and intercept

    **2. Evaluation Metrics**
    - **MAE**: Average absolute error (easy to interpret)
    - **RMSE**: Penalizes large errors more
    - **R²**: Fraction of variation explained (0 to 1)

    **3. Simpson's Paradox**
    - Overall trends can reverse in subgroups
    - Always check for hidden variables
    - Visualize different groupings

    **4. Multiple Regression**
    - Use multiple features for better predictions
    - Each feature has its own coefficient
    - Interpret as "holding others constant"
    - Watch for correlated features
    - More features isn't always better (overfitting!)

    ### Next Session

    **Predicting from Past Data II**
    - Handling categorical variables (neighborhoods, etc.)
    - Understanding and preventing overfitting
    - Train/test splits
    - Telling your data story
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
