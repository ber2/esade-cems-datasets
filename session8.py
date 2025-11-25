"""
Session 8: Predicting from Past Data II + Storytelling
CEMS Data Analytics - 2025-11-26

This marimo notebook accompanies the lecture slides.
Students will learn about categorical variables, overfitting, and communication.
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
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures
    return (
        LinearRegression,
        PolynomialFeatures,
        go,
        mo,
        np,
        pd,
        px,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md("""
    # Session 8: Predicting from Past Data II + Storytelling

    ## Learning Goals

    1. Handle categorical variables with one-hot encoding
    2. Understand overfitting and underfitting
    3. Use train/test splits to evaluate models properly
    4. Communicate data insights effectively
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 1: When Simple Variables Aren't Enough

    Last session we built models using numeric features:
    - Square footage
    - Number of bedrooms
    - Number of bathrooms

    But what about **neighborhood**?
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
def _(df):
    # Look at neighborhood variable
    df[["Neighborhood", "SalePrice"]].head(10)
    return


@app.cell
def _(df):
    # How many neighborhoods?
    df["Neighborhood"].value_counts()
    return


@app.cell
def _(mo):
    mo.md("""
    ### The Problem

    Neighborhood clearly affects price, but it's **categorical** (text), not numeric.

    **Question**: Can we include it in our linear regression model?
    """)
    return


@app.cell
def _(df, px):
    # Visualize price by neighborhood
    neighborhood_avg = (
        df.groupby("Neighborhood")["SalePrice"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig_neighborhood = px.bar(
        neighborhood_avg,
        x="Neighborhood",
        y="SalePrice",
        title="Average House Price by Neighborhood",
        labels={
            "Sale_Price": "Average Sale Price ($)",
            "Neighborhood": "Neighborhood",
        },
    )
    fig_neighborhood
    return


@app.cell
def _(df, px):
    px.scatter(df, x="Gr Liv Area", y="SalePrice", color="Neighborhood", opacity=0.3)
    return


@app.cell
def _(mo):
    mo.md("""
    **Clear pattern**: Neighborhoods have very different average prices!

    From ~$100k to ~$300k depending on location.

    But how do we include this in regression?
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Wrong Approaches

    ❌ **Assign numbers**: Downtown=1, Midtown=2, Suburbs=3
    - Problem: Implies order and arithmetic (2 is "between" 1 and 3)

    ❌ **Use zip codes**: 02138, 02139, etc.
    - Problem: Arithmetic doesn't make sense (02138 + 02139 ≠ meaningful)

    ✅ **One-Hot Encoding**: Create binary columns for each category
    """)
    return


@app.cell
def _(df):
    # Demonstrate one-hot encoding with a small example
    sample_df = df[["Neighborhood", "SalePrice"]].head(5).copy()

    # Before encoding
    print("Before one-hot encoding:")
    return (sample_df,)


@app.cell
def _(sample_df):
    sample_df
    return


@app.cell
def _(pd, sample_df):
    # After encoding
    sample_encoded = pd.get_dummies(
        sample_df, columns=["Neighborhood"], prefix="Neighborhood"
    )
    sample_encoded
    return


@app.cell
def _(mo):
    mo.md("""
    ### One-Hot Encoding Explained

    **For each category, create a binary (0/1) column**:
    - `Neighborhood_CollgCr` = 1 if College Creek, 0 otherwise
    - `Neighborhood_Veenker` = 1 if Veenker, 0 otherwise
    - etc.

    **Now we have numbers that regression can use!**

    **Important**: One category is typically dropped to avoid multicollinearity (dummy variable trap).
    The dropped category becomes the "reference" or "baseline".
    """)
    return


@app.cell
def _(df, pd):
    # Apply one-hot encoding to the full dataset
    df_encoded = pd.get_dummies(df, columns=["Neighborhood"], drop_first=True)

    # Show the new columns
    neighborhood_cols = [
        col for col in df_encoded.columns if col.startswith("Neighborhood_")
    ]
    return df_encoded, neighborhood_cols


@app.cell
def _(mo, neighborhood_cols):
    mo.md(f"""
    Created **{len(neighborhood_cols)} neighborhood dummy variables**

    (One was dropped as reference category)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Building Model With Neighborhoods
    """)
    return


@app.cell
def _(LinearRegression, df_encoded, neighborhood_cols):
    # Model without neighborhoods (baseline for comparison)
    X_no_neighborhood = df_encoded[["Gr Liv Area", "Bedroom AbvGr", "Full Bath"]]
    y = df_encoded["SalePrice"]

    model_no_neighborhood = LinearRegression()
    model_no_neighborhood.fit(X_no_neighborhood, y)
    r2_no_neighborhood = model_no_neighborhood.score(X_no_neighborhood, y)

    # Model WITH neighborhoods
    X_with_neighborhood = df_encoded[
        ["Gr Liv Area", "Bedroom AbvGr", "Full Bath"] + neighborhood_cols
    ]

    model_with_neighborhood = LinearRegression()
    model_with_neighborhood.fit(X_with_neighborhood, y)
    r2_with_neighborhood = model_with_neighborhood.score(X_with_neighborhood, y)
    return (
        X_with_neighborhood,
        model_with_neighborhood,
        r2_no_neighborhood,
        r2_with_neighborhood,
        y,
    )


@app.cell
def _(mo, r2_no_neighborhood, r2_with_neighborhood):
    mo.md(f"""
    ### Performance Comparison

    | Model | R² |
    |-------|-----|
    | Without neighborhood | {r2_no_neighborhood:.3f} |
    | **With neighborhood** | **{r2_with_neighborhood:.3f}** |

    **Improvement**: +{(r2_with_neighborhood - r2_no_neighborhood) * 100:.1f} percentage points

    Location matters!
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Interpreting Neighborhood Coefficients

    Each neighborhood coefficient represents the **premium or discount** relative to
    the reference neighborhood (the one we dropped), holding other variables constant.

    **Example interpretation**:
    - `Neighborhood_StoneBr` coefficient = +$40,000
    - Meaning: "A house in Stone Brook costs $40,000 more than a house in the reference
      neighborhood, all else being equal (same size, bedrooms, bathrooms)"
    """)
    return


@app.cell
def _(model_with_neighborhood):
    model_with_neighborhood.feature_names_in_
    return


@app.cell
def _(model_with_neighborhood):
    model_with_neighborhood.coef_
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 2: The Model That Knew Too Much

    Let's explore what happens when we add **too many** features.
    """)
    return


@app.cell
def _(df_encoded, np):
    # Select many features (everything numeric)
    feature_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols.remove("SalePrice")  # Don't include target!

    # Keep only features without missing values for simplicity
    feature_cols_clean = [
        col for col in feature_cols if df_encoded[col].notna().all()
    ]
    return (feature_cols_clean,)


@app.cell
def _(feature_cols_clean, mo):
    mo.md(f"""
    We have **{len(feature_cols_clean)} numeric features** available.

    What if we use ALL of them?
    """)
    return


@app.cell
def _(LinearRegression, df_encoded, feature_cols_clean):
    # Build model with ALL features
    X_all_features = df_encoded[feature_cols_clean].dropna()
    y_all_features = df_encoded.loc[X_all_features.index, "SalePrice"]

    model_all_features = LinearRegression()
    model_all_features.fit(X_all_features, y_all_features)
    r2_all_features = model_all_features.score(X_all_features, y_all_features)
    return X_all_features, r2_all_features, y_all_features


@app.cell
def _(mo, r2_all_features):
    mo.md(f"""
    ### Result: R² = {r2_all_features:.3f}

    Amazing! We explain {r2_all_features * 100:.1f}% of price variation!

    **But wait... is this too good to be true?**
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### The Problem: Overfitting

    When a model is too complex, it can **memorize** the training data instead of
    learning general patterns.

    **Analogy**: Like a student who memorizes answers to practice problems but doesn't
    understand the concepts. Perfect on practice test, fails on real exam.

    **How do we detect overfitting?**
    → **Train/Test Split**
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Train/Test Split

    **Idea**: Hold out some data to simulate the "future"

    1. **Training set** (90%): Use to build model
    2. **Test set** (10%): Use ONLY to evaluate final model

    **Rule**: Never look at test set during model development!
    """)
    return


@app.cell
def _(X_all_features, train_test_split, y_all_features):
    # Split the data
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
        X_all_features, y_all_features, test_size=0.05, random_state=42
    )
    return X_test_all, X_train_all, y_test_all, y_train_all


@app.cell
def _(LinearRegression, X_test_all, X_train_all, y_test_all, y_train_all):
    # Retrain model on training set only
    model_all_train = LinearRegression()
    model_all_train.fit(X_train_all, y_train_all)

    # Evaluate on both train and test
    r2_train_all = model_all_train.score(X_train_all, y_train_all)
    r2_test_all = model_all_train.score(X_test_all, y_test_all)
    return r2_test_all, r2_train_all


@app.cell
def _(mo, r2_test_all, r2_train_all):
    mo.md(f"""
    ### Train vs Test Performance

    | Dataset | R² |
    |---------|-----|
    | Training | {r2_train_all:.3f} |
    | **Test** | **{r2_test_all:.3f}** |

    **Gap**: {(r2_train_all - r2_test_all) * 100:.1f} percentage points

    The model performs worse on unseen data - a sign of overfitting!
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Comparing Different Model Complexities

    Let's compare simple vs complex models on train and test sets.
    """)
    return


@app.cell
def _(LinearRegression, X_with_neighborhood, train_test_split, y):
    # Simple model: just area, bed, bath, neighborhood
    X_simple_split = X_with_neighborhood
    y_simple_split = y

    X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
        X_simple_split, y_simple_split, test_size=0.05, random_state=42
    )

    # Train simple model
    model_simple_split = LinearRegression()
    model_simple_split.fit(X_train_simple, y_train_simple)

    r2_train_simple = model_simple_split.score(X_train_simple, y_train_simple)
    r2_test_simple = model_simple_split.score(X_test_simple, y_test_simple)
    return r2_test_simple, r2_train_simple


@app.cell
def _(mo, r2_test_all, r2_test_simple, r2_train_all, r2_train_simple):
    mo.md(f"""
    ### Model Comparison

    | Model | # Features | Train R² | Test R² | Gap |
    |-------|-----------|----------|---------|-----|
    | Simple | ~30 | {r2_train_simple:.3f} | {r2_test_simple:.3f} | {(r2_train_simple - r2_test_simple):.3f} |
    | Complex | ~80 | {r2_train_all:.3f} | {r2_test_all:.3f} | {(r2_train_all - r2_test_all):.3f} |

    **Key insight**: The simpler model generalizes better! Lower gap = less overfitting.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Visualizing Overfitting: Polynomial Regression

    Let's see overfitting with a simpler example: polynomial regression.

    We'll fit polynomials of different degrees to the relationship between
    living area and price.
    """)
    return


@app.cell
def _(df):
    # Use a subset for clearer visualization
    df_subset = df.sample(200, random_state=42)
    return (df_subset,)


@app.cell
def _(
    LinearRegression,
    PolynomialFeatures,
    df_subset,
    go,
    np,
    px,
    train_test_split,
):
    # Prepare data
    X_poly = df_subset[["Gr Liv Area"]].values
    y_poly = df_subset["SalePrice"].values

    X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
        X_poly, y_poly, test_size=0.1, random_state=42
    )

    # Fit models with different polynomial degrees
    degrees = [1, 3, 7]
    models_poly = {}
    predictions_poly = {}

    for degree in degrees:
        # Transform features
        poly = PolynomialFeatures(degree=degree)
        X_train_poly_transformed = poly.fit_transform(X_train_poly)
        X_test_poly_transformed = poly.transform(X_test_poly)

        # Fit model
        model = LinearRegression()
        model.fit(X_train_poly_transformed, y_train_poly)

        # Store
        models_poly[degree] = (model, poly)

        # For visualization, create predictions across range
        x_range_poly = np.linspace(X_poly.min(), X_poly.max(), 300).reshape(-1, 1)
        x_range_transformed = poly.transform(x_range_poly)
        y_pred_range = model.predict(x_range_transformed)

        predictions_poly[degree] = (x_range_poly, y_pred_range)

        # Calculate scores
        train_score = model.score(X_train_poly_transformed, y_train_poly)
        test_score = model.score(X_test_poly_transformed, y_test_poly)

    # Visualize
    fig_poly = px.scatter(
        x=X_train_poly.flatten(),
        y=y_train_poly,
        title="Polynomial Regression: Underfitting vs Good Fit vs Overfitting",
        labels={"x": "Living Area (sq ft)", "y": "Sale Price ($)"},
        opacity=0.5,
    )

    colors = ["red", "green", "blue"]
    names = ["Degree 1 (Underfit)", "Degree 3 (Good Fit)", "Degree 7 (Overfit)"]

    for degree, color, name in zip(degrees, colors, names):
        x_range_poly, y_pred_range = predictions_poly[degree]
        fig_poly.add_trace(
            go.Scatter(
                x=x_range_poly.flatten(),
                y=y_pred_range,
                mode="lines",
                name=name,
                line=dict(color=color, width=3),
            )
        )

    fig_poly
    return X_test_poly, X_train_poly, degrees, y_test_poly, y_train_poly


@app.cell
def _(
    LinearRegression,
    PolynomialFeatures,
    X_test_poly,
    X_train_poly,
    degrees,
    mo,
    y_test_poly,
    y_train_poly,
):
    # Calculate and display scores for each degree
    results_poly = []
    for deg in degrees:
        poly_temp = PolynomialFeatures(degree=deg)
        X_train_transformed_temp = poly_temp.fit_transform(X_train_poly)
        X_test_transformed_temp = poly_temp.transform(X_test_poly)

        model_temp = LinearRegression()
        model_temp.fit(X_train_transformed_temp, y_train_poly)

        train_r2 = model_temp.score(X_train_transformed_temp, y_train_poly)
        test_r2 = model_temp.score(X_test_transformed_temp, y_test_poly)

        results_poly.append(
            {
                "Degree": deg,
                "Train R²": f"{train_r2:.3f}",
                "Test R²": f"{test_r2:.3f}",
                "Gap": f"{(train_r2 - test_r2):.3f}",
            }
        )

    mo.md("""
    ### Polynomial Regression Results
    """)
    return (results_poly,)


@app.cell
def _(pd, results_poly):
    pd.DataFrame(results_poly)
    return


@app.cell
def _(mo):
    mo.md("""
    **Observations**:

    - **Degree 1 (Underfit)**: Too simple, misses the pattern
    - **Degree 3 (Good Fit)**: Captures main pattern, generalizes well
    - **Degree 15 (Overfit)**: Wiggly line through every point, terrible on test data!

    **The lesson**: More complexity ≠ better model. Find the right balance.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Understanding the Bias-Variance Tradeoff

    The phenomenon we just observed has a theoretical explanation.

    **Every prediction error comes from two sources**:
    1. **Bias**: Error from oversimplification (wrong assumptions)
    2. **Variance**: Error from being too sensitive to specific training data

    Let's explore this with our polynomial examples.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Bias: Systematic Error from Oversimplification

    **High bias** occurs when our model is too simple to capture the true pattern.

    **Example**: Degree 1 polynomial (straight line)
    - Makes strong assumption: relationship is perfectly linear
    - Consistently misses the curved pattern
    - Error is systematic and predictable

    This is **underfitting** - the model hasn't learned enough from the data.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Variance: Sensitivity to Training Data

    **High variance** occurs when our model is so flexible it fits noise in the training data.

    **Example**: Degree 15 polynomial (wiggly line)
    - Makes almost no assumptions: can fit any curve
    - Changes dramatically if we use different training data
    - Fits training perfectly but fails on new data

    This is **overfitting** - the model has "memorized" rather than learned.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### The Tradeoff

    **We cannot minimize both bias and variance simultaneously.**

    - Reducing bias (more complex model) → increases variance
    - Reducing variance (simpler model) → increases bias

    **Goal**: Find the model complexity that minimizes **total error** = bias + variance

    This is why the middle ground (degree 3 in our example) performs best!
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Visualizing the Tradeoff

    Let's see how error changes with model complexity.
    """)
    return


@app.cell
def _(
    LinearRegression,
    PolynomialFeatures,
    X_test_poly,
    X_train_poly,
    pd,
    px,
    y_test_poly,
    y_train_poly,
):
    # Calculate train and test error for many polynomial degrees
    degrees_range = range(1, 16)
    errors_by_degree = []

    for deg in degrees_range:
        poly_bv = PolynomialFeatures(degree=deg)
        X_train_poly_bv = poly_bv.fit_transform(X_train_poly)
        X_test_poly_bv = poly_bv.transform(X_test_poly)

        model_bv = LinearRegression()
        model_bv.fit(X_train_poly_bv, y_train_poly)

        train_r2_bv = model_bv.score(X_train_poly_bv, y_train_poly)
        test_r2_bv = model_bv.score(X_test_poly_bv, y_test_poly)

        errors_by_degree.append({
            'Degree': deg,
            'Train Error': 1 - train_r2_bv,  # Convert R² to error
            'Test Error': 1 - test_r2_bv,
            'Gap': (1 - test_r2_bv) - (1 - train_r2_bv)
        })

    df_errors = pd.DataFrame(errors_by_degree)

    # Plot training vs test error
    fig_bv = px.line(
        df_errors.melt(id_vars='Degree', value_vars=['Train Error', 'Test Error']),
        x='Degree',
        y='value',
        color='variable',
        title='Bias-Variance Tradeoff: Error vs Model Complexity',
        labels={'value': 'Error (1 - R²)', 'variable': 'Dataset'},
        markers=True
    )

    # Add annotation for sweet spot
    sweet_spot_degree = df_errors.loc[df_errors['Test Error'].idxmin(), 'Degree']
    fig_bv.add_annotation(
        x=sweet_spot_degree,
        y=df_errors.loc[df_errors['Degree'] == sweet_spot_degree, 'Test Error'].values[0],
        text=f"Sweet Spot<br>(Degree {int(sweet_spot_degree)})",
        showarrow=True,
        arrowhead=2
    )

    fig_bv
    return


@app.cell
def _(mo):
    mo.md("""
    ### Interpreting the Plot

    **Training error** (blue line):
    - Decreases as we add complexity
    - Model fits training data better and better
    - At degree 15: almost perfect fit (low error)

    **Test error** (red line):
    - Decreases initially (reducing bias)
    - Reaches minimum at "sweet spot"
    - Then increases (variance takes over)

    **The gap between lines** grows with overfitting.

    **Key insight**: The best model for **new data** is NOT the one that fits
    training data perfectly. It's the one that balances bias and variance.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Practical Implications

    1. **Start simple**: Begin with simple models, add complexity only if needed
    2. **Use test sets**: Always evaluate on held-out data
    3. **Watch the gap**: Large train-test gap = overfitting
    4. **Regularization**: Techniques exist to control variance (beyond this course)
    5. **More data helps**: With more training data, variance decreases

    This framework applies to **all** machine learning models, not just polynomials!
    """)
    return


if __name__ == "__main__":
    app.run()
