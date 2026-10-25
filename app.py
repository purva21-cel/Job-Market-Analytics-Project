```python
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Job Market Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Job Market Analytics and Salary Prediction")
st.write("Explore online job listings and estimate salary in LPA using machine learning.")

DATA_FILE = Path(__file__).parent / "online_job_listings_salary_dataset.csv"

FEATURES = [
    "Job_Title",
    "Location",
    "Job_Type",
    "Experience_Level",
    "Experience_Years",
    "Education",
    "Skills",
    "Posting_Age_Days"
]

TARGET = "Salary_LPA"

NUMERIC = ["Experience_Years", "Posting_Age_Days"]

CATEGORICAL = [
    "Job_Title",
    "Location",
    "Job_Type",
    "Experience_Level",
    "Education",
    "Skills"
]


@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop_duplicates().copy()
    data = data.dropna(subset=[TARGET])
    return data


@st.cache_resource
def train_models(data):
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC),
        ("cat", categorical_pipeline, CATEGORICAL)
    ])

    models = {
        "Multiple Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01, max_iter=10000),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    }

    trained = {}
    results = []

    for name, estimator in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])

        pipe.fit(X_train, y_train)
        predictions = pipe.predict(X_test)

        trained[name] = pipe

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
            "R2 Score": r2_score(y_test, predictions)
        })

    result_df = pd.DataFrame(results).sort_values("RMSE")
    best_name = result_df.iloc[0]["Model"]

    return trained, result_df, best_name, X_test, y_test


# Load the original dataset
if not DATA_FILE.exists():
    st.error(
        "The original dataset file is missing. "
        "Upload online_job_listings_salary_dataset.csv "
        "to the same GitHub repository as app.py."
    )
    st.stop()

try:
    original_df = load_data(str(DATA_FILE))
except Exception as error:
    st.error(f"Could not read the original dataset: {error}")
    st.stop()

# Dataset uploader
st.header("📂 Upload Your Own Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV dataset",
    type=["csv"]
)

df = original_df.copy()

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)

        if uploaded_df.empty:
            st.error("The uploaded dataset is empty.")
        else:
            st.success("Dataset uploaded successfully!")

            st.subheader("Uploaded Dataset Overview")

            c1, c2 = st.columns(2)
            c1.metric("Total Rows", len(uploaded_df))
            c2.metric("Total Columns", len(uploaded_df.columns))

            st.subheader("Uploaded Dataset Preview")
            st.dataframe(
                uploaded_df.head(20),
                use_container_width=True
            )

            st.subheader("Dataset Statistics")
            st.dataframe(
                uploaded_df.describe(include="all").transpose(),
                use_container_width=True
            )

            st.subheader("Missing Values")
            st.dataframe(
                uploaded_df.isnull().sum().to_frame(
                    "Missing Values"
                )
            )

            numeric_columns = uploaded_df.select_dtypes(
                include="number"
            ).columns.tolist()

            if numeric_columns:
                selected_column = st.selectbox(
                    "Select a numeric column to visualize",
                    numeric_columns
                )

                st.bar_chart(uploaded_df[selected_column])

            # Use the uploaded dataset if it supports salary analysis
            required_columns = FEATURES + [TARGET]

            missing_columns = [
                column
                for column in required_columns
                if column not in uploaded_df.columns
            ]

            if not missing_columns:
                cleaned_df = uploaded_df.drop_duplicates().copy()
                cleaned_df[TARGET] = pd.to_numeric(
                    cleaned_df[TARGET],
                    errors="coerce"
                )
                cleaned_df = cleaned_df.dropna(
                    subset=[TARGET]
                )

                if not cleaned_df.empty:
                    df = cleaned_df
                    st.success(
                        "The uploaded dataset will be used "
                        "for the dashboard and salary prediction."
                    )
                else:
                    st.warning(
                        "The uploaded dataset has no valid salary values. "
                        "The original dataset will be used."
                    )
            else:
                st.warning(
                    "This CSV is available for preview and analysis, "
                    "but it is missing columns required for salary "
                    "prediction. The original dataset will be used "
                    "for the dashboard and prediction pages. Missing: "
                    + ", ".join(missing_columns)
                )

    except Exception as error:
        st.error(f"Unable to read the uploaded CSV: {error}")


# Check the dataset used by the dashboard and prediction model
missing_columns = [
    column
    for column in FEATURES + [TARGET]
    if column not in df.columns
]

if missing_columns:
    st.error(
        "The active dataset is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

# Navigation
st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    ["Dashboard", "Salary Prediction", "Model Comparison"]
)


if page == "Dashboard":

    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Job Listings", f"{len(df):,}")
    c2.metric("Average Salary", f"{df[TARGET].mean():.2f} LPA")
    c3.metric("Median Salary", f"{df[TARGET].median():.2f} LPA")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Salary Distribution")
    st.bar_chart(
        df[TARGET].value_counts(
            bins=20,
            sort=False
        ).sort_index()
    )

    st.subheader("Average Salary by Job Title")

    salary_by_job = (
        df.groupby("Job_Title")[TARGET]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(salary_by_job)

    st.subheader("Job Listings by Location")
    st.bar_chart(df["Location"].value_counts())


elif page == "Salary Prediction":

    st.subheader("Estimate a Salary")

    st.caption(
        "Choose job details below. "
        "The model estimates salary in LPA."
    )

    with st.form("salary_form"):

        a, b = st.columns(2)

        with a:
            job_title = st.selectbox(
                "Job Title",
                sorted(df["Job_Title"].dropna().astype(str).unique())
            )

            location = st.selectbox(
                "Location",
                sorted(df["Location"].dropna().astype(str).unique())
            )

            job_type = st.selectbox(
                "Job Type",
                sorted(df["Job_Type"].dropna().astype(str).unique())
            )

            experience_level = st.selectbox(
                "Experience Level",
                sorted(
                    df["Experience_Level"]
                    .dropna()
                    .astype(str)
                    .unique()
                )
            )

        with b:
            experience_years = st.number_input(
                "Experience (years)",
                min_value=0.0,
                max_value=60.0,
                value=2.0,
                step=1.0
            )

            education = st.selectbox(
                "Education",
                sorted(df["Education"].dropna().astype(str).unique())
            )

            skills = st.selectbox(
                "Skills",
                sorted(df["Skills"].dropna().astype(str).unique())
            )

            posting_age_days = st.number_input(
                "Posting Age (days)",
                min_value=0,
                max_value=3650,
                value=10,
                step=1
            )

        submitted = st.form_submit_button("Predict Salary")

    if submitted:

        if len(df) < 2:
            st.error(
                "At least two valid dataset rows are required "
                "to train the prediction model."
            )
        else:
            try:
                with st.spinner(
                    "Training models and estimating salary..."
                ):
                    trained, results, best_name, _, _ = train_models(df)

                row = pd.DataFrame([{
                    "Job_Title": job_title,
                    "Location": location,
                    "Job_Type": job_type,
                    "Experience_Level": experience_level,
                    "Experience_Years": experience_years,
                    "Education": education,
                    "Skills": skills,
                    "Posting_Age_Days": posting_age_days
                }])

                prediction = trained[best_name].predict(row)[0]

                st.success(
                    f"Estimated Salary: ₹{max(0, prediction):.2f} LPA"
                )

                st.caption(
                    f"Model used: {best_name}. "
                    "This is an estimate, not a guaranteed salary offer."
                )

            except Exception as error:
                st.error(f"Could not predict salary: {error}")


else:

    st.subheader("Regression Model Comparison")

    if len(df) < 2:
        st.warning(
            "At least two valid dataset rows are required "
            "for model comparison."
        )
    else:
        try:
            with st.spinner(
                "Training and evaluating regression models..."
            ):
                trained, results, best_name, X_test, y_test = train_models(df)

            st.dataframe(
                results.round(3),
                use_container_width=True
            )

            st.bar_chart(results.set_index("Model")["RMSE"])

            best = results.iloc[0]

            st.success(
                f"Best model by lowest RMSE: {best_name}"
            )

            st.write(f"MAE: {best['MAE']:.3f} LPA")
            st.write(f"RMSE: {best['RMSE']:.3f} LPA")
            st.write(f"R² score: {best['R2 Score']:.3f}")

        except Exception as error:
            st.error(f"Model comparison failed: {error}")


st.divider()

st.caption(
    "Academic project demo | Built with Python, Pandas, "
    "Scikit-learn and Streamlit"
)
```
