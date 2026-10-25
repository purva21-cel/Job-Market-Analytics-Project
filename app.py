import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Job Market Analytics", page_icon="📊", layout="wide")

st.title("📊 Job Market Analytics and Salary Prediction")
st.write("Upload a CSV dataset to explore its data. Salary prediction and model comparison become available when the required salary columns are present.")

st.header("📂 Upload Your Own Dataset")
uploaded_file = st.file_uploader("Choose a CSV dataset", type=["csv"])

# Keep all analysis pages hidden until a CSV has been uploaded.
if uploaded_file is None:
    st.info("Upload a CSV file to get started. Your dataset preview, statistics, charts, and available analysis pages will appear here.")
    st.stop()

try:
    uploaded_df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Unable to read the uploaded CSV: {error}")
    st.stop()

if uploaded_df.empty:
    st.error("The uploaded dataset is empty. Please upload a CSV containing data.")
    st.stop()

df = uploaded_df.copy()
df = df.drop_duplicates().copy()

st.success("Dataset uploaded successfully!")

FEATURES = [
    "Job_Title", "Location", "Job_Type", "Experience_Level",
    "Experience_Years", "Education", "Skills", "Posting_Age_Days"
]
TARGET = "Salary_LPA"
REQUIRED_SALARY_COLUMNS = FEATURES + [TARGET]
missing_salary_columns = [c for c in REQUIRED_SALARY_COLUMNS if c not in df.columns]
salary_ready = not missing_salary_columns

NUMERIC = ["Experience_Years", "Posting_Age_Days"]
CATEGORICAL = [
    "Job_Title", "Location", "Job_Type",
    "Experience_Level", "Education", "Skills"
]

@st.cache_data
def train_models(data):
    data = data.drop_duplicates().copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET])
    if len(data) < 5:
        raise ValueError("At least 5 rows with valid salary values are recommended for a train/test split.")
    X = data[FEATURES].copy()
    y = data[TARGET].astype(float)
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    estimators = {
        "Multiple Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01, max_iter=10000),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    trained, results = {}, []
    for name, estimator in estimators.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        predictions = pipe.predict(X_test)
        trained[name] = pipe
        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
            "R2 Score": r2_score(y_test, predictions) if len(y_test) > 1 else np.nan
        })
    result_df = pd.DataFrame(results).sort_values("RMSE")
    return trained, result_df, result_df.iloc[0]["Model"]

# Pages only become available after upload. Salary-specific pages appear only
# when this uploaded CSV contains the exact columns required by the existing model.
pages = ["Dashboard"]
if salary_ready:
    pages += ["Salary Prediction", "Model Comparison"]

st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose a page", pages)

if not salary_ready:
    st.warning(
        "This CSV is available for general data exploration. Salary prediction and model comparison are hidden because the following required columns are missing: "
        + ", ".join(missing_salary_columns)
    )

if page == "Dashboard":
    st.subheader("Uploaded Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows", f"{len(df):,}")
    c2.metric("Total Columns", f"{len(df.columns):,}")
    c3.metric("Duplicate Rows Removed", f"{len(uploaded_df) - len(df):,}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Dataset Statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    st.subheader("Missing Values")
    missing_df = df.isnull().sum().rename("Missing Values").to_frame()
    missing_df["Missing %"] = (df.isnull().mean() * 100).round(2)
    st.dataframe(missing_df, use_container_width=True)

    st.subheader("Explore Charts")
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    all_columns = df.columns.tolist()

    if numeric_columns:
        chart_type = st.selectbox(
            "Choose chart type",
            ["Histogram", "Bar chart", "Line chart"],
            key="numeric_chart_type"
        )
        selected_numeric = st.selectbox(
            "Select a numeric column",
            numeric_columns,
            key="numeric_chart_column"
        )
        chart_data = df[selected_numeric].dropna()
        if chart_type == "Histogram":
            hist = pd.cut(chart_data, bins=min(20, max(1, chart_data.nunique()))).value_counts().sort_index()
            st.bar_chart(hist)
        elif chart_type == "Bar chart":
            st.bar_chart(chart_data.reset_index(drop=True))
        else:
            st.line_chart(chart_data.reset_index(drop=True))
    else:
        st.info("No numeric columns were detected, so numeric charts are unavailable for this dataset.")

    categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if categorical_columns:
        st.subheader("Category Counts")
        selected_category = st.selectbox(
            "Select a category column",
            categorical_columns,
            key="category_chart_column"
        )
        counts = df[selected_category].fillna("(Missing)").astype(str).value_counts().head(20)
        st.bar_chart(counts)

    if salary_ready:
        st.subheader("Salary Overview")
        salary_values = pd.to_numeric(df[TARGET], errors="coerce").dropna()
        if not salary_values.empty:
            a, b = st.columns(2)
            a.metric("Average Salary", f"{salary_values.mean():.2f} LPA")
            b.metric("Median Salary", f"{salary_values.median():.2f} LPA")
            st.bar_chart(salary_values.value_counts(bins=20, sort=False).sort_index())

elif page == "Salary Prediction":
    st.subheader("Estimate a Salary")
    st.caption("Predictions use the dataset you uploaded. The target column is Salary_LPA.")

    valid_df = df.copy()
    valid_df[TARGET] = pd.to_numeric(valid_df[TARGET], errors="coerce")
    valid_df = valid_df.dropna(subset=[TARGET])
    if len(valid_df) < 5:
        st.warning("At least 5 rows with valid Salary_LPA values are recommended before training a prediction model.")
        st.stop()

    with st.form("salary_form"):
        a, b = st.columns(2)
        with a:
            job_title = st.selectbox("Job Title", sorted(valid_df["Job_Title"].dropna().astype(str).unique()))
            location = st.selectbox("Location", sorted(valid_df["Location"].dropna().astype(str).unique()))
            job_type = st.selectbox("Job Type", sorted(valid_df["Job_Type"].dropna().astype(str).unique()))
            experience_level = st.selectbox("Experience Level", sorted(valid_df["Experience_Level"].dropna().astype(str).unique()))
        with b:
            experience_years = st.number_input("Experience (years)", min_value=0.0, max_value=60.0, value=2.0, step=1.0)
            education = st.selectbox("Education", sorted(valid_df["Education"].dropna().astype(str).unique()))
            skills = st.selectbox("Skills", sorted(valid_df["Skills"].dropna().astype(str).unique()))
            posting_age_days = st.number_input("Posting Age (days)", min_value=0, max_value=3650, value=10, step=1)
        submitted = st.form_submit_button("Predict Salary")

    if submitted:
        try:
            with st.spinner("Training models and estimating salary..."):
                trained, results, best_name = train_models(valid_df)
            row = pd.DataFrame([{
                "Job_Title": job_title, "Location": location, "Job_Type": job_type,
                "Experience_Level": experience_level, "Experience_Years": experience_years,
                "Education": education, "Skills": skills, "Posting_Age_Days": posting_age_days
            }])
            prediction = float(trained[best_name].predict(row)[0])
            st.success(f"Estimated Salary: ₹{max(0, prediction):.2f} LPA")
            st.caption(f"Model used: {best_name}. This is an estimate, not a guaranteed salary offer.")
        except Exception as error:
            st.error(f"Could not predict salary: {error}")

elif page == "Model Comparison":
    st.subheader("Regression Model Comparison")
    valid_df = df.copy()
    valid_df[TARGET] = pd.to_numeric(valid_df[TARGET], errors="coerce")
    valid_df = valid_df.dropna(subset=[TARGET])
    try:
        with st.spinner("Training and evaluating models on your uploaded dataset..."):
            trained, results, best_name = train_models(valid_df)
        st.dataframe(results.round(3), use_container_width=True)
        st.bar_chart(results.set_index("Model")["RMSE"])
        best = results.iloc[0]
        st.success(f"Best model by lowest RMSE: {best_name}")
        st.write(f"MAE: {best['MAE']:.3f} LPA")
        st.write(f"RMSE: {best['RMSE']:.3f} LPA")
        st.write(f"R² score: {best['R2 Score']:.3f}")
    except Exception as error:
        st.error(f"Model comparison failed: {error}")

st.divider()
st.caption("Academic project demo | Built with Python, Pandas, Scikit-learn and Streamlit")
