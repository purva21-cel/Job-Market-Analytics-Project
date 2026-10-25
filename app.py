import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Job Market Analytics",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# DARK THEME
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: #e6edf3;
}

[data-testid="stHeader"] {
    background-color: #0e1117;
}

[data-testid="stSidebar"] {
    background-color: #202c30;
}

[data-testid="stSidebar"] * {
    color: #e6edf3;
}

h1, h2, h3, h4 {
    color: #f4f7fb !important;
}

p, label {
    color: #c9d1d9;
}

[data-testid="stMetric"] {
    background-color: #191e27;
    padding: 18px;
    border: 1px solid #303641;
    border-radius: 10px;
}

[data-testid="stMetricLabel"] {
    color: #9da8b5 !important;
}

[data-testid="stMetricValue"] {
    color: #f4f7fb !important;
}

.stButton > button {
    background-color: #d45d43;
    color: white;
    border: none;
    border-radius: 7px;
}

.stButton > button:hover {
    background-color: #ed7659;
    color: white;
}

[data-testid="stDataFrame"] {
    border: 1px solid #303641;
    border-radius: 8px;
}

hr {
    border-color: #343b46;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "dataset" not in st.session_state:
    st.session_state.dataset = None

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}

if "model_results" not in st.session_state:
    st.session_state.model_results = None

if "training_target" not in st.session_state:
    st.session_state.training_target = None

if "training_features" not in st.session_state:
    st.session_state.training_features = []

if "training_data" not in st.session_state:
    st.session_state.training_data = None

if "training_numeric_features" not in st.session_state:
    st.session_state.training_numeric_features = []

if "training_categorical_features" not in st.session_state:
    st.session_state.training_categorical_features = []


# --------------------------------------------------
# SIDEBAR AND CSV UPLOAD
# --------------------------------------------------

st.sidebar.markdown("# 💼 JobLens")
st.sidebar.caption("Insights for any CSV file")

uploaded_file = st.sidebar.file_uploader(
    "Upload any CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        uploaded_df = pd.read_csv(uploaded_file)

        uploaded_df.columns = (
            uploaded_df.columns.astype(str).str.strip()
        )

        uploaded_df = uploaded_df.loc[
            :, ~uploaded_df.columns.duplicated()
        ]

        uploaded_df = uploaded_df.drop_duplicates().reset_index(drop=True)

        if uploaded_df.empty:
            st.sidebar.error("The dataset is empty.")

        else:
            current_signature = (
                uploaded_file.name,
                uploaded_file.size
            )

            if st.session_state.get("dataset_signature") != current_signature:

                st.session_state.dataset = uploaded_df
                st.session_state.dataset_signature = current_signature

                st.session_state.trained_models = {}
                st.session_state.model_results = None
                st.session_state.training_target = None
                st.session_state.training_features = []
                st.session_state.training_data = None
                st.session_state.training_numeric_features = []
                st.session_state.training_categorical_features = []

    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")


# --------------------------------------------------
# WELCOME PAGE
# --------------------------------------------------

if st.session_state.dataset is None:

    st.title("💼 What Drives the Job Market?")

    st.markdown("""
    Upload a CSV file to explore data, discover trends, compare
    salaries, analyze important skills, and train machine-learning models.

    The dashboard adapts to the columns in your uploaded dataset.
    """)

    st.info("Upload a CSV file from the sidebar to get started.")

    st.markdown("### Dashboard Features")

    c1, c2, c3 = st.columns(3)

    c1.metric("Analytics", "Interactive")
    c2.metric("Input Format", "CSV")
    c3.metric("ML Models", "4")

    st.markdown("""
    - Dataset profiling
    - Job-market overview
    - Salary analysis
    - Skills and company analysis
    - Location and posting trends
    - Machine-learning model comparison
    - Numeric target prediction
    """)

    st.stop()


df = st.session_state.dataset.copy()


# --------------------------------------------------
# COLUMN DETECTION
# --------------------------------------------------

def find_column(names):

    normalized_names = [
        re.sub(r"[^a-z0-9]", "", name.lower())
        for name in names
    ]

    for column in df.columns:

        normalized_column = re.sub(
            r"[^a-z0-9]", "", column.lower()
        )

        if normalized_column in normalized_names:
            return column

    return None


title_col = find_column([
    "job_title", "title", "job", "position",
    "role", "designation", "job_role"
])

company_col = find_column([
    "company", "company_name", "employer",
    "organization", "organisation"
])

location_col = find_column([
    "location", "job_location", "city",
    "work_location", "job_city"
])

salary_col = find_column([
    "salary", "average_salary", "salary_avg",
    "annual_salary", "salary_yearly",
    "estimated_salary", "salary_in_usd",
    "salary_usd"
])

skills_col = find_column([
    "skills", "required_skills", "skill",
    "technical_skills", "skill_set"
])

date_col = find_column([
    "date_posted", "posting_date", "posted_date",
    "job_posted_date", "created_at",
    "publication_date", "post_date"
])

work_col = find_column([
    "work_type", "employment_type",
    "job_type", "work_mode", "remote_type"
])

experience_col = find_column([
    "experience_level", "experience",
    "seniority", "years_experience"
])


# --------------------------------------------------
# SALARY CLEANING
# --------------------------------------------------

def clean_salary(series):

    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace("USD", "", regex=False)
        .str.replace("INR", "", regex=False)
        .str.replace(r"[^\d.\-]", "", regex=True)
    )

    return pd.to_numeric(cleaned, errors="coerce")


if salary_col is not None:
    df[salary_col] = clean_salary(df[salary_col])

if date_col is not None:
    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )


# --------------------------------------------------
# SIDEBAR DATA INFORMATION
# --------------------------------------------------

st.sidebar.success("Dataset loaded")

st.sidebar.write("Records:", len(df))
st.sidebar.write("Columns:", len(df.columns))

sample = pd.DataFrame({
    "Job Title": ["Data Analyst", "Data Scientist"],
    "Company": ["ABC Technologies", "XYZ Solutions"],
    "Location": ["Mumbai", "Pune"],
    "Salary": [800000, 1200000],
    "Skills": ["Python, SQL, Excel", "Python, ML, SQL"],
    "Date Posted": ["2026-09-15", "2026-09-18"]
})

st.sidebar.download_button(
    "Download Sample Template",
    data=sample.to_csv(index=False).encode("utf-8"),
    file_name="job_market_template.csv",
    mime="text/csv"
)


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Data Profile",
        "Overview",
        "Model Performance",
        "Predictor",
        "Statistics",
        "Explore Data"
    ]
)


# --------------------------------------------------
# FILTERS
# --------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")

filtered_df = df.copy()


def apply_filter(data, column, label):

    if column is None:
        return data

    values = sorted(
        data[column].dropna().astype(str).unique().tolist()
    )

    if not values:
        return data

    selected = st.sidebar.multiselect(
        label,
        values,
        key="filter_" + column
    )

    if selected:
        data = data[
            data[column].astype(str).isin(selected)
        ]

    return data


filtered_df = apply_filter(filtered_df, title_col, "Job Title")
filtered_df = apply_filter(filtered_df, company_col, "Company")
filtered_df = apply_filter(filtered_df, location_col, "Location")
filtered_df = apply_filter(filtered_df, work_col, "Work Type")


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def category_counts(column, data=None):

    if column is None:
        return pd.Series(dtype="int64")

    if data is None:
        data = filtered_df

    values = (
        data[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = values[values != ""]

    return values.value_counts()


def get_skills(data):

    if skills_col is None:
        return pd.Series(dtype="int64")

    skills = []

    for value in data[skills_col].dropna():

        for item in re.split(r"[,;|]", str(value)):

            item = item.strip()

            if item:
                skills.append(item)

    if not skills:
        return pd.Series(dtype="int64")

    return pd.Series(skills).value_counts()


def show_chart(series, title):

    st.subheader(title)

    if series is None or len(series) == 0:
        st.info("No data available for this chart.")
        return

    st.bar_chart(series)


# --------------------------------------------------
# DATA PROFILE
# --------------------------------------------------

if page == "Data Profile":

    st.title("Data Profile")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.markdown("### Dataset Preview")

    st.dataframe(df.head(15), use_container_width=True)

    st.markdown("### Column Profile")

    profile = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(profile, use_container_width=True)

    st.markdown("### Missing Values by Column")

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing):
        st.bar_chart(missing)
    else:
        st.success("No missing values found.")

    st.markdown("### Statistical Summary")

    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

elif page == "Overview":

    st.title("Job Market Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Job Postings", len(filtered_df))

    c2.metric(
        "Companies",
        filtered_df[company_col].nunique()
        if company_col else 0
    )

    c3.metric(
        "Locations",
        filtered_df[location_col].nunique()
        if location_col else 0
    )

    c4.metric(
        "Job Titles",
        filtered_df[title_col].nunique()
        if title_col else 0
    )

    if salary_col is not None:

        salaries = filtered_df[salary_col].dropna()

        if not salaries.empty:

            st.markdown("### Salary Overview")

            c1, c2, c3 = st.columns(3)

            c1.metric("Average Salary", f"{salaries.mean():,.2f}")
            c2.metric("Median Salary", f"{salaries.median():,.2f}")
            c3.metric("Maximum Salary", f"{salaries.max():,.2f}")

    col1, col2 = st.columns(2)

    with col1:
        show_chart(
            category_counts(title_col).head(10),
            "Top Job Titles"
        )

    with col2:
        show_chart(
            category_counts(company_col).head(10),
            "Top Hiring Companies"
        )

    col1, col2 = st.columns(2)

    with col1:
        show_chart(
            category_counts(location_col).head(10),
            "Top Job Locations"
        )

    with col2:
        show_chart(
            get_skills(filtered_df).head(15),
            "Most In-Demand Skills"
        )

    if work_col:
        show_chart(
            category_counts(work_col),
            "Employment Type Distribution"
        )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.title("🤖 Model Performance Comparison")

    st.write(
        "Train four regression models on your uploaded dataset. "
        "The application compares their performance and identifies "
        "the model with the lowest test-set RMSE."
    )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_columns:

        st.warning(
            "No numeric columns were found. "
            "A numeric target column is required for regression."
        )

        st.stop()

    default_target = (
        salary_col
        if salary_col in numeric_columns
        else numeric_columns[-1]
    )

    target = st.selectbox(
        "Select the column you want to predict",
        numeric_columns,
        index=numeric_columns.index(default_target)
    )

    test_size = st.slider(
        "Test set percentage",
        min_value=10,
        max_value=40,
        value=20
    )

    if st.button("🚀 Train and Compare Models"):

        data = df.copy()

        data[target] = pd.to_numeric(
            data[target],
            errors="coerce"
        )

        data = data.replace(
            [np.inf, -np.inf],
            np.nan
        )

        data = data.dropna(subset=[target])

        if len(data) < 10:

            st.error(
                "At least 10 valid rows are recommended "
                "for model comparison."
            )

            st.stop()

        X = data.drop(columns=[target])
        y = data[target]

        # Remove identifier columns
        id_columns = [
            col for col in X.columns
            if col.lower().replace("_", "").replace(" ", "")
            in [
                "id",
                "jobid",
                "companyid",
                "employeeid",
                "recordid"
            ]
        ]

        X = X.drop(
            columns=id_columns,
            errors="ignore"
        )

        # Remove columns with only one unique value
        X = X.loc[
            :,
            X.nunique(dropna=True) > 1
        ]

        # Keep supported feature types
        supported_columns = X.select_dtypes(
            include=["number", "object", "category", "bool", "string"]
        ).columns.tolist()

        X = X[supported_columns]

        if X.empty:

            st.error(
                "No usable input features were found. "
                "Upload a dataset with more informative columns."
            )

            st.stop()

        if y.nunique() < 2:

            st.error(
                "The selected target has fewer than two distinct "
                "values. Choose another numeric column."
            )

            st.stop()

        numeric_features = X.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical_features = X.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers = []

        if numeric_features:

            transformers.append((
                "numeric",
                numeric_pipeline,
                numeric_features
            ))

        if categorical_features:

            transformers.append((
                "categorical",
                categorical_pipeline,
                categorical_features
            ))

        preprocessor = ColumnTransformer(
            transformers=transformers
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size / 100,
            random_state=42
        )

        models = {
            "Ridge Regression": Ridge(alpha=1.0),

            "Linear Regression": LinearRegression(),

            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                random_state=42
            ),

            "Decision Tree": DecisionTreeRegressor(
                random_state=42
            )
        }

        results = []
        trained_models = {}

        progress = st.progress(0)
        status = st.empty()

        for i, (name, model) in enumerate(models.items()):

            status.write(f"Training {name}...")

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            try:

                pipeline.fit(X_train, y_train)

                predictions = pipeline.predict(X_test)

                mae = mean_absolute_error(
                    y_test,
                    predictions
                )

                rmse = np.sqrt(
                    mean_squared_error(
                        y_test,
                        predictions
                    )
                )

                r2 = r2_score(
                    y_test,
                    predictions
                )

                results.append({
                    "Model": name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R²": r2
                })

                trained_models[name] = pipeline

            except Exception as e:

                st.warning(
                    f"{name} could not be trained: {e}"
                )

            progress.progress(
                (i + 1) / len(models)
            )

        status.empty()

        if not results:

            st.error(
                "No models could be trained. "
                "Check your dataset and selected target."
            )

            st.stop()

        results_df = pd.DataFrame(results)

        st.session_state.model_results = results_df
        st.session_state.trained_models = trained_models
        st.session_state.training_target = target
        st.session_state.training_features = X.columns.tolist()
        st.session_state.training_data = X
        st.session_state.training_numeric_features = numeric_features
        st.session_state.training_categorical_features = categorical_features
        st.session_state.training_test_size = test_size

        st.success("Model training and comparison completed!")

    # Display saved results
    if st.session_state.model_results is not None:

        results_df = st.session_state.model_results.copy()

        # Ensure the selected target matches the trained target
        if target != st.session_state.training_target:

            st.info(
                "You selected a different target. "
                "Click 'Train and Compare Models' to train models "
                "for this target."
            )

        else:

            valid_results = results_df.dropna(
                subset=["RMSE", "MAE", "R²"]
            )

            if valid_results.empty:

                st.error("No valid model results are available.")

            else:

                # Lowest RMSE is the primary selection metric
                best_model = valid_results.loc[
                    valid_results["RMSE"].idxmin()
                ]

                st.markdown("---")
                st.markdown("## 🏆 Best Performing Model")

                st.success(
                    f"🏆 Best Model: {best_model['Model']}"
                )

                st.caption(
                    "Selected using the lowest RMSE on the test dataset. "
                    "A lower RMSE means smaller prediction errors "
                    "on this test set."
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Best Model",
                    best_model["Model"]
                )

                col2.metric(
                    "Lowest RMSE",
                    f"{best_model['RMSE']:,.2f}"
                )

                col3.metric(
                    "R² Score",
                    f"{best_model['R²']:.3f}"
                )

                st.markdown("### 📊 Model Leaderboard")

                leaderboard = valid_results.sort_values(
                    by="RMSE",
                    ascending=True
                ).reset_index(drop=True)

                leaderboard.index = leaderboard.index + 1
                leaderboard.index.name = "Rank"

                st.dataframe(
                    leaderboard.style.format({
                        "MAE": "{:,.2f}",
                        "RMSE": "{:,.2f}",
                        "R²": "{:.3f}"
                    }),
                    use_container_width=True
                )

                st.markdown("### 📉 RMSE Comparison")

                rmse_chart = valid_results.set_index(
                    "Model"
                )[["RMSE"]].sort_values(
                    "RMSE",
                    ascending=True
                )

                st.bar_chart(rmse_chart)

                st.caption(
                    "The shortest bar represents the model "
                    "with the lowest RMSE."
                )

                st.markdown("### 📈 R² Score Comparison")

                r2_chart = valid_results.set_index(
                    "Model"
                )[["R²"]].sort_values(
                    "R²",
                    ascending=False
                )

                st.bar_chart(r2_chart)

                st.caption(
                    "Higher R² is generally better. A negative "
                    "R² means the model performed worse than "
                    "predicting the test-set mean."
                )

                st.markdown("### 📝 Final Result")

                st.write(
                    f"Among the successfully evaluated models, "
                    f"**{best_model['Model']}** achieved the "
                    f"lowest RMSE of **{best_model['RMSE']:,.2f}**."
                )

                st.write(
                    f"Its MAE is **{best_model['MAE']:,.2f}** "
                    f"and its R² score is **{best_model['R²']:.3f}**."
                )

                if best_model["R²"] < 0:

                    st.warning(
                        "Although this model has the lowest RMSE "
                        "among the tested models, its negative R² "
                        "indicates that it performed worse than the "
                        "test-set mean baseline. Improve the dataset "
                        "or features before relying on predictions."
                    )

                elif best_model["R²"] < 0.5:

                    st.info(
                        "The best model has limited explanatory "
                        "performance. Consider improving the "
                        "features, data quality, or model settings."
                    )

                else:

                    st.success(
                        "The best model explains a meaningful "
                        "proportion of the target variation "
                        "on this test set."
                    )


# --------------------------------------------------
# PREDICTOR
# --------------------------------------------------

elif page == "Predictor":

    st.title("🔮 Numeric Target Predictor")

    if not st.session_state.trained_models:

        st.info(
            "Open Model Performance, select a numeric target, "
            "and train the models before making predictions."
        )

        st.stop()

    models = st.session_state.trained_models

    selected_model = st.selectbox(
        "Select a trained model",
        list(models.keys())
    )

    feature_columns = st.session_state.training_features
    training_data = st.session_state.training_data

    numeric_features = st.session_state.training_numeric_features
    categorical_features = st.session_state.training_categorical_features

    st.markdown(
        f"### Enter values to predict "
        f"{st.session_state.training_target}"
    )

    input_values = {}

    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_columns):

        container = col1 if i % 2 == 0 else col2

        with container:

            if feature in numeric_features:

                values = pd.to_numeric(
                    training_data[feature],
                    errors="coerce"
                ).dropna()

                if values.empty:

                    input_values[feature] = 0.0

                else:

                    minimum = float(values.min())
                    maximum = float(values.max())
                    median = float(values.median())

                    if minimum == maximum:

                        input_values[feature] = minimum

                    else:

                        input_values[feature] = st.number_input(
                            feature,
                            min_value=minimum,
                            max_value=maximum,
                            value=median,
                            key="predict_" + feature
                        )

            else:

                values = (
                    training_data[feature]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if values:

                    input_values[feature] = st.selectbox(
                        feature,
                        values,
                        key="predict_" + feature
                    )

                else:

                    input_values[feature] = ""

    if st.button("🔮 Predict"):

        input_df = pd.DataFrame(
            [input_values],
            columns=feature_columns
        )

        try:

            prediction = models[selected_model].predict(
                input_df
            )[0]

            st.success("Prediction completed!")

            st.metric(
                f"Predicted {st.session_state.training_target}",
                f"{prediction:,.2f}"
            )

            st.write("Model used:", selected_model)

        except Exception as e:

            st.error(f"Prediction failed: {e}")


# --------------------------------------------------
# STATISTICS
# --------------------------------------------------

elif page == "Statistics":

    st.title("📊 Statistical Analysis")

    numeric_df = filtered_df.select_dtypes(
        include=np.number
    )

    if numeric_df.empty:

        st.warning("No numeric columns are available.")

    else:

        st.subheader("Descriptive Statistics")

        st.dataframe(
            numeric_df.describe().T,
            use_container_width=True
        )

        st.subheader("Correlation Matrix")

        if len(numeric_df.columns) >= 2:

            corr = numeric_df.corr()

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            image = ax.imshow(
                corr,
                cmap="coolwarm",
                aspect="auto",
                vmin=-1,
                vmax=1
            )

            ax.set_xticks(
                range(len(corr.columns))
            )

            ax.set_yticks(
                range(len(corr.columns))
            )

            ax.set_xticklabels(
                corr.columns,
                rotation=90,
                color="white"
            )

            ax.set_yticklabels(
                corr.columns,
                color="white"
            )

            ax.set_facecolor("#0e1117")
            fig.patch.set_facecolor("#0e1117")

            fig.colorbar(
                image,
                ax=ax
            )

            fig.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

        else:

            st.info(
                "At least two numeric columns are required "
                "for correlation."
            )

        selected_column = st.selectbox(
            "Choose a column for distribution analysis",
            numeric_df.columns
        )

        st.subheader(
            f"Distribution: {selected_column}"
        )

        fig, ax = plt.subplots()

        ax.hist(
            numeric_df[selected_column].dropna(),
            bins=25,
            edgecolor="black"
        )

        ax.set_xlabel(selected_column)
        ax.set_ylabel("Frequency")

        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")

        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

        st.pyplot(fig)

        plt.close(fig)


# --------------------------------------------------
# EXPLORE DATA
# --------------------------------------------------

elif page == "Explore Data":

    st.title("🔎 Explore Data")

    search = st.text_input(
        "Search across all columns"
    )

    explore_df = filtered_df.copy()

    if search:

        mask = explore_df.astype(str).apply(
            lambda column: column.str.contains(
                search,
                case=False,
                na=False,
                regex=False
            )
        ).any(axis=1)

        explore_df = explore_df[mask]

    st.write(
        "Matching records:",
        len(explore_df)
    )

    st.dataframe(
        explore_df,
        use_container_width=True
    )

    st.download_button(
        "Download Filtered CSV",
        data=explore_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="job_market_filtered.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.caption("JobLens | Job Market Analytics")
