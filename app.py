
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


st.set_page_config(
    page_title="Job Market Analytics",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)



st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
    color: #1f2937;
}

[data-testid="stHeader"] {
    background-color: #f4f7fb;
}

[data-testid="stAppViewContainer"] {
    background-color: #f4f7fb;
}

h1, h2, h3, h4 {
    color: #17365d !important;
}

.stMarkdown, .stMarkdown p, .stMarkdown li {
    color: #263445;
}

[data-testid="stSidebar"] {
    background-color: #e8edf5;
}

[data-testid="stSidebar"] * {
    color: #1f2937;
}

[data-testid="stMetric"] {
    background-color: #ffffff;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #dce4ef;
}

[data-testid="stMetricLabel"] {
    color: #526174 !important;
}

[data-testid="stMetricValue"] {
    color: #17365d !important;
}

.stButton > button {
    background-color: #17365d;
    color: white !important;
    border-radius: 8px;
    border: none;
    padding: 8px 16px;
}

.stButton > button:hover {
    background-color: #28558a;
    color: white !important;
}

[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 8px;
}

div[data-testid="stAlert"] p {
    color: inherit;
}
</style>
""", unsafe_allow_html=True)



st.title("💼 Job Market Analytics Dashboard")

st.markdown(
    "Explore job opportunities, salary trends, hiring companies, "
    "locations, and in-demand skills."
)



uploaded_file = st.sidebar.file_uploader(
    "Upload Your Own Dataset",
    type=["csv"]
)

if uploaded_file is None:

    st.info("Please upload your job-market CSV dataset to begin.")

    st.markdown("## Welcome to Job Market Analytics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Analysis Type", "Job Market")
    col2.metric("Supported Format", "CSV")
    col3.metric("Dashboard", "Interactive")

    st.markdown("""
    ### Features

    - 📊 Job market overview
    - 💼 Job title analysis
    - 💰 Salary distribution and comparisons
    - 🧠 In-demand skills analysis
    - 🏢 Company hiring analysis
    - 📍 Location analysis
    - 📈 Job posting trends
    - 📥 Download filtered datasets

    ### How to use

    1. Upload your own CSV dataset.
    2. Review the dataset and its columns.
    3. Filter jobs by title, company, location, or work type.
    4. Explore the different analytics pages.
    5. Download your filtered data.
    """)

    st.stop()




try:
    df = pd.read_csv(uploaded_file)

except Exception as e:
    st.error(f"Unable to read the CSV file: {e}")
    st.stop()

if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

# Clean column names

df.columns = df.columns.astype(str).str.strip()

# Remove duplicate column names

df = df.loc[:, ~df.columns.duplicated()]

# Remove duplicate records

df = df.drop_duplicates().reset_index(drop=True)

if len(df.columns) == 0:
    st.error("No valid columns were found in the uploaded file.")
    st.stop()



def find_column(possible_names):

    normalized_names = {
        re.sub(r"[^a-z0-9]", "", name.lower())
        for name in possible_names
    }

    for column in df.columns:

        normalized_column = re.sub(
            r"[^a-z0-9]",
            "",
            column.lower()
        )

        if normalized_column in normalized_names:
            return column

    return None


title_col = find_column([
    "job_title",
    "title",
    "job",
    "position",
    "role",
    "designation",
    "job_role"
])

company_col = find_column([
    "company",
    "company_name",
    "employer",
    "organization",
    "organisation"
])

location_col = find_column([
    "location",
    "job_location",
    "city",
    "work_location",
    "job_city"
])

salary_col = find_column([
    "salary",
    "average_salary",
    "salary_avg",
    "annual_salary",
    "salary_in_usd",
    "salary_yearly",
    "estimated_salary",
    "salary_estimate",
    "salary_usd"
])

salary_min_col = find_column([
    "salary_min",
    "min_salary",
    "minimum_salary",
    "salary_from"
])

salary_max_col = find_column([
    "salary_max",
    "max_salary",
    "maximum_salary",
    "salary_to"
])

skills_col = find_column([
    "skills",
    "required_skills",
    "skill",
    "technical_skills",
    "skill_set"
])

date_col = find_column([
    "date_posted",
    "posting_date",
    "posted_date",
    "job_posted_date",
    "created_at",
    "publication_date",
    "post_date"
])

work_type_col = find_column([
    "work_type",
    "employment_type",
    "job_type",
    "work_mode",
    "remote_type",
    "job_schedule"
])

experience_col = find_column([
    "experience_level",
    "experience",
    "seniority",
    "years_experience",
    "experience_required"
])




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
        .str.replace(" ", "", regex=False)
        .str.replace(r"[^\d.\-]", "", regex=True)
    )

    return pd.to_numeric(cleaned, errors="coerce")


if salary_col is not None:

    df[salary_col] = clean_salary(df[salary_col])

elif salary_min_col and salary_max_col:

    df[salary_min_col] = clean_salary(df[salary_min_col])
    df[salary_max_col] = clean_salary(df[salary_max_col])

    df["Average Salary"] = (
        df[salary_min_col] + df[salary_max_col]
    ) / 2

    salary_col = "Average Salary"

elif salary_min_col:

    df[salary_min_col] = clean_salary(df[salary_min_col])
    salary_col = salary_min_col

elif salary_max_col:

    df[salary_max_col] = clean_salary(df[salary_max_col])
    salary_col = salary_max_col



if date_col is not None:

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )


# --------------------------------------------------
# SIDEBAR INFORMATION
# --------------------------------------------------

st.sidebar.success("Dataset uploaded successfully!")

st.sidebar.write("Total Records:", len(df))
st.sidebar.write("Total Columns:", len(df.columns))


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Dataset Explorer",
        "Job Analysis",
        "Salary Analysis",
        "Skills Analysis",
        "Company Analysis",
        "Location Analysis",
        "Job Trends",
        "Download Data"
    ]
)



st.sidebar.divider()
st.sidebar.subheader("Dashboard Filters")

filtered_df = df.copy()


def apply_filter(data, column, label):

    if column is None:
        return data

    values = (
        data[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    values = sorted(values)

    if not values:
        return data

    selected_values = st.sidebar.multiselect(
        label,
        options=values,
        default=[],
        key="filter_" + column
    )

    if selected_values:

        data = data[
            data[column].astype(str).isin(selected_values)
        ]

    return data


filtered_df = apply_filter(
    filtered_df,
    title_col,
    "Job Title"
)

filtered_df = apply_filter(
    filtered_df,
    company_col,
    "Company"
)

filtered_df = apply_filter(
    filtered_df,
    location_col,
    "Location"
)

filtered_df = apply_filter(
    filtered_df,
    work_type_col,
    "Work Type"
)




def show_category_chart(data, column, title, limit=15):

    st.subheader(title)

    if column is None:
        st.warning("The required column was not found.")
        return

    counts = (
        data[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    counts = counts[counts != ""].value_counts().head(limit)

    if counts.empty:
        st.info("No data is available for this chart.")
        return

    chart_data = counts.rename_axis(column).reset_index(
        name="Job Count"
    )

    st.bar_chart(
        chart_data.set_index(column)["Job Count"]
    )

    with st.expander("View chart data"):
        st.dataframe(chart_data, use_container_width=True)




def get_skill_counts(data):

    if skills_col is None:
        return pd.Series(dtype="int64")

    skill_list = []

    for value in data[skills_col].dropna():

        for skill in re.split(r"[,;|]", str(value)):

            skill = skill.strip()

            if skill:
                skill_list.append(skill)

    if not skill_list:
        return pd.Series(dtype="int64")

    return pd.Series(skill_list).value_counts()




if page == "Dashboard Overview":

    st.header("📊 Job Market Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Job Postings",
        f"{len(filtered_df):,}"
    )

    company_count = (
        filtered_df[company_col].nunique()
        if company_col else 0
    )

    location_count = (
        filtered_df[location_col].nunique()
        if location_col else 0
    )

    title_count = (
        filtered_df[title_col].nunique()
        if title_col else 0
    )

    col2.metric("Companies", f"{company_count:,}")
    col3.metric("Locations", f"{location_count:,}")
    col4.metric("Unique Job Titles", f"{title_count:,}")

    if salary_col is not None:

        salaries = filtered_df[salary_col].dropna()

        if not salaries.empty:

            st.subheader("💰 Salary Overview")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Average Salary",
                f"{salaries.mean():,.2f}"
            )

            col2.metric(
                "Median Salary",
                f"{salaries.median():,.2f}"
            )

            col3.metric(
                "Maximum Salary",
                f"{salaries.max():,.2f}"
            )

    col1, col2 = st.columns(2)

    with col1:

        show_category_chart(
            filtered_df,
            title_col,
            "💼 Top Job Titles",
            10
        )

    with col2:

        show_category_chart(
            filtered_df,
            company_col,
            "🏢 Top Hiring Companies",
            10
        )

    st.subheader("🧠 Most Common Skills")

    skill_counts = get_skill_counts(filtered_df)

    if not skill_counts.empty:

        st.bar_chart(skill_counts.head(15))

    else:

        st.info(
            "A skills column containing skill names was not found."
        )

    if experience_col is not None:

        show_category_chart(
            filtered_df,
            experience_col,
            "Experience Level Distribution",
            10
        )



elif page == "Dataset Explorer":

    st.header("📋 Dataset Explorer")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(filtered_df))
    col2.metric("Total Columns", len(filtered_df.columns))

    col3.metric(
        "Missing Values",
        int(filtered_df.isnull().sum().sum())
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        filtered_df.head(20),
        use_container_width=True
    )

    st.subheader("Column Information")

    info_df = pd.DataFrame({
        "Column": filtered_df.columns,
        "Data Type": filtered_df.dtypes.astype(str).values,
        "Missing Values": filtered_df.isnull().sum().values,
        "Unique Values": filtered_df.nunique().values
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        filtered_df.describe(include="all").T,
        use_container_width=True
    )

    st.subheader("Missing Values by Column")

    missing = filtered_df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if not missing.empty:

        st.bar_chart(missing)

    else:

        st.success("No missing values were found.")

    st.subheader("Duplicate Records")

    st.write(
        "Duplicate records remaining after initial cleaning:",
        int(filtered_df.duplicated().sum())
    )




elif page == "Job Analysis":

    st.header("💼 Job Analysis")

    if title_col is None:

        st.warning(
            "No job title column was detected. "
            "Check the column names in Dataset Explorer."
        )

    else:

        job_counts = (
            filtered_df[title_col]
            .dropna()
            .astype(str)
            .str.strip()
        )

        job_counts = job_counts[
            job_counts != ""
        ].value_counts()

        col1, col2 = st.columns(2)

        col1.metric(
            "Unique Job Titles",
            len(job_counts)
        )

        col2.metric(
            "Most Common Job Title Count",
            int(job_counts.iloc[0]) if not job_counts.empty else 0
        )

        st.subheader("Job Title Distribution")

        if not job_counts.empty:

            st.bar_chart(
                job_counts.head(20)
            )

            with st.expander("View job title counts"):

                st.dataframe(
                    job_counts.rename(
                        "Number of Postings"
                    ).to_frame(),
                    use_container_width=True
                )

        else:

            st.info("No job titles are available.")

        st.subheader("Search Job Titles")

        search = st.text_input(
            "Enter a job title or keyword",
            key="job_search"
        )

        jobs = filtered_df.copy()

        if search:

            jobs = jobs[
                jobs[title_col]
                .astype(str)
                .str.contains(
                    search,
                    case=False,
                    na=False,
                    regex=False
                )
            ]

        st.write("Matching job postings:", len(jobs))

        st.dataframe(
            jobs,
            use_container_width=True
        )


# --------------------------------------------------
# SALARY ANALYSIS
# --------------------------------------------------

elif page == "Salary Analysis":

    st.header("💰 Salary Analysis")

    if salary_col is None:

        st.warning(
            "No salary column was detected. "
            "Your dataset needs a numeric salary field."
        )

    else:

        salary_data = pd.to_numeric(
            filtered_df[salary_col],
            errors="coerce"
        ).dropna()

        salary_data = salary_data[
            np.isfinite(salary_data)
        ]

        if salary_data.empty:

            st.warning("No valid salary values were found.")

        else:

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Average Salary",
                f"{salary_data.mean():,.2f}"
            )

            col2.metric(
                "Median Salary",
                f"{salary_data.median():,.2f}"
            )

            col3.metric(
                "Minimum Salary",
                f"{salary_data.min():,.2f}"
            )

            col4.metric(
                "Maximum Salary",
                f"{salary_data.max():,.2f}"
            )

            st.subheader("Salary Distribution")

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.hist(
                salary_data,
                bins=min(30, max(5, salary_data.nunique())),
                edgecolor="black"
            )

            ax.set_xlabel(salary_col)
            ax.set_ylabel("Number of Job Postings")
            ax.set_title("Distribution of Salaries")

            ax.grid(axis="y", alpha=0.25)

            fig.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

            st.subheader("Salary Statistics")

            st.dataframe(
                salary_data.describe().to_frame(
                    "Salary Statistics"
                ),
                use_container_width=True
            )

            if title_col:

                st.subheader("Average Salary by Job Title")

                salary_by_title = (
                    filtered_df
                    .dropna(subset=[salary_col, title_col])
                    .groupby(title_col)[salary_col]
                    .mean()
                    .sort_values(ascending=False)
                    .head(15)
                )

                if not salary_by_title.empty:
                    st.bar_chart(salary_by_title)

            if location_col:

                st.subheader("Average Salary by Location")

                salary_by_location = (
                    filtered_df
                    .dropna(subset=[salary_col, location_col])
                    .groupby(location_col)[salary_col]
                    .mean()
                    .sort_values(ascending=False)
                    .head(15)
                )

                if not salary_by_location.empty:
                    st.bar_chart(salary_by_location)

            if work_type_col:

                st.subheader("Average Salary by Work Type")

                salary_by_work_type = (
                    filtered_df
                    .dropna(subset=[salary_col, work_type_col])
                    .groupby(work_type_col)[salary_col]
                    .mean()
                    .sort_values(ascending=False)
                )

                if not salary_by_work_type.empty:
                    st.bar_chart(salary_by_work_type)



elif page == "Skills Analysis":

    st.header("🧠 Skills Analysis")

    if skills_col is None:

        st.warning(
            "No skills column was detected. "
            "Use a dataset containing Skills or Required Skills."
        )

    else:

        skill_counts = get_skill_counts(filtered_df)

        if skill_counts.empty:

            st.info("No skill information is available.")

        else:

            col1, col2 = st.columns(2)

            col1.metric(
                "Unique Skills",
                len(skill_counts)
            )

            col2.metric(
                "Total Skill Mentions",
                int(skill_counts.sum())
            )

            st.subheader("Top 20 In-Demand Skills")

            st.bar_chart(
                skill_counts.head(20)
            )

            st.subheader("Search for a Skill")

            skill_search = st.text_input(
                "Enter a skill name",
                key="skill_search"
            )

            if skill_search:

                matching_skills = skill_counts[
                    skill_counts.index.str.contains(
                        skill_search,
                        case=False,
                        regex=False
                    )
                ]

                st.dataframe(
                    matching_skills.rename(
                        "Frequency"
                    ).to_frame(),
                    use_container_width=True
                )

            st.subheader("Complete Skill Frequency Table")

            st.dataframe(
                skill_counts.rename(
                    "Frequency"
                ).to_frame(),
                use_container_width=True
            )



elif page == "Company Analysis":

    st.header("🏢 Company Analysis")

    if company_col is None:

        st.warning("No company column was detected.")

    else:

        company_counts = (
            filtered_df[company_col]
            .dropna()
            .astype(str)
            .str.strip()
        )

        company_counts = company_counts[
            company_counts != ""
        ].value_counts()

        col1, col2 = st.columns(2)

        col1.metric(
            "Unique Companies",
            len(company_counts)
        )

        col2.metric(
            "Most Job Postings",
            int(company_counts.iloc[0])
            if not company_counts.empty else 0
        )

        st.subheader("Companies with the Most Job Postings")

        if not company_counts.empty:

            st.bar_chart(
                company_counts.head(20)
            )

        st.subheader("Search Companies")

        company_search = st.text_input(
            "Enter a company name",
            key="company_search"
        )

        companies = filtered_df.copy()

        if company_search:

            companies = companies[
                companies[company_col]
                .astype(str)
                .str.contains(
                    company_search,
                    case=False,
                    na=False,
                    regex=False
                )
            ]

        st.write("Matching records:", len(companies))

        st.dataframe(
            companies,
            use_container_width=True
        )

        if salary_col:

            st.subheader("Average Salary by Company")

            company_salary = (
                filtered_df
                .dropna(subset=[company_col, salary_col])
                .groupby(company_col)[salary_col]
                .mean()
                .sort_values(ascending=False)
                .head(15)
            )

            if not company_salary.empty:
                st.bar_chart(company_salary)




elif page == "Location Analysis":

    st.header("📍 Location Analysis")

    if location_col is None:

        st.warning("No location column was detected.")

    else:

        location_counts = (
            filtered_df[location_col]
            .dropna()
            .astype(str)
            .str.strip()
        )

        location_counts = location_counts[
            location_counts != ""
        ].value_counts()

        col1, col2 = st.columns(2)

        col1.metric(
            "Unique Locations",
            len(location_counts)
        )

        col2.metric(
            "Most Job Postings",
            int(location_counts.iloc[0])
            if not location_counts.empty else 0
        )

        st.subheader("Job Opportunities by Location")

        if not location_counts.empty:

            st.bar_chart(
                location_counts.head(20)
            )

        if salary_col:

            st.subheader("Average Salary by Location")

            location_salary = (
                filtered_df
                .dropna(subset=[location_col, salary_col])
                .groupby(location_col)[salary_col]
                .mean()
                .sort_values(ascending=False)
                .head(20)
            )

            if not location_salary.empty:
                st.bar_chart(location_salary)

        st.subheader("Location Details")

        st.dataframe(
            filtered_df,
            use_container_width=True
        )


# --------------------------------------------------
# JOB TRENDS
# --------------------------------------------------

elif page == "Job Trends":

    st.header("📈 Job Posting Trends")

    if date_col is None:

        st.warning(
            "No recognizable date column was detected. "
            "Use a column such as Date Posted or Posting Date."
        )

    else:

        trend_data = filtered_df.dropna(
            subset=[date_col]
        ).copy()

        if trend_data.empty:

            st.warning("No valid posting dates are available.")

        else:

            trend_data["Month"] = (
                trend_data[date_col]
                .dt.to_period("M")
                .astype(str)
            )

            monthly_counts = (
                trend_data.groupby("Month")
                .size()
                .sort_index()
            )

            st.subheader("Monthly Job Postings")

            st.line_chart(monthly_counts)

            monthly_table = monthly_counts.rename(
                "Job Postings"
            ).to_frame()

            monthly_table["Change from Previous Month"] = (
                monthly_table["Job Postings"].diff()
            )

            st.subheader("Monthly Posting Summary")

            st.dataframe(
                monthly_table,
                use_container_width=True
            )

            if title_col:

                st.subheader("Job Trends by Job Title")

                available_titles = sorted(
                    trend_data[title_col]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                selected_titles = st.multiselect(
                    "Select job titles to compare",
                    options=available_titles,
                    max_selections=10
                )

                if selected_titles:

                    selected_data = trend_data[
                        trend_data[title_col]
                        .astype(str)
                        .isin(selected_titles)
                    ]

                    trend_pivot = pd.crosstab(
                        selected_data["Month"],
                        selected_data[title_col]
                    )

                    st.line_chart(trend_pivot)


# --------------------------------------------------
# DOWNLOAD DATA
# --------------------------------------------------

elif page == "Download Data":

    st.header("📥 Download Job Market Data")

    st.write(
        "Download the complete dataset or the filtered dataset."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Original Records",
        len(df)
    )

    col2.metric(
        "Filtered Records",
        len(filtered_df)
    )

    col3.metric(
        "Available Columns",
        len(filtered_df.columns)
    )

    st.subheader("Filtered Dataset Preview")

    st.dataframe(
        filtered_df.head(20),
        use_container_width=True
    )

    st.download_button(
        label="Download Filtered Dataset",
        data=filtered_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="filtered_job_market_data.csv",
        mime="text/csv"
    )

    st.download_button(
        label="Download Complete Dataset",
        data=df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="job_market_data.csv",
        mime="text/csv"
    )

    st.success(
        f"{len(filtered_df)} records are currently selected."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.sidebar.divider()

st.sidebar.caption(
    "Job Market Analytics | Streamlit Dashboard"
)
