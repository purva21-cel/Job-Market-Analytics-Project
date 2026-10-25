
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

st.set_page_config(
    page_title="Job Market Analytics",
    page_icon="💼",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}
h1 {
    color: #17365d;
}
h2, h3 {
    color: #24476b;
}
div[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #e1e6ef;
}
.stButton > button {
    background-color: #17365d;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# Application title

st.title("💼 Job Market Analytics Dashboard")

st.write(
    "Explore job opportunities, salary trends, "
    "hiring companies, locations, and in-demand skills."
)


# Dataset upload

uploaded_file = st.sidebar.file_uploader(
    "Upload Your Own Dataset",
    type=["csv"]
)

if uploaded_file is None:

    st.info("Upload a job-market CSV file to access the dashboard.")

    st.subheader("Welcome to Job Market Analytics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Analysis Type", "Job Market")
    col2.metric("Input Format", "CSV")
    col3.metric("Dashboard", "Interactive")

    st.markdown("""
    ### What can you explore?

    - Job market overview
    - Job titles and employment demand
    - Salary distribution and comparisons
    - Most frequently requested skills
    - Company hiring activity
    - Job opportunities by location
    - Job posting trends
    - Filtered dataset downloads

    ### Getting started

    1. Upload your job-market CSV dataset.
    2. Review the dataset and its columns.
    3. Apply filters to explore relevant opportunities.
    4. Navigate between analysis pages.
    5. Download filtered records when required.
    """)

    st.stop()


# Load uploaded dataset

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the CSV file: {e}")
    st.stop()


if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()


# Clean column names

df.columns = df.columns.astype(str).str.strip()
df = df.loc[:, ~df.columns.duplicated()]
df = df.drop_duplicates().reset_index(drop=True)


# Detect common column names

def find_column(names):

    for col in df.columns:

        normalized = re.sub(
            r"[^a-z0-9]",
            "",
            col.lower()
        )

        for name in names:

            target = re.sub(
                r"[^a-z0-9]",
                "",
                name.lower()
            )

            if normalized == target:
                return col

    return None


title_col = find_column([
    "job_title",
    "title",
    "job",
    "position",
    "role",
    "designation"
])

company_col = find_column([
    "company",
    "company_name",
    "employer",
    "organization"
])

location_col = find_column([
    "location",
    "job_location",
    "city",
    "work_location"
])

salary_col = find_column([
    "salary",
    "average_salary",
    "salary_avg",
    "annual_salary",
    "salary_in_usd",
    "salary_yearly",
    "estimated_salary"
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
    "technical_skills"
])

date_col = find_column([
    "date_posted",
    "posting_date",
    "posted_date",
    "job_posted_date",
    "created_at",
    "publication_date"
])

work_type_col = find_column([
    "work_type",
    "employment_type",
    "job_type",
    "work_mode",
    "remote_type"
])

experience_col = find_column([
    "experience_level",
    "experience",
    "seniority",
    "years_experience"
])


# Convert salary values to numeric where possible

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


if salary_col:

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


# Convert date column

if date_col:

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )


# Sidebar navigation

st.sidebar.success("Dataset uploaded successfully!")

st.sidebar.write("Records:", len(df))
st.sidebar.write("Columns:", len(df.columns))

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


# Sidebar filters

st.sidebar.divider()
st.sidebar.subheader("Dashboard Filters")

filtered_df = df.copy()


def filter_column(data, column, label):

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

    selected = st.sidebar.multiselect(
        label,
        values,
        default=[]
    )

    if selected:
        data = data[
            data[column].astype(str).isin(selected)
        ]

    return data


filtered_df = filter_column(
    filtered_df,
    title_col,
    "Job Title"
)

filtered_df = filter_column(
    filtered_df,
    company_col,
    "Company"
)

filtered_df = filter_column(
    filtered_df,
    location_col,
    "Location"
)

filtered_df = filter_column(
    filtered_df,
    work_type_col,
    "Work Type"
)


# Dataset explorer

if page == "Dataset Explorer":

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

    st.dataframe(info_df, use_container_width=True)

    st.subheader("Statistical Summary")

    st.dataframe(
        filtered_df.describe(include="all").T,
        use_container_width=True
    )

    st.subheader("Missing Values")

    missing = filtered_df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:
        st.dataframe(
            missing.rename("Missing Count").to_frame()
        )
    else:
        st.success("No missing values found.")


# Dashboard overview

elif page == "Dashboard Overview":

    st.header("📊 Job Market Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Job Postings",
        f"{len(filtered_df):,}"
    )

    if company_col:
        company_count = filtered_df[company_col].nunique()
    else:
        company_count = 0

    col2.metric(
        "Companies",
        f"{company_count:,}"
    )

    if location_col:
        location_count = filtered_df[location_col].nunique()
    else:
        location_count = 0

    col3.metric(
        "Locations",
        f"{location_count:,}"
    )

    if title_col:
        role_count = filtered_df[title_col].nunique()
    else:
        role_count = 0

    col4.metric(
        "Unique Job Titles",
        f"{role_count:,}"
    )

    if salary_col:

        average_salary = filtered_df[salary_col].mean()

        st.subheader("Salary Overview")

        if pd.notna(average_salary):

            st.metric(
                "Average Salary",
                f"{average_salary:,.2f}"
            )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Top Job Titles")

        if title_col:

            top_titles = (
                filtered_df[title_col]
                .dropna()
                .astype(str)
                .value_counts()
                .head(10)
            )

            if not top_titles.empty:
                st.bar_chart(top_titles)

            else:
                st.info("No job-title data available.")

        else:
            st.info("No job-title column was detected.")

    with col2:

        st.subheader("Top Hiring Companies")

        if company_col:

            top_companies = (
                filtered_df[company_col]
                .dropna()
                .astype(str)
                .value_counts()
                .head(10)
            )

            if not top_companies.empty:
                st.bar_chart(top_companies)

            else:
                st.info("No company data available.")

        else:
            st.info("No company column was detected.")

    st.subheader("Most Common Skills")

    if skills_col:

        all_skills = []

        for value in filtered_df[skills_col].dropna():

            skills = re.split(
                r"[,;|]",
                str(value)
            )

            for skill in skills:

                skill = skill.strip()

                if skill:
                    all_skills.append(skill)

        if all_skills:

            skill_counts = pd.Series(
                all_skills
            ).value_counts().head(15)

            st.bar_chart(skill_counts)

        else:
            st.info("No skills were found.")

    else:
        st.info("No skills column was detected.")


# Job analysis

elif page == "Job Analysis":

    st.header("💼 Job Analysis")

    if title_col is None:

        st.warning(
            "A job title column could not be detected. "
            "Check your dataset column names."
        )

    else:

        job_counts = (
            filtered_df[title_col]
            .dropna()
            .astype(str)
            .value_counts()
        )

        st.subheader("Job Title Distribution")

        if len(job_counts) > 0:
            st.bar_chart(job_counts.head(20))

        st.subheader("Search Job Titles")

        search = st.text_input(
            "Enter a keyword to search job titles"
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

        st.dataframe(
            jobs,
            use_container_width=True
        )

        st.write(
            f"Matching records: {len(jobs)}"
        )


# Salary analysis

elif page == "Salary Analysis":

    st.header("💰 Salary Analysis")

    if salary_col is None:

        st.warning(
            "No salary column was detected. "
            "Your dataset needs a numeric salary field."
        )

    else:

        salary_data = filtered_df[
            salary_col
        ].dropna()

        salary_data = salary_data[
            np.isfinite(salary_data)
        ]

        if salary_data.empty:

            st.warning(
                "No valid numeric salary values are available."
            )

        else:

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Average Salary",
                f"{salary_data.mean():,.2f}"
            )

            col2.metric(
                "Median Salary",
                f"{salary_data.median():,.2f}"
            )

            col3.metric(
                "Maximum Salary",
                f"{salary_data.max():,.2f}"
            )

            st.subheader("Salary Distribution")

            fig, ax = plt.subplots()

            ax.hist(
                salary_data,
                bins=30,
                edgecolor="black"
            )

            ax.set_xlabel(salary_col)
            ax.set_ylabel("Number of Records")
            ax.set_title("Salary Distribution")

            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Salary Statistics")

            st.dataframe(
                salary_data.describe().to_frame(
                    "Salary Statistics"
                )
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


# Skills analysis

elif page == "Skills Analysis":

    st.header("🧠 Skills Analysis")

    if skills_col is None:

        st.warning(
            "No skills column was detected. "
            "Use a dataset containing a Skills or Required Skills column."
        )

    else:

        skill_list = []

        for value in filtered_df[skills_col].dropna():

            skills = re.split(
                r"[,;|]",
                str(value)
            )

            for skill in skills:

                skill = skill.strip()

                if skill:
                    skill_list.append(skill)

        if skill_list:

            skill_counts = pd.Series(
                skill_list
            ).value_counts()

            col1, col2 = st.columns(2)

            col1.metric(
                "Unique Skills",
                skill_counts.size
            )

            col2.metric(
                "Skill Mentions",
                len(skill_list)
            )

            st.subheader("Top 20 In-Demand Skills")

            st.bar_chart(
                skill_counts.head(20)
            )

            st.subheader("Search for a Skill")

            skill_search = st.text_input(
                "Enter a skill"
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
                    ).to_frame()
                )

            st.subheader("Complete Skills Frequency")

            st.dataframe(
                skill_counts.rename(
                    "Frequency"
                ).to_frame()
            )

        else:

            st.info(
                "The selected dataset does not contain "
                "usable skill information."
            )


# Company analysis

elif page == "Company Analysis":

    st.header("🏢 Company Analysis")

    if company_col is None:

        st.warning(
            "No company column was detected."
        )

    else:

        company_counts = (
            filtered_df[company_col]
            .dropna()
            .astype(str)
            .value_counts()
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Unique Companies",
            company_counts.size
        )

        if len(company_counts) > 0:

            col2.metric(
                "Top Company Postings",
                int(company_counts.iloc[0])
            )

        st.subheader("Companies with the Most Job Postings")

        st.bar_chart(
            company_counts.head(20)
        )

        st.subheader("Search Companies")

        company_search = st.text_input(
            "Enter a company name"
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


# Location analysis

elif page == "Location Analysis":

    st.header("📍 Location Analysis")

    if location_col is None:

        st.warning(
            "No location column was detected."
        )

    else:

        location_counts = (
            filtered_df[location_col]
            .dropna()
            .astype(str)
            .value_counts()
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Unique Locations",
            location_counts.size
        )

        if len(location_counts) > 0:

            col2.metric(
                "Top Location Postings",
                int(location_counts.iloc[0])
            )

        st.subheader("Job Opportunities by Location")

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


# Job trends

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

            st.warning(
                "No valid posting dates are available."
            )

        else:

            trend_data["Month"] = (
                trend_data[date_col]
                .dt.to_period("M")
                .astype(str)
            )

            monthly_counts = (
                trend_data["Month"]
                .value_counts()
                .sort_index()
            )

            st.subheader("Monthly Job Postings")

            st.line_chart(monthly_counts)

            st.subheader("Monthly Posting Summary")

            monthly_table = monthly_counts.rename(
                "Job Postings"
            ).to_frame()

            monthly_table["Change"] = (
                monthly_table["Job Postings"].diff()
            )

            st.dataframe(
                monthly_table,
                use_container_width=True
            )

            if title_col:

                st.subheader("Job Posting Trends by Job Title")

                selected_titles = st.multiselect(
                    "Select job titles",
                    sorted(
                        trend_data[title_col]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )[:100]
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


# Download data

elif page == "Download Data":

    st.header("📥 Download Job Market Data")

    st.write(
        "Download the original dataset or the filtered dataset."
    )

    st.subheader("Filtered Dataset Preview")

    st.dataframe(
        filtered_df.head(20),
        use_container_width=True
    )

    st.download_button(
        label="Download Filtered Data as CSV",
        data=filtered_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="filtered_job_market_data.csv",
        mime="text/csv"
    )

    st.download_button(
        label="Download Complete Dataset as CSV",
        data=df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="job_market_data.csv",
        mime="text/csv"
    )

    st.success(
        f"{len(filtered_df)} records are currently selected."
    )

