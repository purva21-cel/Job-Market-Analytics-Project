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
            # Use plain string labels instead of pandas Interval objects.
            # Interval indexes can trigger Altair schema validation errors.
            values = chart_data.to_numpy(dtype=float)
            if len(values) == 0:
                st.info("There are no non-missing values to plot for this column.")
            elif np.nanmin(values) == np.nanmax(values):
                st.bar_chart(pd.DataFrame({
                    "Count": [len(values)]
                }, index=[str(values[0])]))
            else:
                bin_count = min(20, max(1, pd.Series(values).nunique()))
                counts, edges = np.histogram(values, bins=bin_count)
                labels = [
                    f"{edges[i]:.3g}–{edges[i + 1]:.3g}"
                    for i in range(len(edges) - 1)
                ]
                hist = pd.DataFrame({"Count": counts}, index=labels)
                hist.index.name = "Value range"
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
