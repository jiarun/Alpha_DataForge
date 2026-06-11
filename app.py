import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import processing as proc

st.set_page_config(page_title="Alpha DataForge", layout="wide")
st.title("🔥 Alpha DataForge")


def compute_quality_score(dataframe):
    """Compute a 0-100 data quality score."""
    n_rows, n_cols = dataframe.shape
    if n_rows == 0 or n_cols == 0:
        return 0, {}

    # Completeness (40%): % of non-null cells
    completeness = 1 - dataframe.isnull().sum().sum() / (n_rows * n_cols)

    # Uniqueness (20%): % of non-duplicate rows
    uniqueness = 1 - dataframe.duplicated().sum() / n_rows

    # Consistency (20%): % of columns with consistent types (no mixed types)
    consistency_scores = []
    for col in dataframe.columns:
        try:
            if dataframe[col].dtype == "object":
                numeric_ratio = pd.to_numeric(dataframe[col], errors="coerce").notna().sum() / max(dataframe[col].notna().sum(), 1)
                consistency_scores.append(1 if numeric_ratio < 0.1 or numeric_ratio > 0.9 else 0.5)
            else:
                consistency_scores.append(1.0)
        except Exception:
            consistency_scores.append(0.5)
    consistency = np.mean(consistency_scores) if consistency_scores else 1.0

    # Validity (20%): no extreme outliers in numeric columns
    numeric_cols = dataframe.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        outlier_fractions = []
        for col in numeric_cols:
            data = dataframe[col].dropna()
            if len(data) > 3:
                q1, q3 = data.quantile(0.25), data.quantile(0.75)
                iqr = q3 - q1
                outliers = ((data < q1 - 3 * iqr) | (data > q3 + 3 * iqr)).sum()
                outlier_fractions.append(1 - outliers / len(data))
        validity = np.mean(outlier_fractions) if outlier_fractions else 1.0
    else:
        validity = 1.0

    score = (completeness * 40 + uniqueness * 20 + consistency * 20 + validity * 20)
    breakdown = {
        "Completeness": round(completeness * 100, 1),
        "Uniqueness": round(uniqueness * 100, 1),
        "Consistency": round(consistency * 100, 1),
        "Validity": round(validity * 100, 1),
    }
    return round(score, 1), breakdown


def log_operation(description):
    st.session_state.operation_history.append(description)

uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls", "tsv"])

if uploaded_file:
    ext = uploaded_file.name.split(".")[-1].lower()
    if ext == "csv":
        df = pd.read_csv(uploaded_file)
    elif ext == "tsv":
        df = pd.read_csv(uploaded_file, sep="\t")
    else:
        df = pd.read_excel(uploaded_file)

    if "processed_df" not in st.session_state:
        st.session_state.processed_df = df.copy()
    if "operation_history" not in st.session_state:
        st.session_state.operation_history = []

    st.success(f"Loaded **{uploaded_file.name}** — {df.shape[0]} rows × {df.shape[1]} columns")

    tab_overview, tab_stats, tab_viz, tab_filter, tab_process, tab_compare = st.tabs(
        ["Overview", "Statistics", "Visualize", "Filter", "🔧 Process", "📊 Compare"]
    )

    with tab_overview:
        # Data Quality Score
        score, breakdown = compute_quality_score(df)
        st.markdown("### Data Quality Score")
        score_cols = st.columns(5)
        with score_cols[0]:
            color = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
            st.metric("Overall", f"{color} {score}/100")
        for i, (key, val) in enumerate(breakdown.items()):
            with score_cols[i + 1]:
                st.metric(key, f"{val}%")

        st.dataframe(df.head(100), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Column Types**")
            st.dataframe(df.dtypes.rename("type").to_frame(), use_container_width=True)
        with col2:
            st.markdown("**Missing Values**")
            st.dataframe(df.isnull().sum().rename("nulls").to_frame(), use_container_width=True)

    with tab_stats:
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    with tab_viz:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        all_cols = df.columns.tolist()
        chart_type = st.selectbox("Chart type", ["Histogram", "Scatter", "Bar", "Box"])
        if chart_type == "Histogram" and numeric_cols:
            col = st.selectbox("Column", numeric_cols)
            st.plotly_chart(px.histogram(df, x=col), use_container_width=True)
        elif chart_type == "Scatter" and len(numeric_cols) >= 2:
            c1, c2 = st.columns(2)
            x = c1.selectbox("X", numeric_cols, index=0)
            y = c2.selectbox("Y", numeric_cols, index=min(1, len(numeric_cols) - 1))
            st.plotly_chart(px.scatter(df, x=x, y=y), use_container_width=True)
        elif chart_type == "Bar":
            col = st.selectbox("Column", all_cols)
            st.plotly_chart(px.bar(df[col].value_counts().head(20).reset_index(), x="index", y=col), use_container_width=True)
        elif chart_type == "Box" and numeric_cols:
            col = st.selectbox("Column", numeric_cols)
            st.plotly_chart(px.box(df, y=col), use_container_width=True)

    with tab_filter:
        col = st.selectbox("Filter column", all_cols, key="filter_col")
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 50:
            selected = st.multiselect("Values", unique_vals)
            if selected:
                st.dataframe(df[df[col].isin(selected)], use_container_width=True)
        else:
            query = st.text_input(f"Search in '{col}'")
            if query:
                mask = df[col].astype(str).str.contains(query, case=False, na=False)
                st.dataframe(df[mask], use_container_width=True)

    with tab_process:
        work_df = st.session_state.processed_df
        numeric_cols = work_df.select_dtypes(include="number").columns.tolist()
        cat_cols = work_df.select_dtypes(include=["object", "category"]).columns.tolist()
        all_cols = work_df.columns.tolist()

        st.info(f"Working dataset: {work_df.shape[0]} rows × {work_df.shape[1]} columns")

        CATEGORIES = [
            "Data Quality", "Numeric Cleaning", "Outlier Processing",
            "Distribution Conditioning", "Feature Scaling", "Categorical Processing",
            "Text Processing", "Datetime Processing", "Feature Engineering",
            "Time-Series Features", "Statistical Analysis", "Feature Selection",
            "Dimensionality Reduction", "Target Conditioning",
        ]

        category = st.selectbox("Category", CATEGORIES, key="proc_cat")

        # --- Data Quality ---
        if category == "Data Quality":
            action = st.selectbox("Action", [
                "Missing Value Detection", "Missing Value Imputation",
                "Duplicate Detection", "Duplicate Removal",
            ])
            if action == "Missing Value Detection":
                cols = st.multiselect("Columns", all_cols, default=all_cols, key="mvd_cols")
                if st.button("Run") and cols:
                    st.dataframe(proc.missing_value_detection(work_df, cols))

            elif action == "Missing Value Imputation":
                cols = st.multiselect("Columns", all_cols, key="mvi_cols")
                strategy = st.selectbox("Strategy", ["mean", "median", "mode", "ffill", "bfill", "zero", "drop_rows"])
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.missing_value_imputation(work_df, cols, strategy)
                    log_operation(f"Impute missing values ({strategy}) on {cols}")
                    st.rerun()

            elif action == "Duplicate Detection":
                if st.button("Run"):
                    dupes = proc.duplicate_detection(work_df)
                    st.write(f"Found {len(dupes)} duplicate rows")
                    st.dataframe(dupes.head(100))

            elif action == "Duplicate Removal":
                keep = st.selectbox("Keep", ["first", "last", False])
                if st.button("Apply"):
                    st.session_state.processed_df = proc.duplicate_removal(work_df, keep)
                    log_operation(f"Remove duplicates (keep={keep})")
                    st.rerun()

        # --- Numeric Cleaning ---
        elif category == "Numeric Cleaning":
            action = st.selectbox("Action", [
                "Numeric Conversion", "Precision Standardization",
            ])
            if action == "Numeric Conversion":
                cols = st.multiselect("Columns", cat_cols, key="nc_cols")
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.numeric_conversion(work_df, cols)
                    log_operation(f"Numeric conversion on {cols}")
                    st.rerun()
            elif action == "Precision Standardization":
                cols = st.multiselect("Columns", numeric_cols, key="ps_cols")
                decimals = st.number_input("Decimal places", 0, 10, 2)
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.precision_standardization(work_df, cols, decimals)
                    log_operation(f"Precision standardization ({decimals} dp) on {cols}")
                    st.rerun()

        # --- Outlier Processing ---
        elif category == "Outlier Processing":
            action = st.selectbox("Action", [
                "Z-Score Detection", "IQR Detection", "Isolation Forest",
                "Local Outlier Factor", "Outlier Capping", "Outlier Removal",
            ])
            cols = st.multiselect("Columns", numeric_cols, key="out_cols")
            if action == "Z-Score Detection":
                threshold = st.slider("Z threshold", 1.0, 5.0, 3.0, 0.1)
                if st.button("Run") and cols:
                    mask = proc.zscore_outlier_detection(work_df, cols, threshold)
                    st.write(f"Outliers per column: {mask.sum().to_dict()}")
                    st.dataframe(work_df[mask.any(axis=1)].head(100))
            elif action == "IQR Detection":
                factor = st.slider("IQR factor", 1.0, 3.0, 1.5, 0.1)
                if st.button("Run") and cols:
                    mask = proc.iqr_outlier_detection(work_df, cols, factor)
                    st.write(f"Outliers per column: {mask.sum().to_dict()}")
                    st.dataframe(work_df[mask.any(axis=1)].head(100))
            elif action == "Isolation Forest":
                contamination = st.slider("Contamination", 0.01, 0.5, 0.1, 0.01)
                if st.button("Run") and cols:
                    mask = proc.isolation_forest_detection(work_df, cols, contamination)
                    st.write(f"Anomalies detected: {mask.sum()}")
                    st.dataframe(work_df[mask].head(100))
            elif action == "Local Outlier Factor":
                n_neighbors = st.slider("Neighbors", 5, 50, 20)
                contamination = st.slider("Contamination", 0.01, 0.5, 0.1, 0.01, key="lof_cont")
                if st.button("Run") and cols:
                    mask = proc.local_outlier_factor(work_df, cols, n_neighbors, contamination)
                    st.write(f"Anomalies detected: {mask.sum()}")
                    st.dataframe(work_df[mask].head(100))
            elif action == "Outlier Capping":
                method = st.selectbox("Method", ["iqr", "percentile"])
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.outlier_capping(work_df, cols, method)
                    log_operation(f"Outlier capping ({method}) on {cols}")
                    st.rerun()
            elif action == "Outlier Removal":
                method = st.selectbox("Method", ["zscore", "iqr"])
                threshold = st.slider("Threshold", 1.0, 5.0, 3.0, 0.1, key="or_thresh")
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.outlier_removal(work_df, cols, method, threshold)
                    log_operation(f"Outlier removal ({method}, thresh={threshold}) on {cols}")
                    st.rerun()

        # --- Distribution Conditioning ---
        elif category == "Distribution Conditioning":
            action = st.selectbox("Action", [
                "Distribution Profiling", "Log Transformation",
                "Square Root Transformation", "Box-Cox Transformation",
                "Yeo-Johnson Transformation", "Quantile Transformation",
            ])
            cols = st.multiselect("Columns", numeric_cols, key="dist_cols")
            if action == "Distribution Profiling":
                if st.button("Run") and cols:
                    st.dataframe(proc.distribution_profiling(work_df, cols))
            elif action == "Log Transformation":
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.log_transformation(work_df, cols)
                    log_operation(f"Log transformation on {cols}")
                    st.rerun()
            elif action == "Square Root Transformation":
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.sqrt_transformation(work_df, cols)
                    log_operation(f"Sqrt transformation on {cols}")
                    st.rerun()
            elif action == "Box-Cox Transformation":
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.boxcox_transformation(work_df, cols)
                    log_operation(f"Box-Cox transformation on {cols}")
                    st.rerun()
            elif action == "Yeo-Johnson Transformation":
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.yeojohnson_transformation(work_df, cols)
                    log_operation(f"Yeo-Johnson transformation on {cols}")
                    st.rerun()
            elif action == "Quantile Transformation":
                dist = st.selectbox("Target distribution", ["normal", "uniform"])
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.quantile_transformation(work_df, cols, dist)
                    log_operation(f"Quantile transformation ({dist}) on {cols}")
                    st.rerun()

        # --- Feature Scaling ---
        elif category == "Feature Scaling":
            action = st.selectbox("Action", [
                "Standard Scaling", "Min-Max Scaling", "Robust Scaling",
                "MaxAbs Scaling", "Unit Vector Normalization",
            ])
            cols = st.multiselect("Columns", numeric_cols, key="scale_cols")
            if action == "Standard Scaling":
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.standard_scaling(work_df, cols)
                    log_operation(f"Standard scaling on {cols}")
                    st.rerun()
            elif action == "Min-Max Scaling":
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.minmax_scaling(work_df, cols)
                    log_operation(f"Min-Max scaling on {cols}")
                    st.rerun()
            elif action == "Robust Scaling":
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.robust_scaling(work_df, cols)
                    log_operation(f"Robust scaling on {cols}")
                    st.rerun()
            elif action == "MaxAbs Scaling":
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.maxabs_scaling(work_df, cols)
                    log_operation(f"MaxAbs scaling on {cols}")
                    st.rerun()
            elif action == "Unit Vector Normalization":
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.unit_vector_normalization(work_df, cols)
                    log_operation(f"Unit vector normalization on {cols}")
                    st.rerun()

        # --- Categorical Processing ---
        elif category == "Categorical Processing":
            action = st.selectbox("Action", [
                "Category Frequency Analysis", "Rare Category Detection",
                "Category Consolidation", "Label Encoding", "One-Hot Encoding",
            ])
            cols = st.multiselect("Columns", cat_cols, key="cat_cols")
            if action == "Category Frequency Analysis":
                if st.button("Run") and cols:
                    for col, freq in proc.category_frequency_analysis(work_df, cols).items():
                        st.markdown(f"**{col}**")
                        st.dataframe(freq)
            elif action == "Rare Category Detection":
                threshold = st.slider("Threshold (%)", 0.1, 10.0, 1.0, 0.1) / 100
                if st.button("Run") and cols:
                    rare = proc.rare_category_detection(work_df, cols, threshold)
                    for col, cats in rare.items():
                        st.write(f"**{col}**: {cats}")
            elif action == "Category Consolidation":
                threshold = st.slider("Threshold (%)", 0.1, 10.0, 1.0, 0.1, key="cc_thresh") / 100
                replacement = st.text_input("Replacement label", "Other")
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.category_consolidation(work_df, cols, threshold, replacement)
                    log_operation(f"Category consolidation on {cols}")
                    st.rerun()
            elif action == "Label Encoding":
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.label_encoding(work_df, cols)
                    log_operation(f"Label encoding on {cols}")
                    st.rerun()
            elif action == "One-Hot Encoding":
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.onehot_encoding(work_df, cols)
                    log_operation(f"One-Hot encoding on {cols}")
                    st.rerun()

        # --- Text Processing ---
        elif category == "Text Processing":
            action = st.selectbox("Action", [
                "Lowercasing", "Whitespace Cleaning", "Punctuation Removal",
            ])
            cols = st.multiselect("Columns", cat_cols, key="text_cols")
            if action == "Lowercasing":
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.text_lowercase(work_df, cols)
                    log_operation(f"Lowercasing on {cols}")
                    st.rerun()
            elif action == "Whitespace Cleaning":
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.text_whitespace_clean(work_df, cols)
                    log_operation(f"Whitespace cleaning on {cols}")
                    st.rerun()
            elif action == "Punctuation Removal":
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.text_punctuation_removal(work_df, cols)
                    log_operation(f"Punctuation removal on {cols}")
                    st.rerun()

        # --- Datetime Processing ---
        elif category == "Datetime Processing":
            action = st.selectbox("Action", [
                "Datetime Parsing", "Extract Components",
            ])
            cols = st.multiselect("Columns", all_cols, key="dt_cols")
            if action == "Datetime Parsing":
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.datetime_parsing(work_df, cols)
                    log_operation(f"Datetime parsing on {cols}")
                    st.rerun()
            elif action == "Extract Components":
                col = st.selectbox("Column", cols if cols else all_cols, key="dt_extract_col")
                if st.button("Apply") and col:
                    st.session_state.processed_df = proc.datetime_extract_components(work_df, col)
                    log_operation(f"Extract datetime components from {col}")
                    st.rerun()

        # --- Feature Engineering ---
        elif category == "Feature Engineering":
            action = st.selectbox("Action", [
                "Ratio Feature", "Difference Feature", "Interaction Features",
                "Polynomial Features", "Binning", "Rank Features", "Percentile Features",
            ])
            if action in ("Ratio Feature", "Difference Feature"):
                c1, c2 = st.columns(2)
                col1 = c1.selectbox("Column 1", numeric_cols, key="fe_c1")
                col2 = c2.selectbox("Column 2", numeric_cols, key="fe_c2")
                if st.button("Apply"):
                    if action == "Ratio Feature":
                        st.session_state.processed_df = proc.ratio_features(work_df, col1, col2)
                        log_operation(f"Ratio feature: {col1}/{col2}")
                    else:
                        st.session_state.processed_df = proc.difference_features(work_df, col1, col2)
                        log_operation(f"Difference feature: {col1}-{col2}")
                    st.rerun()
            elif action == "Interaction Features":
                cols = st.multiselect("Columns", numeric_cols, key="interact_cols")
                if st.button("Apply") and len(cols) >= 2:
                    st.session_state.processed_df = proc.interaction_features(work_df, cols)
                    log_operation(f"Interaction features on {cols}")
                    st.rerun()
            elif action == "Polynomial Features":
                cols = st.multiselect("Columns", numeric_cols, key="poly_cols")
                degree = st.number_input("Degree", 2, 4, 2)
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.polynomial_features(work_df, cols, degree)
                    log_operation(f"Polynomial features (degree={degree}) on {cols}")
                    st.rerun()
            elif action == "Binning":
                col = st.selectbox("Column", numeric_cols, key="bin_col")
                n_bins = st.number_input("Bins", 2, 50, 5)
                strategy = st.selectbox("Strategy", ["equal_width", "quantile"])
                if st.button("Apply"):
                    st.session_state.processed_df = proc.binning(work_df, col, n_bins, strategy)
                    log_operation(f"Binning ({strategy}, {n_bins} bins) on {col}")
                    st.rerun()
            elif action == "Rank Features":
                cols = st.multiselect("Columns", numeric_cols, key="rank_cols")
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.rank_features(work_df, cols)
                    log_operation(f"Rank features on {cols}")
                    st.rerun()
            elif action == "Percentile Features":
                cols = st.multiselect("Columns", numeric_cols, key="pctl_cols")
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.percentile_features(work_df, cols)
                    log_operation(f"Percentile features on {cols}")
                    st.rerun()

        # --- Time-Series Features ---
        elif category == "Time-Series Features":
            action = st.selectbox("Action", [
                "Lag Features", "Lead Features", "Rolling Mean", "Rolling Median",
                "Rolling Std", "Exponential Moving Average",
            ])
            col = st.selectbox("Column", numeric_cols, key="ts_col")
            if action == "Lag Features":
                lags = st.text_input("Lags (comma-separated)", "1,2,3", key="ts_lags")
                if st.button("Apply"):
                    lag_list = [int(x.strip()) for x in lags.split(",")]
                    st.session_state.processed_df = proc.lag_features(work_df, col, lag_list)
                    log_operation(f"Lag features ({lag_list}) on {col}")
                    st.rerun()
            elif action == "Lead Features":
                leads = st.text_input("Leads (comma-separated)", "1,2,3", key="ts_leads")
                if st.button("Apply"):
                    lead_list = [int(x.strip()) for x in leads.split(",")]
                    st.session_state.processed_df = proc.lead_features(work_df, col, lead_list)
                    log_operation(f"Lead features ({lead_list}) on {col}")
                    st.rerun()
            elif action == "Rolling Mean":
                window = st.number_input("Window", 2, 100, 3, key="ts_win")
                if st.button("Apply"):
                    st.session_state.processed_df = proc.rolling_features(work_df, col, window, "mean")
                    log_operation(f"Rolling mean (window={window}) on {col}")
                    st.rerun()
            elif action == "Rolling Median":
                window = st.number_input("Window", 2, 100, 3, key="ts_win_med")
                if st.button("Apply"):
                    st.session_state.processed_df = proc.rolling_features(work_df, col, window, "median")
                    log_operation(f"Rolling median (window={window}) on {col}")
                    st.rerun()
            elif action == "Rolling Std":
                window = st.number_input("Window", 2, 100, 3, key="ts_win2")
                if st.button("Apply"):
                    st.session_state.processed_df = proc.rolling_features(work_df, col, window, "std")
                    log_operation(f"Rolling std (window={window}) on {col}")
                    st.rerun()
            elif action == "Exponential Moving Average":
                span = st.number_input("Span", 2, 100, 5, key="ts_ewm")
                if st.button("Apply"):
                    st.session_state.processed_df = proc.ewm_features(work_df, col, span)
                    log_operation(f"EWM (span={span}) on {col}")
                    st.rerun()

        # --- Statistical Analysis ---
        elif category == "Statistical Analysis":
            action = st.selectbox("Action", [
                "Summary Statistics", "Correlation Analysis", "Covariance Analysis",
                "Hypothesis Test (t-test)", "ANOVA", "Chi-Square Test",
            ])
            if action in ("Summary Statistics", "Correlation Analysis", "Covariance Analysis"):
                cols = st.multiselect("Columns", numeric_cols, default=numeric_cols[:5] if numeric_cols else [], key="stat_cols")
                if action == "Summary Statistics":
                    if st.button("Run") and cols:
                        st.dataframe(proc.summary_statistics(work_df, cols))
                elif action == "Correlation Analysis":
                    if st.button("Run") and cols:
                        corr = proc.correlation_analysis(work_df, cols)
                        st.dataframe(corr)
                        fig = px.imshow(corr, text_auto=".2f", aspect="auto")
                        st.plotly_chart(fig, use_container_width=True)
                elif action == "Covariance Analysis":
                    if st.button("Run") and cols:
                        st.dataframe(proc.covariance_analysis(work_df, cols))
            elif action == "Hypothesis Test (t-test)":
                col = st.selectbox("Numeric column", numeric_cols, key="ttest_col")
                group_col = st.selectbox("Group column", cat_cols, key="ttest_grp")
                if st.button("Run"):
                    result = proc.hypothesis_test_ttest(work_df, col, group_col)
                    if result:
                        st.json(result)
                    else:
                        st.warning("Need at least 2 groups.")
            elif action == "ANOVA":
                col = st.selectbox("Numeric column", numeric_cols, key="anova_col")
                group_col = st.selectbox("Group column", cat_cols, key="anova_grp")
                if st.button("Run"):
                    result = proc.anova_test(work_df, col, group_col)
                    st.json(result)
            elif action == "Chi-Square Test":
                c1, c2 = st.columns(2)
                col1 = c1.selectbox("Column 1", cat_cols, key="chi_c1")
                col2 = c2.selectbox("Column 2", cat_cols, key="chi_c2")
                if st.button("Run"):
                    result = proc.chi_square_test(work_df, col1, col2)
                    st.json(result)

        # --- Feature Selection ---
        elif category == "Feature Selection":
            action = st.selectbox("Action", [
                "Variance Threshold", "Correlation Filtering",
                "Mutual Information", "Tree Importance", "RFE", "LASSO",
            ])
            cols = st.multiselect("Columns", numeric_cols, default=numeric_cols, key="fs_cols")
            if action == "Variance Threshold":
                threshold = st.number_input("Min variance", 0.0, 100.0, 0.0, 0.01)
                if st.button("Run") and cols:
                    kept = proc.variance_threshold_selection(work_df, cols, threshold)
                    st.write(f"Kept {len(kept)}/{len(cols)} columns: {kept}")
            elif action == "Correlation Filtering":
                threshold = st.slider("Correlation threshold", 0.5, 1.0, 0.95, 0.01)
                if st.button("Run") and cols:
                    to_drop = proc.correlation_filtering(work_df, cols, threshold)
                    st.write(f"Suggested to drop ({len(to_drop)}): {to_drop}")
            elif action == "Mutual Information":
                target_col = st.selectbox("Target column", all_cols, key="mi_target")
                task = st.selectbox("Task", ["classification", "regression"], key="mi_task")
                k = st.number_input("Top K features", 1, len(cols), min(5, len(cols)), key="mi_k")
                if st.button("Run") and cols:
                    selected, scores = proc.mutual_information_selection(work_df, cols, target_col, task, k)
                    st.write(f"Top {k} features: {selected}")
                    st.dataframe(scores.to_frame("MI Score"))
            elif action == "Tree Importance":
                target_col = st.selectbox("Target column", all_cols, key="tree_target")
                k = st.number_input("Top K features", 1, len(cols), min(5, len(cols)), key="tree_k")
                if st.button("Run") and cols:
                    selected, importances = proc.tree_importance_selection(work_df, cols, target_col, k)
                    st.write(f"Top {k} features: {selected}")
                    st.dataframe(importances.to_frame("Importance"))
            elif action == "RFE":
                target_col = st.selectbox("Target column", all_cols, key="rfe_target")
                n_features = st.number_input("N features to select", 1, len(cols), min(5, len(cols)), key="rfe_n")
                if st.button("Run") and cols:
                    selected = proc.rfe_selection(work_df, cols, target_col, n_features)
                    st.write(f"Selected features: {selected}")
            elif action == "LASSO":
                target_col = st.selectbox("Target column", all_cols, key="lasso_target")
                alpha = st.number_input("Alpha", 0.001, 1.0, 0.01, 0.001, key="lasso_alpha")
                if st.button("Run") and cols:
                    selected, coefs = proc.lasso_selection(work_df, cols, target_col, alpha)
                    st.write(f"Non-zero features: {selected}")
                    st.dataframe(coefs.to_frame("Coefficient"))

        # --- Dimensionality Reduction ---
        elif category == "Dimensionality Reduction":
            action = st.selectbox("Action", [
                "PCA", "Kernel PCA", "ICA", "Factor Analysis", "t-SNE",
            ])
            cols = st.multiselect("Columns", numeric_cols, key="dr_cols")
            n_components = st.number_input("Components", 1, 10, 2, key="dr_n")
            if action == "PCA":
                if st.button("Apply") and cols:
                    result_df, explained = proc.pca_reduction(work_df, cols, n_components)
                    st.session_state.processed_df = result_df
                    log_operation(f"PCA ({n_components} components) on {cols}")
                    st.rerun()
            elif action == "Kernel PCA":
                kernel = st.selectbox("Kernel", ["rbf", "poly", "sigmoid", "cosine"])
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.kernel_pca_reduction(work_df, cols, n_components, kernel)
                    log_operation(f"Kernel PCA ({kernel}, {n_components} components) on {cols}")
                    st.rerun()
            elif action == "ICA":
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.ica_reduction(work_df, cols, n_components)
                    log_operation(f"ICA ({n_components} components) on {cols}")
                    st.rerun()
            elif action == "Factor Analysis":
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.factor_analysis_reduction(work_df, cols, n_components)
                    log_operation(f"Factor Analysis ({n_components} components) on {cols}")
                    st.rerun()
            elif action == "t-SNE":
                perplexity = st.slider("Perplexity", 5, 50, 30, key="tsne_perp")
                if st.button("Apply") and cols:
                    st.session_state.processed_df = proc.tsne_reduction(work_df, cols, n_components, perplexity)
                    log_operation(f"t-SNE ({n_components} components, perp={perplexity}) on {cols}")
                    st.rerun()

        # --- Target Conditioning ---
        elif category == "Target Conditioning":
            action = st.selectbox("Action", [
                "Class Distribution Analysis", "Oversampling (Random)",
                "Oversampling (SMOTE)", "Oversampling (ADASYN)", "Undersampling",
            ])
            col = st.selectbox("Target column", all_cols, key="target_col")
            if action == "Class Distribution Analysis":
                if st.button("Run"):
                    st.dataframe(proc.class_distribution_analysis(work_df, col))
            else:
                feature_cols = st.multiselect("Feature columns", numeric_cols, default=numeric_cols, key="tc_feat")
                if st.button("Apply") and feature_cols:
                    if action == "Oversampling (Random)":
                        st.session_state.processed_df = proc.oversampling(work_df, feature_cols, col, "random")
                        log_operation(f"Random oversampling (target={col})")
                    elif action == "Oversampling (SMOTE)":
                        st.session_state.processed_df = proc.oversampling(work_df, feature_cols, col, "smote")
                        log_operation(f"SMOTE oversampling (target={col})")
                    elif action == "Oversampling (ADASYN)":
                        st.session_state.processed_df = proc.oversampling(work_df, feature_cols, col, "adasyn")
                        log_operation(f"ADASYN oversampling (target={col})")
                    elif action == "Undersampling":
                        st.session_state.processed_df = proc.undersampling(work_df, feature_cols, col)
                        log_operation(f"Undersampling (target={col})")
                    st.rerun()

        # --- Footer: Show processed data & download ---
        st.divider()
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("🔄 Reset to Original"):
                st.session_state.processed_df = df.copy()
                st.session_state.operation_history = []
                st.rerun()
        with col_b:
            st.download_button(
                "⬇️ Download Processed CSV",
                work_df.to_csv(index=False).encode(),
                "processed_data.csv",
                "text/csv",
            )
        with col_c:
            st.write(f"Shape: {work_df.shape}")

        # Operation History
        if st.session_state.operation_history:
            st.markdown("**🕐 Operation History**")
            for i, op in enumerate(st.session_state.operation_history, 1):
                st.text(f"{i}. {op}")

        st.markdown("**Processed Data Preview**")
        st.dataframe(work_df.head(50), use_container_width=True)

    # --- Before/After Compare Tab ---
    with tab_compare:
        work_df = st.session_state.processed_df
        st.markdown("### Before vs After Processing")

        # Quality score comparison
        score_orig, bd_orig = compute_quality_score(df)
        score_proc, bd_proc = compute_quality_score(work_df)
        cmp1, cmp2 = st.columns(2)
        with cmp1:
            st.metric("Original Quality Score", f"{score_orig}/100")
        with cmp2:
            delta = score_proc - score_orig
            st.metric("Processed Quality Score", f"{score_proc}/100", delta=f"{delta:+.1f}")

        # Shape comparison
        st.markdown(f"**Original:** {df.shape[0]} rows × {df.shape[1]} cols → **Processed:** {work_df.shape[0]} rows × {work_df.shape[1]} cols")

        # Distribution comparison for numeric columns
        orig_numeric = df.select_dtypes(include="number").columns.tolist()
        proc_numeric = work_df.select_dtypes(include="number").columns.tolist()
        common_numeric = [c for c in orig_numeric if c in proc_numeric]

        if common_numeric:
            compare_col = st.selectbox("Compare column distribution", common_numeric, key="cmp_col")
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Original", "Processed"))
            fig.add_trace(
                go.Histogram(x=df[compare_col].dropna(), name="Original", marker_color="#636EFA"),
                row=1, col=1,
            )
            fig.add_trace(
                go.Histogram(x=work_df[compare_col].dropna(), name="Processed", marker_color="#00CC96"),
                row=1, col=2,
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Stats comparison table
            stats_orig = df[compare_col].describe()
            stats_proc = work_df[compare_col].describe()
            compare_df = pd.DataFrame({"Original": stats_orig, "Processed": stats_proc})
            compare_df["Change"] = compare_df["Processed"] - compare_df["Original"]
            st.dataframe(compare_df, use_container_width=True)

        # Missing values comparison
        st.markdown("**Missing Values Comparison**")
        common_cols = [c for c in df.columns if c in work_df.columns]
        if common_cols:
            missing_orig = df[common_cols].isnull().sum()
            missing_proc = work_df[common_cols].isnull().sum()
            missing_cmp = pd.DataFrame({"Original": missing_orig, "Processed": missing_proc})
            missing_cmp["Resolved"] = missing_cmp["Original"] - missing_cmp["Processed"]
            missing_cmp = missing_cmp[missing_cmp["Original"] > 0]
            if not missing_cmp.empty:
                st.dataframe(missing_cmp, use_container_width=True)
            else:
                st.write("No missing values in common columns.")
