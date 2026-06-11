import pandas as pd
import numpy as np
import processing as proc

df = pd.read_csv("sample_data.csv")
passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name}")
        failed += 1


print("=" * 60)
print("DATA PROCESSING TEST SUITE")
print("=" * 60)
print(f"\nLoaded sample_data.csv: {df.shape[0]} rows × {df.shape[1]} columns\n")

# --- Data Quality ---
print("📋 Data Quality")
report = proc.missing_value_detection(df, df.columns.tolist())
check("Missing Value Detection", report["missing_count"].sum() > 0)

imputed = proc.missing_value_imputation(df, ["age"], "mean")
check("Missing Value Imputation (mean)", imputed["age"].isnull().sum() == 0)

imputed2 = proc.missing_value_imputation(df, ["city"], "mode")
check("Missing Value Imputation (mode)", imputed2["city"].isnull().sum() == 0)

dupes = proc.duplicate_detection(df, subset=["name", "age", "salary"])
check("Duplicate Detection", len(dupes) > 0)

deduped = proc.duplicate_removal(df.drop(columns=["id"]), keep="first")
check("Duplicate Removal", len(deduped) <= len(df))

# --- Numeric Cleaning ---
print("\n🔢 Numeric Cleaning")
df_text = df.copy()
df_text["age"] = df_text["age"].astype(str)
converted = proc.numeric_conversion(df_text, ["age"])
check("Numeric Conversion", converted["age"].dtype in [np.float64, np.int64])

precise = proc.precision_standardization(df, ["salary"], 0)
check("Precision Standardization", all(precise["salary"].dropna() == precise["salary"].dropna().astype(int)))

# --- Outlier Processing ---
print("\n📊 Outlier Processing")
z_mask = proc.zscore_outlier_detection(df, ["salary"], 2.0)
check("Z-Score Outlier Detection", z_mask.any().any())

iqr_mask = proc.iqr_outlier_detection(df, ["salary"], 1.5)
check("IQR Outlier Detection", iqr_mask.any().any())

iso_mask = proc.isolation_forest_detection(df, ["age", "salary", "score"], 0.1)
check("Isolation Forest Detection", iso_mask.sum() > 0)

lof_mask = proc.local_outlier_factor(df, ["age", "salary", "score"], 5, 0.2)
check("Local Outlier Factor", lof_mask.sum() > 0)

capped = proc.outlier_capping(df, ["salary"], "iqr")
check("Outlier Capping", capped["salary"].max() <= df["salary"].max())

removed = proc.outlier_removal(df, ["salary"], "zscore", 2.0)
check("Outlier Removal", len(removed) <= len(df))

# --- Distribution Conditioning ---
print("\n📈 Distribution Conditioning")
profile = proc.distribution_profiling(df, ["salary"])
check("Distribution Profiling", "skewness" in profile.columns)

logged = proc.log_transformation(df, ["salary"])
check("Log Transformation", logged["salary"].max() < df["salary"].max())

sqrted = proc.sqrt_transformation(df, ["salary"])
check("Square Root Transformation", sqrted["salary"].max() < df["salary"].max())

yj = proc.yeojohnson_transformation(df, ["salary"])
check("Yeo-Johnson Transformation", yj["salary"].std() != df["salary"].std())

qt = proc.quantile_transformation(df, ["salary"], "normal")
check("Quantile Transformation", abs(qt["salary"].mean()) < 1)

# --- Feature Scaling ---
print("\n⚖️ Feature Scaling")
std_scaled = proc.standard_scaling(df, ["salary"])
check("Standard Scaling", abs(std_scaled["salary"].mean()) < 0.01)

mm_scaled = proc.minmax_scaling(df, ["salary"])
check("Min-Max Scaling", mm_scaled["salary"].min() >= 0 and mm_scaled["salary"].max() <= 1)

rob_scaled = proc.robust_scaling(df, ["salary"])
check("Robust Scaling", rob_scaled is not None and len(rob_scaled) == len(df))

ma_scaled = proc.maxabs_scaling(df, ["salary"])
check("MaxAbs Scaling", ma_scaled["salary"].abs().max() <= 1.0)

uv = proc.unit_vector_normalization(df, ["age", "salary", "score"])
check("Unit Vector Normalization", uv is not None and len(uv) == len(df))

# --- Categorical Processing ---
print("\n🏷️ Categorical Processing")
freq = proc.category_frequency_analysis(df, ["department"])
check("Category Frequency Analysis", "department" in freq)

rare = proc.rare_category_detection(df, ["city"], 0.15)
check("Rare Category Detection", len(rare["city"]) > 0)

consolidated = proc.category_consolidation(df, ["city"], 0.15, "Other")
check("Category Consolidation", "Other" in consolidated["city"].values)

encoded = proc.label_encoding(df, ["department"])
check("Label Encoding", encoded["department"].dtype in [np.int8, np.int16, np.int32, np.int64])

onehot = proc.onehot_encoding(df, ["department"])
check("One-Hot Encoding", onehot.shape[1] > df.shape[1])

# --- Text Processing ---
print("\n📝 Text Processing")
lowered = proc.text_lowercase(df, ["name"])
check("Lowercasing", all(lowered["name"] == lowered["name"].str.lower()))

cleaned = proc.text_whitespace_clean(df, ["notes"])
check("Whitespace Cleaning", "  " not in cleaned["notes"].iloc[1])

no_punct = proc.text_punctuation_removal(df, ["notes"])
check("Punctuation Removal", "!" not in no_punct["notes"].iloc[2])

# --- Datetime Processing ---
print("\n📅 Datetime Processing")
parsed = proc.datetime_parsing(df, ["hire_date"])
check("Datetime Parsing", pd.api.types.is_datetime64_any_dtype(parsed["hire_date"]))

extracted = proc.datetime_extract_components(df, "hire_date")
check("Extract Components", "hire_date_year" in extracted.columns)

# --- Feature Engineering ---
print("\n🛠️ Feature Engineering")
ratio = proc.ratio_features(df, "salary", "age")
check("Ratio Feature", "salary_div_age" in ratio.columns)

diff = proc.difference_features(df, "salary", "score")
check("Difference Feature", "salary_minus_score" in diff.columns)

interact = proc.interaction_features(df, ["age", "salary", "score"])
check("Interaction Features", "age_x_salary" in interact.columns)

poly = proc.polynomial_features(df, ["age", "score"], 2)
check("Polynomial Features", poly.shape[1] > df.shape[1])

binned = proc.binning(df, "salary", 3, "equal_width")
check("Binning", "salary_binned" in binned.columns)

ranked = proc.rank_features(df, ["salary"])
check("Rank Features", "salary_rank" in ranked.columns)

pctl = proc.percentile_features(df, ["salary"])
check("Percentile Features", "salary_percentile" in pctl.columns)

# --- Time-Series Features ---
print("\n⏰ Time-Series Features")
lagged = proc.lag_features(df, "salary", [1, 2])
check("Lag Features", "salary_lag_1" in lagged.columns)

lead = proc.lead_features(df, "salary", [1, 2])
check("Lead Features", "salary_lead_1" in lead.columns)

rolled = proc.rolling_features(df, "salary", 3, "mean")
check("Rolling Mean", "salary_rolling_mean_3" in rolled.columns)

rolled_std = proc.rolling_features(df, "salary", 3, "std")
check("Rolling Std", "salary_rolling_std_3" in rolled_std.columns)

ewm = proc.ewm_features(df, "salary", 5)
check("Exponential Moving Average", "salary_ewm_5" in ewm.columns)

# --- Statistical Analysis ---
print("\n📐 Statistical Analysis")
stats_res = proc.summary_statistics(df, ["age", "salary"])
check("Summary Statistics", "mean" in stats_res.index)

corr = proc.correlation_analysis(df, ["age", "salary", "score"])
check("Correlation Analysis", corr.shape == (3, 3))

cov = proc.covariance_analysis(df, ["age", "salary"])
check("Covariance Analysis", cov.shape == (2, 2))

ttest = proc.hypothesis_test_ttest(df, "salary", "department")
check("T-Test", ttest is not None and "p_value" in ttest)

anova = proc.anova_test(df, "salary", "department")
check("ANOVA", "p_value" in anova)

chi = proc.chi_square_test(df, "department", "city")
check("Chi-Square Test", "p_value" in chi)

# --- Feature Selection ---
print("\n🎯 Feature Selection")
kept = proc.variance_threshold_selection(df, ["age", "salary", "score"], 0.0)
check("Variance Threshold", len(kept) > 0)

to_drop = proc.correlation_filtering(df, ["age", "salary", "score"], 0.95)
check("Correlation Filtering", isinstance(to_drop, list))

mi_sel, mi_scores = proc.mutual_information_selection(df, ["age", "salary", "score"], "department", "classification", 2)
check("Mutual Information Selection", len(mi_sel) == 2)

tree_sel, tree_imp = proc.tree_importance_selection(df, ["age", "salary", "score"], "department", 2)
check("Tree Importance Selection", len(tree_sel) == 2)

# --- Dimensionality Reduction ---
print("\n🔬 Dimensionality Reduction")
pca_df, explained = proc.pca_reduction(df, ["age", "salary", "score"], 2)
check("PCA", "PC1" in pca_df.columns and len(explained) == 2)

kpca_df = proc.kernel_pca_reduction(df, ["age", "salary", "score"], 2)
check("Kernel PCA", "KPC1" in kpca_df.columns)

ica_df = proc.ica_reduction(df, ["age", "salary", "score"], 2)
check("ICA", "IC1" in ica_df.columns)

fa_df = proc.factor_analysis_reduction(df, ["age", "salary", "score"], 2)
check("Factor Analysis", "FA1" in fa_df.columns)

tsne_df = proc.tsne_reduction(df, ["age", "salary", "score"], 2, 5)
check("t-SNE", "tSNE1" in tsne_df.columns)

# --- Target Conditioning ---
print("\n🎯 Target Conditioning")
dist = proc.class_distribution_analysis(df, "department")
check("Class Distribution Analysis", "count" in dist.columns)

over = proc.oversampling(df, ["age", "salary", "score"], "department", "random")
check("Random Oversampling", len(over) >= len(df))

smote_df = proc.oversampling(df, ["age", "salary", "score"], "department", "smote")
check("SMOTE", len(smote_df) >= len(df))

under = proc.undersampling(df, ["age", "salary", "score"], "department")
check("Undersampling", len(under) <= len(df))

# --- Summary ---
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
print("=" * 60)
if failed == 0:
    print("🎉 All tests passed!")
else:
    print(f"⚠️  {failed} test(s) need attention.")
