import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    PowerTransformer, QuantileTransformer, Normalizer,
    PolynomialFeatures,
)
from sklearn.decomposition import PCA, FastICA, KernelPCA, FactorAnalysis
from sklearn.manifold import TSNE
from sklearn.feature_selection import (
    VarianceThreshold, mutual_info_classif, mutual_info_regression, RFE,
)
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import Lasso
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


# --- Data Quality ---

def missing_value_detection(df, columns):
    report = df[columns].isnull().sum().to_frame("missing_count")
    report["missing_pct"] = (report["missing_count"] / len(df) * 100).round(2)
    return report


def missing_value_imputation(df, columns, strategy="mean"):
    df = df.copy()
    for col in columns:
        if strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else np.nan
            df[col] = df[col].fillna(mode_val)
        elif strategy == "ffill":
            df[col] = df[col].ffill()
        elif strategy == "bfill":
            df[col] = df[col].bfill()
        elif strategy == "zero":
            df[col] = df[col].fillna(0)
        elif strategy == "drop_rows":
            df = df.dropna(subset=[col])
    return df


def duplicate_detection(df, subset=None):
    dupes = df.duplicated(subset=subset, keep=False)
    return df[dupes]


def duplicate_removal(df, keep="first"):
    return df.drop_duplicates(keep=keep)


# --- Numeric Cleaning ---

def numeric_conversion(df, columns):
    df = df.copy()
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def precision_standardization(df, columns, decimals=2):
    df = df.copy()
    for col in columns:
        df[col] = df[col].round(decimals)
    return df


# --- Outlier Processing ---

def zscore_outlier_detection(df, columns, threshold=3.0):
    mask = pd.DataFrame(False, index=df.index, columns=columns)
    for col in columns:
        z = np.abs(stats.zscore(df[col].dropna()))
        mask.loc[df[col].notna(), col] = z > threshold
    return mask


def iqr_outlier_detection(df, columns, factor=1.5):
    mask = pd.DataFrame(False, index=df.index, columns=columns)
    for col in columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        mask[col] = (df[col] < q1 - factor * iqr) | (df[col] > q3 + factor * iqr)
    return mask


def isolation_forest_detection(df, columns, contamination=0.1):
    data = df[columns].dropna()
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(data)
    mask = pd.Series(False, index=df.index)
    mask.loc[data.index] = preds == -1
    return mask


def local_outlier_factor(df, columns, n_neighbors=20, contamination=0.1):
    data = df[columns].dropna()
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    preds = lof.fit_predict(data)
    mask = pd.Series(False, index=df.index)
    mask.loc[data.index] = preds == -1
    return mask


def outlier_capping(df, columns, method="iqr", factor=1.5):
    df = df.copy()
    for col in columns:
        if method == "iqr":
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - factor * iqr, q3 + factor * iqr
        else:  # percentile
            lower, upper = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)
    return df


def outlier_removal(df, columns, method="zscore", threshold=3.0):
    if method == "zscore":
        mask = zscore_outlier_detection(df, columns, threshold)
    else:
        mask = iqr_outlier_detection(df, columns, threshold)
    return df[~mask.any(axis=1)]


# --- Distribution Conditioning ---

def distribution_profiling(df, columns):
    results = {}
    for col in columns:
        data = df[col].dropna()
        results[col] = {
            "skewness": data.skew(),
            "kurtosis": data.kurtosis(),
            "shapiro_p": stats.shapiro(data.sample(min(5000, len(data))))[1] if len(data) > 3 else None,
        }
    return pd.DataFrame(results).T


def log_transformation(df, columns):
    df = df.copy()
    for col in columns:
        df[col] = np.log1p(df[col].clip(lower=0))
    return df


def sqrt_transformation(df, columns):
    df = df.copy()
    for col in columns:
        df[col] = np.sqrt(df[col].clip(lower=0))
    return df


def boxcox_transformation(df, columns):
    df = df.copy()
    for col in columns:
        data = df[col].dropna()
        if (data > 0).all():
            transformed, _ = stats.boxcox(data)
            df.loc[data.index, col] = transformed
    return df


def yeojohnson_transformation(df, columns):
    df = df.copy()
    pt = PowerTransformer(method="yeo-johnson")
    df[columns] = pt.fit_transform(df[columns])
    return df


def quantile_transformation(df, columns, output_distribution="normal"):
    df = df.copy()
    qt = QuantileTransformer(output_distribution=output_distribution, random_state=42)
    df[columns] = qt.fit_transform(df[columns])
    return df


# --- Feature Scaling ---

def standard_scaling(df, columns):
    df = df.copy()
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def minmax_scaling(df, columns, feature_range=(0, 1)):
    df = df.copy()
    scaler = MinMaxScaler(feature_range=feature_range)
    df[columns] = scaler.fit_transform(df[columns])
    return df


def robust_scaling(df, columns):
    df = df.copy()
    scaler = RobustScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def maxabs_scaling(df, columns):
    df = df.copy()
    scaler = MaxAbsScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def unit_vector_normalization(df, columns):
    df = df.copy()
    scaler = Normalizer()
    mask = df[columns].notna().all(axis=1)
    df.loc[mask, columns] = scaler.fit_transform(df.loc[mask, columns])
    return df


# --- Categorical Processing ---

def category_frequency_analysis(df, columns):
    results = {}
    for col in columns:
        results[col] = df[col].value_counts()
    return results


def rare_category_detection(df, columns, threshold=0.01):
    results = {}
    for col in columns:
        freq = df[col].value_counts(normalize=True)
        results[col] = freq[freq < threshold].index.tolist()
    return results


def category_consolidation(df, columns, threshold=0.01, replacement="Other"):
    df = df.copy()
    for col in columns:
        freq = df[col].value_counts(normalize=True)
        rare = freq[freq < threshold].index
        df[col] = df[col].replace(rare, replacement)
    return df


def label_encoding(df, columns):
    df = df.copy()
    for col in columns:
        df[col] = df[col].astype("category").cat.codes
    return df


def onehot_encoding(df, columns):
    return pd.get_dummies(df, columns=columns, prefix=columns)


# --- Text Processing ---

def text_lowercase(df, columns):
    df = df.copy()
    for col in columns:
        df[col] = df[col].astype(str).str.lower()
    return df


def text_whitespace_clean(df, columns):
    df = df.copy()
    for col in columns:
        df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    return df


def text_punctuation_removal(df, columns):
    import string
    df = df.copy()
    for col in columns:
        df[col] = df[col].astype(str).str.replace(f"[{string.punctuation}]", "", regex=True)
    return df


# --- Datetime Processing ---

def datetime_parsing(df, columns):
    df = df.copy()
    for col in columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def datetime_extract_components(df, column):
    df = df.copy()
    dt = pd.to_datetime(df[column], errors="coerce")
    df[f"{column}_year"] = dt.dt.year
    df[f"{column}_month"] = dt.dt.month
    df[f"{column}_day"] = dt.dt.day
    df[f"{column}_weekday"] = dt.dt.weekday
    return df


# --- Feature Engineering ---

def binning(df, column, n_bins=5, strategy="equal_width"):
    df = df.copy()
    if strategy == "equal_width":
        df[f"{column}_binned"] = pd.cut(df[column], bins=n_bins, labels=False)
    else:
        df[f"{column}_binned"] = pd.qcut(df[column], q=n_bins, labels=False, duplicates="drop")
    return df


def rank_features(df, columns):
    df = df.copy()
    for col in columns:
        df[f"{col}_rank"] = df[col].rank()
    return df


def polynomial_features(df, columns, degree=2):
    df = df.copy()
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    data = df[columns].fillna(0)
    transformed = poly.fit_transform(data)
    names = poly.get_feature_names_out(columns)
    poly_df = pd.DataFrame(transformed, columns=names, index=df.index)
    # Only add new columns (skip original ones)
    new_cols = [c for c in poly_df.columns if c not in columns]
    for c in new_cols:
        df[c] = poly_df[c]
    return df


def interaction_features(df, columns):
    df = df.copy()
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            c1, c2 = columns[i], columns[j]
            df[f"{c1}_x_{c2}"] = df[c1] * df[c2]
    return df


def percentile_features(df, columns):
    df = df.copy()
    for col in columns:
        df[f"{col}_percentile"] = df[col].rank(pct=True)
    return df


def ratio_features(df, col1, col2):
    df = df.copy()
    df[f"{col1}_div_{col2}"] = df[col1] / df[col2].replace(0, np.nan)
    return df


def difference_features(df, col1, col2):
    df = df.copy()
    df[f"{col1}_minus_{col2}"] = df[col1] - df[col2]
    return df


# --- Time-Series Features ---

def lag_features(df, column, lags):
    df = df.copy()
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df


def rolling_features(df, column, window, agg="mean"):
    df = df.copy()
    roller = df[column].rolling(window)
    if agg == "mean":
        df[f"{column}_rolling_mean_{window}"] = roller.mean()
    elif agg == "median":
        df[f"{column}_rolling_median_{window}"] = roller.median()
    elif agg == "std":
        df[f"{column}_rolling_std_{window}"] = roller.std()
    return df


def lead_features(df, column, leads):
    df = df.copy()
    for lead in leads:
        df[f"{column}_lead_{lead}"] = df[column].shift(-lead)
    return df


def ewm_features(df, column, span):
    df = df.copy()
    df[f"{column}_ewm_{span}"] = df[column].ewm(span=span).mean()
    return df


# --- Statistical Analysis ---

def summary_statistics(df, columns):
    return df[columns].describe()


def correlation_analysis(df, columns):
    return df[columns].corr()


def covariance_analysis(df, columns):
    return df[columns].cov()


def hypothesis_test_ttest(df, column, group_col):
    groups = df[group_col].unique()
    if len(groups) < 2:
        return None
    g1 = df[df[group_col] == groups[0]][column].dropna()
    g2 = df[df[group_col] == groups[1]][column].dropna()
    stat, p = stats.ttest_ind(g1, g2)
    return {"statistic": stat, "p_value": p, "group_1": groups[0], "group_2": groups[1]}


def anova_test(df, column, group_col):
    groups = [g[column].dropna().values for _, g in df.groupby(group_col)]
    stat, p = stats.f_oneway(*groups)
    return {"f_statistic": stat, "p_value": p}


def chi_square_test(df, col1, col2):
    contingency = pd.crosstab(df[col1], df[col2])
    stat, p, dof, expected = stats.chi2_contingency(contingency)
    return {"chi2_statistic": stat, "p_value": p, "dof": dof}


# --- Feature Selection ---

def variance_threshold_selection(df, columns, threshold=0.0):
    vt = VarianceThreshold(threshold=threshold)
    vt.fit(df[columns])
    return [columns[i] for i, keep in enumerate(vt.get_support()) if keep]


def correlation_filtering(df, columns, threshold=0.95):
    corr = df[columns].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    return to_drop


def mutual_information_selection(df, columns, target_col, task="classification", k=5):
    X = df[columns].fillna(0)
    y = df[target_col]
    if task == "classification":
        mi = mutual_info_classif(X, y, random_state=42)
    else:
        mi = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi, index=columns).sort_values(ascending=False)
    return mi_series.head(k).index.tolist(), mi_series


def rfe_selection(df, columns, target_col, n_features=5):
    X = df[columns].fillna(0)
    y = df[target_col]
    estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    selector = RFE(estimator, n_features_to_select=n_features)
    selector.fit(X, y)
    return [columns[i] for i, keep in enumerate(selector.support_) if keep]


def lasso_selection(df, columns, target_col, alpha=0.01):
    X = df[columns].fillna(0)
    y = df[target_col]
    lasso = Lasso(alpha=alpha, random_state=42)
    lasso.fit(X, y)
    coefs = pd.Series(np.abs(lasso.coef_), index=columns)
    return coefs[coefs > 0].index.tolist(), coefs.sort_values(ascending=False)


def tree_importance_selection(df, columns, target_col, k=5):
    X = df[columns].fillna(0)
    y = df[target_col]
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=columns).sort_values(ascending=False)
    return importances.head(k).index.tolist(), importances


# --- Dimensionality Reduction ---

def pca_reduction(df, columns, n_components=2):
    data = df[columns].fillna(0)
    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(data)
    result_df = df.copy()
    for i in range(n_components):
        result_df[f"PC{i+1}"] = transformed[:, i]
    explained = pca.explained_variance_ratio_
    return result_df, explained


def kernel_pca_reduction(df, columns, n_components=2, kernel="rbf"):
    data = df[columns].fillna(0)
    kpca = KernelPCA(n_components=n_components, kernel=kernel, random_state=42)
    transformed = kpca.fit_transform(data)
    result_df = df.copy()
    for i in range(n_components):
        result_df[f"KPC{i+1}"] = transformed[:, i]
    return result_df


def ica_reduction(df, columns, n_components=2):
    data = df[columns].fillna(0)
    ica = FastICA(n_components=n_components, random_state=42)
    transformed = ica.fit_transform(data)
    result_df = df.copy()
    for i in range(n_components):
        result_df[f"IC{i+1}"] = transformed[:, i]
    return result_df


def factor_analysis_reduction(df, columns, n_components=2):
    data = df[columns].fillna(0)
    fa = FactorAnalysis(n_components=n_components, random_state=42)
    transformed = fa.fit_transform(data)
    result_df = df.copy()
    for i in range(n_components):
        result_df[f"FA{i+1}"] = transformed[:, i]
    return result_df


def tsne_reduction(df, columns, n_components=2, perplexity=30):
    data = df[columns].fillna(0)
    perp = min(perplexity, len(data) - 1)
    tsne = TSNE(n_components=n_components, perplexity=perp, random_state=42)
    transformed = tsne.fit_transform(data)
    result_df = df.copy()
    for i in range(n_components):
        result_df[f"tSNE{i+1}"] = transformed[:, i]
    return result_df


# --- Target Conditioning ---

def class_distribution_analysis(df, column):
    dist = df[column].value_counts()
    dist_pct = df[column].value_counts(normalize=True) * 100
    return pd.DataFrame({"count": dist, "percentage": dist_pct.round(2)})


def oversampling(df, feature_columns, target_col, strategy="random"):
    X = df[feature_columns].fillna(0)
    y = df[target_col]
    if strategy == "random":
        sampler = RandomOverSampler(random_state=42)
    elif strategy == "smote":
        min_class_count = y.value_counts().min()
        k = min(5, min_class_count - 1) if min_class_count > 1 else 1
        sampler = SMOTE(random_state=42, k_neighbors=k)
    elif strategy == "adasyn":
        min_class_count = y.value_counts().min()
        k = min(5, min_class_count - 1) if min_class_count > 1 else 1
        sampler = ADASYN(random_state=42, n_neighbors=k)
    else:
        sampler = RandomOverSampler(random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)
    result = pd.DataFrame(X_res, columns=feature_columns)
    result[target_col] = y_res
    return result


def undersampling(df, feature_columns, target_col):
    X = df[feature_columns].fillna(0)
    y = df[target_col]
    sampler = RandomUnderSampler(random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)
    result = pd.DataFrame(X_res, columns=feature_columns)
    result[target_col] = y_res
    return result
