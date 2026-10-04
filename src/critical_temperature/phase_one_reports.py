"""Regenerate full-dataset exploratory tables, never model feature selection.

Records, duplicate measurements and potential outliers are counted, not removed.
The near-zero-variance heuristic flags constants, or features with <=10% unique
values and a dominant/second-most-common frequency ratio strictly greater than 19.
"""
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_inputs(root=PROJECT_ROOT, *, train=None, unique=None, processed=None):
    """Load raw paired sources and optional full-row processed analysis data."""
    root = Path(root)
    train = pd.read_csv(root / "data/raw/train.csv") if train is None else train.copy()
    unique = pd.read_csv(root / "data/raw/unique_m.csv") if unique is None else unique.copy()
    train = train.reset_index(drop=True)
    unique = unique.reset_index(drop=True)
    if len(train) == 0 or len(train) != len(unique) or not train.critical_temp.equals(unique.critical_temp):
        raise ValueError("Raw train and unique_m must align row by row and be nonempty")
    if processed is None:
        path = root / "data/processed/train_transformed.csv"
        processed = pd.read_csv(path) if path.exists() else train.copy()
    processed = processed.reset_index(drop=True).copy()
    # Phase I uses the record-preserving cleaning path, not Phase II deduplication.
    if len(processed) != len(train) or not processed.critical_temp.equals(train.critical_temp):
        raise ValueError("Processed analysis records must align with all raw records")
    processed["above_77k"] = processed.critical_temp.gt(77).astype(int)
    return train, unique, processed


def generate_reports(root=PROJECT_ROOT, *, train=None, unique=None, processed=None):
    """Write all Phase I CSVs and return filename-to-DataFrame mappings.

    Inputs can be supplied as DataFrames; otherwise raw and optional transformed
    CSVs are loaded relative to root. Source files are never written.
    """
    root = Path(root)
    train, unique, df = load_inputs(root, train=train, unique=unique, processed=processed)
    numeric = train.select_dtypes(include="number")
    features = df.select_dtypes(include="number").drop(columns=["critical_temp", "above_77k"])
    elements = unique.drop(columns=["critical_temp", "material"]).select_dtypes(include="number")
    complexity = elements.gt(0).sum(axis=1)
    n = len(train)
    tables = {}
    counts = df.above_77k.value_counts().reindex([0, 1], fill_value=0)
    tables["class_balance.csv"] = pd.DataFrame({"above_77k": counts.index,
        "record_count": counts.values, "record_proportion": counts.values / n})
    tables["descriptive_statistics.csv"] = df.describe().T.rename_axis("feature").reset_index()
    correlation_inputs = features.assign(critical_temp=df.critical_temp)
    correlations = pd.DataFrame({
        "pearson_correlation": correlation_inputs.corr()["critical_temp"].drop("critical_temp"),
        "spearman_correlation": correlation_inputs.rank().corr()["critical_temp"].drop("critical_temp"),
    })
    correlations["absolute_correlation"] = correlations.pearson_correlation.abs()
    tables["feature_target_correlations.csv"] = correlations.sort_values(
        "absolute_correlation", ascending=False).rename_axis("feature").reset_index()
    nzv = []
    for name in features:
        frequencies = features[name].value_counts(dropna=False)
        distinct = len(frequencies)
        ratio = frequencies.iloc[0] / frequencies.iloc[1] if distinct > 1 else np.inf
        constant = distinct == 1
        nzv.append({"feature": name, "unique_count": distinct, "unique_percent": distinct / n * 100,
                    "frequency_ratio": ratio, "constant": constant,
                    "near_zero_variance": constant or (distinct / n <= .10 and ratio > 19)})
    tables["near_zero_variance.csv"] = pd.DataFrame(nzv)
    class_summary = []
    for label, group in df.groupby("above_77k"):
        summary = group.drop(columns="above_77k").describe().T.rename_axis("feature").reset_index()
        summary.insert(0, "above_77k", label)
        class_summary.append(summary)
    tables["class_feature_summary.csv"] = pd.concat(class_summary, ignore_index=True)
    composition = unique.groupby("material").critical_temp.agg(
        record_count="size", unique_temperature_count="nunique", minimum_temperature="min",
        maximum_temperature="max").reset_index()
    composition["temperature_range"] = composition.maximum_temperature - composition.minimum_temperature
    composition["crosses_77k_boundary"] = composition.minimum_temperature.le(77) & composition.maximum_temperature.gt(77)
    composition = composition.sort_values(["record_count", "material"], ascending=[False, True])
    tables["composition_frequency.csv"] = composition
    tables["repeated_formula_examples.csv"] = composition.loc[composition.record_count.gt(1)].head(20)
    complexity_summary = pd.DataFrame({"element_count": complexity, "critical_temp": train.critical_temp,
                                       "above_77k": df.above_77k}).groupby("element_count").agg(
        record_count=("critical_temp", "size"), mean_temperature=("critical_temp", "mean"),
        median_temperature=("critical_temp", "median"), minimum_temperature=("critical_temp", "min"),
        maximum_temperature=("critical_temp", "max"), above_77k_count=("above_77k", "sum")).reset_index()
    complexity_summary["above_77k_proportion"] = complexity_summary.above_77k_count / complexity_summary.record_count
    tables["composition_complexity.csv"] = complexity_summary
    prevalence = elements.gt(0).sum().rename("record_count").rename_axis("element").reset_index()
    prevalence["record_proportion"] = prevalence.record_count / n
    tables["element_prevalence.csv"] = prevalence.sort_values("record_count", ascending=False)
    q1, q3 = numeric.quantile(.25), numeric.quantile(.75)
    iqr = q3 - q1
    outliers = (numeric.lt(q1 - 1.5 * iqr) | numeric.gt(q3 + 1.5 * iqr)).sum()
    outlier_table = outliers.rename("record_count").rename_axis("feature").reset_index()
    outlier_table["record_proportion"] = outlier_table.record_count / n
    tables["top_iqr_outliers.csv"] = outlier_table.sort_values("record_count", ascending=False).head(20)
    quality = {
        "train_records": n, "unique_m_records": len(unique),
        "train_missing_values": int(train.isna().sum().sum()),
        "unique_m_missing_values": int(unique.isna().sum().sum()),
        "train_infinite_values": int(np.isinf(numeric).sum().sum()),
        "unique_m_infinite_values": int(np.isinf(unique.select_dtypes(include="number")).sum().sum()),
        "train_exact_duplicates": int(train.duplicated().sum()),
        "unique_m_exact_duplicates": int(unique.duplicated().sum()),
        "duplicated_predictors": int(train.drop(columns="critical_temp").duplicated().sum()),
        "blank_material_labels": int(unique.material.fillna("").str.strip().eq("").sum()),
        "material_surrounding_spaces": int(unique.material.ne(unique.material.str.strip()).sum()),
        "nonpositive_critical_temp": int(train.critical_temp.le(0).sum()),
        "nonpositive_number_of_elements": int(train.number_of_elements.le(0).sum()),
        "negative_element_quantities": int(elements.lt(0).sum().sum()),
        "element_count_mismatches": int(complexity.ne(train.number_of_elements).sum()),
        "temperature_mismatches": int(train.critical_temp.ne(unique.critical_temp).sum()),
        "constant_features": int(tables["near_zero_variance.csv"].constant.sum()),
        "near_zero_variance_features": int(tables["near_zero_variance.csv"].near_zero_variance.sum()),
        "repeated_formulas": int(composition.record_count.gt(1).sum()),
        "records_removed": n - len(df),
    }
    tables["data_quality_summary.csv"] = pd.DataFrame(quality.items(), columns=["metric", "value"])
    tables["cleaning_summary.csv"] = pd.DataFrame([
        {"check": "raw_records", "observed": n, "decision": "retained", "status": "reported"},
        {"check": "processed_records", "observed": len(df), "decision": "retained", "status": "reported"},
        {"check": "records_removed", "observed": 0, "decision": "none", "status": "passed"},
        {"check": "exact_train_duplicates", "observed": quality["train_exact_duplicates"], "decision": "retained", "status": "reported"},
        {"check": "potential_iqr_outlier_cells", "observed": int(outliers.sum()), "decision": "retained", "status": "reported"},
    ])
    directory = root / "reports/tables"
    directory.mkdir(parents=True, exist_ok=True)
    for filename, table in tables.items():
        table.to_csv(directory / filename, index=False)
    return tables
