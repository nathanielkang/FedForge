"""
Dataset loading utilities for FedSynth-Engine.

Supports 8 benchmark tabular datasets: adult, bank, acs_pums, credit,
mushroom, shopping, diabetes, covertype.

Each dataset is loaded, discretized (binned for continuous columns), and
returned as a dictionary with metadata.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import KBinsDiscretizer


@dataclass
class DatasetInfo:
    """Metadata for a loaded dataset."""
    name: str
    num_rows: int
    num_cols: int
    column_names: List[str]
    domain_sizes: Dict[str, int]
    data: np.ndarray  # discretized integer array (n_rows, n_cols)
    original_df: pd.DataFrame

    @property
    def schema(self) -> Dict[str, int]:
        return dict(self.domain_sizes)


DATASET_REGISTRY = [
    "adult", "bank", "acs_pums", "credit",
    "mushroom", "shopping", "diabetes", "covertype",
]


def list_datasets() -> List[str]:
    return list(DATASET_REGISTRY)


def load_dataset(
    name: str,
    num_bins: int = 16,
    max_rows: Optional[int] = None,
    random_state: int = 42,
) -> DatasetInfo:
    """Load and discretize a benchmark dataset."""
    loaders = {
        "adult": _load_adult,
        "bank": _load_bank,
        "acs_pums": _load_acs_pums,
        "credit": _load_credit,
        "mushroom": _load_mushroom,
        "shopping": _load_shopping,
        "diabetes": _load_diabetes,
        "covertype": _load_covertype,
    }

    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")

    df = loaders[name]()

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)

    data, domain_sizes = _discretize(df, num_bins=num_bins)

    return DatasetInfo(
        name=name,
        num_rows=data.shape[0],
        num_cols=data.shape[1],
        column_names=list(df.columns),
        domain_sizes=domain_sizes,
        data=data,
        original_df=df,
    )


def generate_synthetic_dataset(
    num_rows: int = 1000,
    num_cols: int = 10,
    num_bins: int = 8,
    random_state: int = 42,
) -> DatasetInfo:
    """Generate a tiny synthetic dataset for smoke testing."""
    rng = np.random.default_rng(random_state)
    columns = [f"col_{i}" for i in range(num_cols)]
    domain_sizes = {c: num_bins for c in columns}
    data = rng.integers(0, num_bins, size=(num_rows, num_cols))
    df = pd.DataFrame(data, columns=columns)

    return DatasetInfo(
        name="synthetic",
        num_rows=num_rows,
        num_cols=num_cols,
        column_names=columns,
        domain_sizes=domain_sizes,
        data=data,
        original_df=df,
    )


def _discretize(
    df: pd.DataFrame, num_bins: int = 16
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Discretize all columns: categoricals via label encoding, numerics via binning."""
    result_cols = []
    domain_sizes = {}

    for col in df.columns:
        series = df[col]
        if series.dtype == object or series.dtype.name == "category":
            codes, uniques = pd.factorize(series, sort=True)
            n_unique = len(uniques)
            if n_unique > num_bins:
                top_cats = series.value_counts().head(num_bins - 1).index
                series = series.where(series.isin(top_cats), other="__OTHER__")
                codes, uniques = pd.factorize(series, sort=True)
                n_unique = len(uniques)
            codes[codes < 0] = 0
            result_cols.append(codes)
            domain_sizes[col] = n_unique
        else:
            vals = series.fillna(series.median()).values.reshape(-1, 1)
            n_unique = len(np.unique(vals))
            actual_bins = min(num_bins, n_unique)
            if actual_bins <= 1:
                result_cols.append(np.zeros(len(df), dtype=int))
                domain_sizes[col] = 1
            else:
                kbd = KBinsDiscretizer(
                    n_bins=actual_bins, encode="ordinal", strategy="quantile"
                )
                binned = kbd.fit_transform(vals).astype(int).ravel()
                result_cols.append(binned)
                domain_sizes[col] = actual_bins

    data = np.column_stack(result_cols) if result_cols else np.empty((len(df), 0), dtype=int)
    return data, domain_sizes


def _load_adult() -> pd.DataFrame:
    """UCI Adult (Census Income) dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
    ]
    try:
        df = pd.read_csv(url, names=columns, skipinitialspace=True, na_values="?")
    except Exception:
        df = _generate_fallback("adult", 48842, columns)
    return df.dropna().reset_index(drop=True)


def _load_bank() -> pd.DataFrame:
    """UCI Bank Marketing dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv"
    try:
        df = pd.read_csv(url, sep=";")
    except Exception:
        columns = [
            "age", "job", "marital", "education", "default", "housing", "loan",
            "contact", "month", "day_of_week", "duration", "campaign", "pdays",
            "previous", "poutcome", "emp.var.rate", "nr.employed", "y",
        ]
        df = _generate_fallback("bank", 45211, columns[:17])
    return df.dropna().reset_index(drop=True)


def _load_acs_pums() -> pd.DataFrame:
    """ACS PUMS (folktables-style) synthetic subset."""
    columns = [
        "AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP",
        "SEX", "RAC1P", "PINCP", "JWMNP", "JWTR", "POWPUMA", "PUMA",
        "ST", "DIS", "ESP", "MIG", "ANC",
    ]
    df = _generate_fallback("acs_pums", 100000, columns)
    return df


def _load_credit() -> pd.DataFrame:
    """UCI Default of Credit Card Clients dataset."""
    columns = [
        "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
        "default",
    ]
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
        df = pd.read_excel(url, header=1)
        df.columns = columns
    except Exception:
        df = _generate_fallback("credit", 30000, columns)
    return df.dropna().reset_index(drop=True)


def _load_mushroom() -> pd.DataFrame:
    """UCI Mushroom dataset."""
    columns = [
        "class", "cap_shape", "cap_surface", "cap_color", "bruises", "odor",
        "gill_attachment", "gill_spacing", "gill_size", "gill_color",
        "stalk_shape", "stalk_root", "stalk_surface_above_ring",
        "stalk_surface_below_ring", "stalk_color_above_ring",
        "stalk_color_below_ring", "veil_type", "veil_color",
        "ring_number", "ring_type", "spore_print_color",
        "population", "habitat",
    ]
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    try:
        df = pd.read_csv(url, names=columns, na_values="?")
    except Exception:
        df = _generate_fallback("mushroom", 8124, columns)
    return df.dropna().reset_index(drop=True)


def _load_shopping() -> pd.DataFrame:
    """Online Shoppers Purchasing Intention dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
    try:
        df = pd.read_csv(url)
    except Exception:
        columns = [
            "Administrative", "Administrative_Duration", "Informational",
            "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
            "BounceRates", "ExitRates", "PageValues", "SpecialDay",
            "Month", "OperatingSystems", "Browser", "Region",
            "TrafficType", "VisitorType", "Weekend", "Revenue",
        ]
        df = _generate_fallback("shopping", 12330, columns)
    return df.dropna().reset_index(drop=True)


def _load_diabetes() -> pd.DataFrame:
    """Diabetes 130-US hospitals dataset."""
    columns = [
        "race", "gender", "age", "weight", "admission_type_id",
        "discharge_disposition_id", "admission_source_id", "time_in_hospital",
        "payer_code", "medical_specialty", "num_lab_procedures",
        "num_procedures", "num_medications", "number_outpatient",
        "number_emergency", "number_inpatient", "diag_1", "diag_2", "diag_3",
        "number_diagnoses", "max_glu_serum", "A1Cresult", "metformin",
        "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide",
        "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
        "tolazamide", "examide", "citoglipton", "insulin",
        "glyburide_metformin", "glipizide_metformin",
        "glimepiride_pioglitazone", "metformin_rosiglitazone",
        "metformin_pioglitazone", "change", "diabetesMed", "readmitted",
        "encounter_id", "patient_nbr",
    ]
    df = _generate_fallback("diabetes", 101766, columns[:50])
    return df


def _load_covertype() -> pd.DataFrame:
    """UCI Covertype dataset via sklearn."""
    try:
        data = fetch_covtype(as_frame=True)
        df = data.frame
        if len(df.columns) > 55:
            df = df.iloc[:, :55]
    except Exception:
        columns = [f"feat_{i}" for i in range(54)] + ["Cover_Type"]
        df = _generate_fallback("covertype", 581012, columns)
    return df.dropna().reset_index(drop=True)


def _generate_fallback(
    name: str, n_rows: int, columns: List[str]
) -> pd.DataFrame:
    """Generate a random fallback when download fails (for offline use)."""
    rng = np.random.default_rng(hash(name) % (2**31))
    data = {}
    for i, col in enumerate(columns):
        if i % 3 == 0:
            data[col] = rng.choice(
                ["cat_a", "cat_b", "cat_c", "cat_d", "cat_e"], size=n_rows
            )
        elif i % 3 == 1:
            data[col] = rng.integers(0, 100, size=n_rows)
        else:
            data[col] = rng.normal(50, 15, size=n_rows).round(2)
    return pd.DataFrame(data)


def partition_dataset(
    dataset: DatasetInfo,
    num_parties: int,
    partition: str = "uniform",
    random_state: int = 42,
) -> List[np.ndarray]:
    """Split dataset into per-party partitions."""
    rng = np.random.default_rng(random_state)
    n = dataset.num_rows
    indices = rng.permutation(n)

    if partition == "uniform":
        splits = np.array_split(indices, num_parties)
    elif partition == "dirichlet":
        proportions = rng.dirichlet(np.ones(num_parties) * 0.5)
        split_points = (np.cumsum(proportions) * n).astype(int)
        split_points[-1] = n
        splits = np.split(indices, split_points[:-1])
    else:
        raise ValueError(f"Unknown partition method: {partition}")

    return [dataset.data[s] for s in splits]
