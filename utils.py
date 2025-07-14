import pandas as pd
import numpy as np
import gc
import re


def _normalise_utc_offset(ts: str) -> str:
    """Return ``ts`` with ``UTC±HH`` offsets expanded to ``±HH:MM``.

    ``pd.to_datetime`` misinterprets offsets like ``UTC+5`` as ``UTC-05``.
    This helper rewrites them to a standard ``+05:00`` style so pandas
    parses them correctly.
    """
    if not isinstance(ts, str):
        return ts
    match = re.search(r"\sUTC([+-])(\d{1,2})(?::?(\d{2}))?$", ts)
    if match:
        sign, hour, minute = match.groups()
        hour = hour.zfill(2)
        minute = minute or "00"
        repl = f"{sign}{hour}:{minute}"
        ts = ts[: match.start()] + repl
    return ts

IMPUTE_RULES = {
    "ZERO": [
        'is_one_way','has_return','is_direct_leg0','is_direct_leg1',
        'both_direct','has_baggage','has_fees','is_cheapest',
        'is_most_expensive','n_segments_leg0','n_segments_leg1',
        'total_segments','is_popular_route','is_major_carrier',
    ],
    "MINUS1": [
        'booking_lead_days',
    ],
    "KEEP_NA": [
        'price_per_tax','tax_rate',
    ]
}

def smart_fill_numeric(df, zero_cols=()):
    """S\u00f3lo forz\u00e1 0 en binarios donde NA es imposible."""
    for c in zero_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(np.int8)
    return df

# --- utils_fill.py ----------------------------------------------------------
def add_missing_flags(df: pd.DataFrame, cols):
    """Para cada col num\u00e9rica en `cols`, crea col+'_missing' y deja el NaN sin tocar."""
    for c in cols:
        if df[c].isna().any():
            df[f'{c}_missing'] = df[c].isna().astype(np.int8)
    return df

def fill_with_flag(df, col, value):
    flag = f"{col}_missing"
    if flag not in df.columns:
        df = add_missing_flags(df, [col])
    df[col] = df[col].fillna(value)
    return df

def unify_nan_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise NaN handling and create missing-value indicators.

    Numeric columns keep their NaNs so LightGBM can leverage them
    directly except for those listed under ``ZERO`` or ``MINUS1``.
    ``ZERO`` columns have NaNs replaced with ``0`` (while recording
    the missingness) so they can remain integer typed. ``MINUS1``
    columns have NaNs replaced with ``-1`` and corresponding
    ``*_missing`` indicators added.  Missing flags are added for every
    column that contains NaNs.
    """

    # Replace any leftover infinities from feature engineering
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Columns where -1 encodes a specific meaning
    for col in IMPUTE_RULES["MINUS1"]:
        if col in df.columns:
            df = fill_with_flag(df, col, -1)

    # Columns where NaNs should become 0 so they remain integers
    for col in IMPUTE_RULES["ZERO"]:
        if col in df.columns:
            df = fill_with_flag(df, col, 0)
            # keep these features as small integers to save memory
            df[col] = df[col].astype(np.int8)

    # Ensure nullable integers with remaining NaNs become floats for LGBM
    for c in df.select_dtypes("Int64").columns:
        if df[c].isna().any():
            df[c] = df[c].astype("float32")

    for c in df.columns[df.isna().any()]:
        flag = f"{c}_missing"
        if flag not in df.columns:
            df[flag] = df[c].isna().astype(np.int8)

    return df

# ———————————————————————— CLEAN FEATURES ————————————————————————

def clean_features(X: pd.DataFrame,
                   X_test: pd.DataFrame,
                   *,
                   low_var_thresh: int = 0.01,
                   corr_thresh: float = 0.95,
                   verbose: bool = True):
    """Drop near-constant and highly correlated columns.

    Columns with ``low_var_thresh`` or fewer unique values (including ``NaN``)
    are removed. With the default of ``1``, only columns that contain a single
    unique value are dropped.
    """
    dropped = {"low_var": [], "high_corr": []}
    low_var = [c for c in X.columns if X[c].nunique(dropna=False) <= low_var_thresh]
    X = X.drop(columns=low_var)
    X_test = X_test.drop(columns=low_var, errors="ignore")
    dropped["low_var"] = low_var
    # Compute correlation without forcing NaNs to zero to avoid
    # spurious high correlations when missingness patterns overlap.
    # pandas' corr already uses pairwise deletion, so NaNs are skipped
    # on a per-column-pair basis.
    num_df = X.select_dtypes(include=np.number)
    corr = num_df.corr(min_periods=1).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # iterate over pairs in the upper triangle so we keep the first
    # column encountered and drop the second when correlation exceeds
    # the threshold
    to_drop = set()
    cols = list(upper.columns)
    for i, col1 in enumerate(cols[:-1]):
        for j in range(i + 1, len(cols)):
            col2 = cols[j]
            if pd.notna(upper.loc[col1, col2]) and upper.loc[col1, col2] >= corr_thresh:
                to_drop.add(col2)

    X = X.drop(columns=to_drop)
    X_test = X_test.drop(columns=to_drop, errors="ignore")
    dropped["high_corr"] = list(to_drop)
    if verbose:
        print(f"  • low-var  : {len(low_var)} cols drop")
        print(f"  • high-corr: {len(to_drop)} cols drop")
    return X, X_test, dropped

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    return df

def log_mem_usage(df: pd.DataFrame, label: str = "") -> None:
    """Print memory usage information for ``df``.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe whose memory footprint should be logged.
    label : str, optional
        Descriptive label printed alongside the memory stats.
    """

    tag = f"[{label}] " if label else ""
    usage = df.memory_usage(deep=True)
    total_mb = usage.sum() / 1024 ** 2
    print(f"{tag}Memory usage (deep):")
    print(usage.to_string())
    print(f"{tag}Total: {total_mb:.2f} MB")

def frequency_encode(train_series: pd.Series,
                     test_series: pd.Series | None = None,
                     *,
                     log: bool = True) -> tuple[pd.Series, pd.Series | None]:
    """Encode categories by their frequency derived from ``train_series``.

    Parameters
    ----------
    train_series : pd.Series
        Training column containing categorical values.
    test_series : pd.Series or None, optional
        Corresponding column from the test set. If provided, the same
        frequency mapping is applied.
    log : bool, default True
        Whether to apply ``np.log1p`` to the counts.

    Returns
    -------
    tuple
        Encoded training series and, when ``test_series`` is given, the
        encoded test series as well. Encodings are returned as
        ``float32``.
    """

    freq = train_series.value_counts()
    train_enc = train_series.map(freq).fillna(1)
    if test_series is not None:
        test_enc = test_series.map(freq).fillna(1)
    if log:
        train_enc = np.log1p(train_enc)
        if test_series is not None:
            test_enc = np.log1p(test_enc)
    train_enc = train_enc.astype("float32")
    if test_series is not None:
        test_enc = test_enc.astype("float32")
        return train_enc, test_enc
    return train_enc, None

def calculate_hit_rate_at_3(df_preds_with_true_and_rank):
    """Calculates HitRate@3."""
    hits = 0
    valid_queries_count = 0
    for ranker_id, group in df_preds_with_true_and_rank.groupby('ranker_id'):
        if len(group) <= 10:
            continue
        valid_queries_count += 1
        true_selected_item = group[group['selected'] == 1]
        if not true_selected_item.empty:
            rank_of_true_item = true_selected_item.iloc[0]['predicted_rank']
            if rank_of_true_item <= 3:
                hits += 1
    if valid_queries_count == 0:
        return 0.0
    return hits / valid_queries_count

def lgb_hit_rate_at_3(labels, preds, weight=None, group=None):
    """LightGBM evaluation metric for HitRate@3.

    Parameters
    ----------
    labels : array-like
        True labels for each item.
    preds : array-like
        Predicted scores for each item.
    weight : array-like or None
        Unused sample weights.
    group : array-like or None
        Group sizes for ranking tasks.

    Returns
    -------
    tuple
        ``("hr@3", score, True)`` where ``score`` is the HitRate@3 value and
        ``True`` indicates that higher values are better.
    """
    if group is None:
        raise ValueError("Group information is required to compute HitRate@3")

    hits = 0
    valid_queries = 0
    idx = 0
    for g in group:
        end = idx + g
        if g > 10:
            valid_queries += 1
            group_preds = preds[idx:end]
            group_labels = labels[idx:end]
            if np.any(group_labels == 1):
                true_idx = np.where(group_labels == 1)[0][0]
                rank = (-group_preds).argsort()
                rank_pos = np.where(rank == true_idx)[0][0] + 1
                if rank_pos <= 3:
                    hits += 1
        idx = end

    score = hits / valid_queries if valid_queries else 0.0
    return "hr@3", score, True

# -------------------------------------------------------------------
# Feature engineering helpers

def create_features(df):
    def hms_to_minutes(s: pd.Series) -> np.ndarray:
        out = np.full(len(s), np.nan, dtype="float32")
        if s.notna().any():
            ss = s.astype(str)
            hms = ss.str.extract(r'^(?P<h>\d{1,2}):(?P<m>\d{1,2})(?::\d{1,2})?$')
            mask = hms.notna().all(axis=1)
            out[mask] = hms['h'].astype(float).values[mask] * 60 + hms['m'].astype(float).values[mask]
            iso = ss.str.extract(r'^PT(?:(?P<ih>\d+)H)?(?:(?P<im>\d+)M)?')
            mask = iso.notna().any(axis=1)
            out[mask] = iso['ih'].fillna(0).astype(float).values[mask] * 60 + iso['im'].fillna(0).astype(float).values[mask]
            mask = ss.str.fullmatch(r'\d+(\.\d+)?')
            out[mask] = ss[mask].astype(float).values
        return out

    dur_cols = [
        'legs0_duration', 'legs1_duration',
        'legs0_segments0_duration', 'legs0_segments1_duration',
        'legs1_segments0_duration', 'legs1_segments1_duration'
    ]
    present = [c for c in dur_cols if c in df.columns]
    for c in present:
        df[c] = hms_to_minutes(df[c]).astype('float32')
    df = add_missing_flags(df, present)

    legs0 = df['legs0_duration'] if 'legs0_duration' in df.columns else np.nan
    legs1 = df['legs1_duration'] if 'legs1_duration' in df.columns else np.nan
    df['total_duration']  = legs0.fillna(0) + legs1.fillna(0)
    df['duration_ratio']  = legs0 / legs1
    bad = (df['duration_ratio'] < 0) | (df['duration_ratio'] > 5)
    df.loc[bad, 'duration_ratio'] = np.nan
    if {'totalPrice','taxes'}.issubset(df.columns):
        feat = {}
        feat["price_per_tax"] = df["totalPrice"] / (df["taxes"] + 1)
        feat["tax_rate"]      = df["taxes"] / (df["totalPrice"] + 1)
        feat["log_price"]     = np.log1p(df["totalPrice"])
    else:
        feat = {}
    legs0 = df.get("legs0_duration")
    legs1 = df.get("legs1_duration")
    seg_counts = {}
    for leg in (0, 1):
        pat = rf'^legs{leg}_segments\d+_departureFrom_airport_iata$'
        seg_cols = [c for c in df.columns if re.match(pat, c)]
        seg_counts[leg] = (
            df[seg_cols].notna().sum(axis=1).astype("int8") if seg_cols else 0
        )
    feat["total_segments"] = seg_counts[0] + seg_counts[1]
    feat["is_one_way"] = (
        df["legs1_duration"].isna() |
        df["legs1_segments0_departureFrom_airport_iata"].isna()
    ).astype("int8")
    feat["has_return"] = 1 - feat["is_one_way"]
    grp = df.groupby("ranker_id")
    feat["price_pct_rank"]    = grp["totalPrice"].rank(pct=True)
    feat["duration_rank"]     = grp["total_duration"].rank()
    feat["duration_pct_rank"] = grp["total_duration"].rank(pct=True)
    feat["is_cheapest"]       = (grp["totalPrice"].transform("min") == df["totalPrice"]).astype("int8")
    feat["is_most_expensive"] = (grp["totalPrice"].transform("max") == df["totalPrice"]).astype("int8")
    ff = df["frequentFlyer"].fillna("").astype(str)
    feat["n_ff_programs"] = ff.str.count("/") + (ff != "")
    carrier_cols = ["legs0_segments0_marketingCarrier_code", "legs1_segments0_marketingCarrier_code"]
    present_airlines = set()
    for col in carrier_cols:
        if col in df.columns:
            present_airlines.update(df[col].dropna().unique())
    ff_flags = {}
    for al in ["SU", "S7", "U6", "TK"]:
        if al in present_airlines:
            ff_flags[al] = ff.str.contains(fr"\b{al}\b").astype("int8")
    feat["ff_matches_carrier"] = 0
    if "legs0_segments0_marketingCarrier_code" in df.columns:
        for al, flag in ff_flags.items():
            feat["ff_matches_carrier"] |= (
                (flag == 1)
                & (df["legs0_segments0_marketingCarrier_code"] == al)
            ).astype("int8")
    feat["baggage_total"] = (
        df["legs0_segments0_baggageAllowance_quantity"].fillna(0)
        + df["legs1_segments0_baggageAllowance_quantity"].fillna(0)
    )
    feat["total_fees"]  = df["miniRules0_monetaryAmount"].fillna(0) + df["miniRules1_monetaryAmount"].fillna(0)
    feat["fee_rate"]    = feat["total_fees"] / (df["totalPrice"] + 1)
    for col in ("legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt"):
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce")
            feat[f"{col}_hour"]      = dt.dt.hour.astype("float16")
            feat[f"{col}_weekday"]   = dt.dt.weekday.astype("float16")
            h = dt.dt.hour
            feat[f"{col}_business_time"] = (((6 <= h) & (h <= 9)) | ((17 <= h) & (h <= 20))).astype("int8")
    feat["is_direct_leg0"] = (seg_counts[0] == 1).astype("int8")
    feat["is_direct_leg1"] = np.where(
        feat["is_one_way"] == 1,
        0,
        (seg_counts[1] == 1).astype("int8"),
    )
    feat["both_direct"]    = (feat["is_direct_leg0"] & feat["is_direct_leg1"]).astype("int8")
    feat["is_vip_freq"]  = ((df["isVip"] == 1) | (feat["n_ff_programs"] > 0)).astype("int8")
    feat["group_size"]    = df.groupby("ranker_id")["Id"].transform("count")
    if "legs0_segments0_marketingCarrier_code" in df.columns:
        feat["is_major_carrier"] = df["legs0_segments0_marketingCarrier_code"].isin(["SU","S7","U6"]).astype("int8")
    else:
        feat["is_major_carrier"] = 0
    popular = {"MOWLED/LEDMOW","LEDMOW/MOWLED","MOWLED","LEDMOW","MOWAER/AERMOW"}
    feat["is_popular_route"] = df["searchRoute"].isin(popular).astype("int8")
    feat["avg_cabin_class"] = df[["legs0_segments0_cabinClass","legs1_segments0_cabinClass"]].mean(axis=1)
    df = pd.concat([df, pd.DataFrame(feat, index=df.index)], axis=1)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    for col in df.select_dtypes(include="object").columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            if "missing" not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(["missing"])
            df[col] = df[col].fillna("missing")
        else:
            df[col] = df[col].astype("category")
            df[col] = df[col].cat.add_categories(["missing"]).fillna("missing")
    return df

def create_initial_datetime_features(df):
    loaded_cols = df.columns
    potential_dt_cols = ['requestDate', 'legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt']
    for col in potential_dt_cols:
        if col in loaded_cols:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                current_dtype = df[col].dtype
                print(f"Converting column {col} (current dtype: {current_dtype}) to datetime.")
                df[col] = pd.to_datetime(
                    df[col]
                    .astype(str)
                    .map(_normalise_utc_offset),
                    errors="coerce",
                )
    return df

def create_remaining_features(df, is_train=True):
    potential_dt_cols_for_components = ['legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt']
    for col in potential_dt_cols_for_components:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col + '_hour'] = df[col].dt.hour.astype(np.int8, errors='ignore')
            df[col + '_dow'] = df[col].dt.dayofweek.astype(np.int8, errors='ignore')
    if (
        'legs0_departureAt' in df.columns
        and 'requestDate' in df.columns
        and pd.api.types.is_datetime64_any_dtype(df['legs0_departureAt'])
        and pd.api.types.is_datetime64_any_dtype(df['requestDate'])
    ):
        df['booking_lead_days'] = (
            df['legs0_departureAt'] - df['requestDate']
        ).dt.total_seconds() / (24 * 60 * 60)
    else:
        df['booking_lead_days'] = np.nan
    df['booking_lead_days'] = df['booking_lead_days'].astype(np.float32)
    if 'searchRoute' in df.columns:
        _ = df['searchRoute'].astype(str).str.contains('/')
    if 'legs1_departureAt' in df.columns and pd.api.types.is_datetime64_any_dtype(df['legs1_departureAt']):
        df['num_legs'] = 1 + df['legs1_departureAt'].notna().astype(np.int8)
    elif 'legs1_departureAt' in df.columns:
        df['num_legs'] = 1 + pd.to_datetime(df['legs1_departureAt'].astype(str), errors='coerce').notna().astype(np.int8)
    else:
        df['num_legs'] = 1
    for leg in range(2):
        pattern = fr'^legs{leg}_segments\d+_departureFrom_airport_iata$'
        seg_cols = [c for c in df.columns if re.match(pattern, c)]
        if seg_cols:
            df[f'num_segments_leg{leg}'] = df[seg_cols].notna().sum(axis=1).astype(np.int8)
        else:
            df[f'num_segments_leg{leg}'] = 0
    df['total_segments'] = (df['num_segments_leg0'] + df['num_segments_leg1']).astype(np.int8)
    for dur_col in ['legs0_duration', 'legs1_duration']:
        if not np.issubdtype(df[dur_col].dtype, np.number):
            df[dur_col] = pd.to_numeric(df[dur_col].astype(str), errors='coerce')
    df['total_flight_duration'] = (df['legs0_duration'] + df['legs1_duration']).astype(np.float32)
    if 'totalPrice' in df.columns and 'taxes' in df.columns:
        df['price_per_duration'] = (
            df['totalPrice'] / (df['total_flight_duration'] + 1e-6)
        ).fillna(0).astype(np.float32)
    else:
        df['price_per_duration'] = 0.0
    if 'pricingInfo_isAccessTP' in df.columns:
        df['is_compliant'] = df['pricingInfo_isAccessTP'].fillna(0).astype(np.int8)
    else:
        df['is_compliant'] = -1
    if 'legs0_segments0_baggageAllowance_quantity' in df.columns:
        df['baggage_leg0_included'] = (df['legs0_segments0_baggageAllowance_quantity'].fillna(0) > 0).astype(np.int8)
    else:
        df['baggage_leg0_included'] = -1
    if 'legs1_segments0_baggageAllowance_quantity' in df.columns:
        df['baggage_leg1_included'] = (df['legs1_segments0_baggageAllowance_quantity'].fillna(0) > 0).astype(np.int8)
        if 'baggage_leg0_included' in df.columns and df['baggage_leg0_included'].iloc[0] != -1:
            df['baggage_both_legs_included'] = (df['baggage_leg0_included'] & df['baggage_leg1_included']).astype(np.int8)
        else:
            df['baggage_both_legs_included'] = -1
    else:
        df['baggage_leg1_included'] = 0
        if 'baggage_leg0_included' in df.columns and df['baggage_leg0_included'].iloc[0] != -1:
            df['baggage_both_legs_included'] = df['baggage_leg0_included'].astype(np.int8)
        else:
            df['baggage_both_legs_included'] = -1
    group_key = 'ranker_id'
    if group_key not in df.columns:
        return df
    cols_for_group_features = []
    user_company_cats_loaded = [c for c in ['sex', 'nationality', 'isVip'] if c in df.columns]
    for col in user_company_cats_loaded:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(str)
        df[col] = df[col].fillna('MISSING').astype('category')
    binary_cols_loaded = [c for c in ['bySelf', 'isAccess3D'] if c in df.columns]
    for col in binary_cols_loaded:
        df[col] = df[col].fillna(0).astype(np.int8)
    df = reduce_mem_usage(df, verbose=False)
    return df

def dur_stats(df):
    for col in ['legs0_duration', 'legs1_duration']:
        print(col,
              "\u2192", df[col].dtype,
              "| NaN:", df[col].isna().mean(),
              "| 0.0:", (df[col] == 0).mean(),
              "| -1:",  (df[col] == -1).mean(),
              "| min:", df[col].min(),
              "| max:", df[col].max())

def check_rank_permutation(group):
    N = len(group)
    sorted_ranks = sorted(list(group['selected']))
    expected_ranks = list(range(1, N + 1))
    if sorted_ranks != expected_ranks:
        print(f"Invalid rank permutation for ranker_id: {group['ranker_id'].iloc[0]}")
        print(f"Expected: {expected_ranks}, Got: {sorted_ranks}")
        return False
    return True


def readme() -> str:
    """Return a short description of this project's purpose.

    This helper can be imported by notebooks or scripts to quickly
    display basic information about the utilities and pipeline
    contained in this repository.
    """

    return (
        "Utilities and pipeline for the AeroClub RecSys 2025 dataset. "
        "The code includes feature engineering helpers and a LightGBM "
        "training pipeline (see :mod:`pipeline`)."
    )

