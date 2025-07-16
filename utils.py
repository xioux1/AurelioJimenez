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

