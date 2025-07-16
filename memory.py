import pandas as pd
import numpy as np


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


def log_mem_usage(
    df: pd.DataFrame,
    label: str = "",
    *,
    summary: bool = False,
    top_n: int = 5,
) -> None:
    """Print memory usage information for ``df``."""
    tag = f"[{label}] " if label else ""
    usage = df.memory_usage(deep=True).sort_values(ascending=False)
    total_mb = usage.sum() / 1024 ** 2
    print(f"{tag}Memory usage (deep):")
    if summary:
        head = usage.head(top_n)
        print(head.to_string())
        if len(usage) > top_n:
            print(f"... (showing top {top_n} of {len(usage)} columns)")
    else:
        print(usage.to_string())
    print(f"{tag}Total: {total_mb:.2f} MB")
