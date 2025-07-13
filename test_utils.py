import pandas as pd
import numpy as np

import utils


def test_unify_nan_strategy_flags_and_nan_preservation():
    df = pd.DataFrame({
        "booking_lead_days": [10, np.nan, 20],
        "is_cheapest": [1, np.nan, 0],
        "price_per_tax": [1.0, np.nan, 2.0],
        "random_int": pd.Series([1, pd.NA, 2], dtype="Int64"),
    })
    out = utils.unify_nan_strategy(df.copy())

    # -1 sentinel should be used for MINUS1 columns
    assert out["booking_lead_days"].tolist() == [10, -1, 20]
    assert out["booking_lead_days_missing"].tolist() == [0, 1, 0]

    # ZERO columns should be filled with 0 and keep int8 dtype
    assert out["is_cheapest"].tolist() == [1, 0, 0]
    assert out["is_cheapest"].dtype == np.int8
    # NaNs should be preserved for KEEP_NA columns
    assert out["price_per_tax"].isna().tolist() == [False, True, False]
    # Nullable integers with NaNs should be cast to float32
    assert out["random_int"].dtype == np.float32

    # Missing indicators should exist
    for col in ["is_cheapest", "price_per_tax", "random_int"]:
        assert f"{col}_missing" in out.columns


def test_create_remaining_features_leaves_nan_for_minus1_cols():
    df = pd.DataFrame(
        {
            "ranker_id": [1],
            "requestDate": [pd.to_datetime("2024-01-01")],
            "legs0_departureAt": [pd.NaT],
            "legs0_duration": [50],
            "legs1_duration": [60],
            "totalPrice": [100.0],
            "taxes": [10.0],
        }
    )
    out = utils.create_remaining_features(df.copy(), is_train=True)
    assert pd.isna(out.loc[0, "booking_lead_days"])

    out2 = utils.unify_nan_strategy(out.copy())
    assert out2.loc[0, "booking_lead_days"] == -1
    assert out2.loc[0, "booking_lead_days_missing"] == 1


def test_clean_features_drops_low_var_and_corr():
    X = pd.DataFrame({
        "single_val": [1, 1, 1, 1],
        "varied1": [1, 2, 3, 4],
        "varied2": [2, 4, 6, 8],
        "varied3": [1, 2, 1, 2],
    })
    X_test = X.copy()

    X_clean, X_test_clean, dropped = utils.clean_features(
        X, X_test, verbose=False
    )

    assert "single_val" not in X_clean.columns
    assert "single_val" not in X_test_clean.columns
    assert dropped["low_var"] == ["single_val"]

    assert "varied2" not in X_clean.columns
    assert "varied2" not in X_test_clean.columns
    assert "varied2" in dropped["high_corr"]
    assert "varied3" in X_clean.columns


def test_calculate_hit_rate_at_3():
    group_a = pd.DataFrame({
        "ranker_id": ["A"] * 11,
        "selected": [0, 1] + [0] * 9,
        "predicted_rank": list(range(1, 12)),
    })
    group_b = pd.DataFrame({
        "ranker_id": ["B"] * 11,
        "selected": [0] * 4 + [1] + [0] * 6,
        "predicted_rank": list(range(1, 12)),
    })
    df = pd.concat([group_a, group_b], ignore_index=True)

    hr = utils.calculate_hit_rate_at_3(df)
    assert hr == 0.5


def test_smart_fill_numeric_fills_and_downcasts():
    df = pd.DataFrame({
        "bin1": [1, np.nan, 0],
        "bin2": pd.Series([pd.NA, 1, 1], dtype="Int64"),
        "other": [1.5, np.nan, 3.2],
    })

    out = utils.smart_fill_numeric(df.copy(), zero_cols=["bin1", "bin2"])

    assert out["bin1"].tolist() == [1, 0, 0]
    assert out["bin1"].dtype == np.int8
    assert out["bin2"].tolist() == [0, 1, 1]
    assert out["bin2"].dtype == np.int8
    assert out["other"].isna().tolist() == [False, True, False]


def test_log_mem_usage_outputs(capsys):
    df = pd.DataFrame({"a": [1, 2, 3]})
    utils.log_mem_usage(df, "dummy")
    captured = capsys.readouterr()
    assert "dummy" in captured.out


def test_create_initial_datetime_features_normalises_offsets():
    df = pd.DataFrame({"legs0_departureAt": ["2024-08-21T16:00:00 UTC+5"]})
    out = utils.create_initial_datetime_features(df.copy())
    ts = out.loc[0, "legs0_departureAt"]
    assert ts.utcoffset().total_seconds() == 5 * 3600
