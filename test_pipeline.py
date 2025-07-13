import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
from unittest.mock import patch
import pipeline


def test_pipeline_main_returns_expected_columns():
    # Dummy data resembling required columns
    train_df = pd.DataFrame({
        'Id': [1, 2],
        'ranker_id': [1, 1],
        'selected': [1, 2],
        'feature': [0.1, 0.2]
    })

    test_df = pd.DataFrame({
        'Id': [3, 4],
        'ranker_id': [1, 2],
        'feature': [0.3, 0.4]
    })

    sample_submission_df = pd.DataFrame({'Id': [3, 4], 'ranker_id': [1, 2], 'selected': [0, 0]})
    test_ids_df = test_df[['Id', 'ranker_id']]

    # Mock pipeline functions to simplify processing
    with patch('pipeline.load_data', return_value=(train_df, test_df, sample_submission_df, test_ids_df)), \
         patch('pipeline.preprocess_dataframe', side_effect=lambda df, is_train: df), \
         patch('pipeline.prepare_matrices', side_effect=lambda a, b: (a[["feature"]], a['selected'], b[["feature"]], a['ranker_id'])), \
         patch('pipeline.encode_categoricals', side_effect=lambda X, X_test: (X, X_test, [])), \
         patch('pipeline.clean_features', side_effect=lambda X, X_test, low_var_thresh=1: (X, X_test, {})), \
         patch('lightgbm.LGBMRanker.fit', return_value=None), \
         patch('pandas.DataFrame.to_parquet', return_value=None), \
         patch('pandas.DataFrame.to_csv', return_value=None):

        def dummy_train_model(X, y, X_test, ranker_ids, cat_features):
            # Call fit once to satisfy mocked fit
            model = pipeline.lgb.LGBMRanker()
            model.fit(X, y, group=[len(X)])
            preds = np.zeros(len(X_test))
            return preds, None

        with patch('pipeline.train_model', side_effect=dummy_train_model):
            result = pipeline.main()

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ['Id', 'ranker_id', 'selected']


def test_encode_categoricals_encodes_union_int32():
    X = pd.DataFrame({'cat': ['a', 'b', 'c']})
    X_test = pd.DataFrame({'cat': ['b', 'd']})

    X_enc, X_test_enc, cats = pipeline.encode_categoricals(X.copy(), X_test.copy())

    assert cats == ['cat']
    assert X_enc['cat'].dtype == np.int32
    assert X_test_enc['cat'].dtype == np.int32
    # union of 4 categories should produce 4 unique codes
    combined = pd.concat([X_enc['cat'], X_test_enc['cat']])
    assert len(set(combined.tolist())) == 4
    # ensure new category from test isn't mapped to -1
    assert -1 not in X_test_enc['cat'].values


def test_encode_categoricals_frequency_encoding():
    X = pd.DataFrame({'legs0_segments0_departureFrom_airport_iata': ['SFO', 'LAX', 'SFO', 'JFK']})
    X_test = pd.DataFrame({'legs0_segments0_departureFrom_airport_iata': ['LAX', 'ORD']})

    X_enc, X_test_enc, cats = pipeline.encode_categoricals(
        X.copy(),
        X_test.copy(),
        high_card_cols=['legs0_segments0_departureFrom_airport_iata'],
    )

    assert 'legs0_segments0_departureFrom_airport_iata' not in cats
    assert np.issubdtype(X_enc['legs0_segments0_departureFrom_airport_iata'].dtype, np.floating)

    freq = X['legs0_segments0_departureFrom_airport_iata'].value_counts()
    expected_train = np.log1p(X['legs0_segments0_departureFrom_airport_iata'].map(freq).fillna(1)).astype(np.float32)
    expected_test = np.log1p(X_test['legs0_segments0_departureFrom_airport_iata'].map(freq).fillna(1)).astype(np.float32)

    assert np.allclose(X_enc['legs0_segments0_departureFrom_airport_iata'].values, expected_train.values)
    assert np.allclose(X_test_enc['legs0_segments0_departureFrom_airport_iata'].values, expected_test.values)
