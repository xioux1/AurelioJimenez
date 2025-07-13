import pandas as pd
import pipeline


def test_preprocess_adds_duration_pct_rank():
    df = pd.DataFrame({
        'Id': [1, 2, 3, 4],
        'ranker_id': [1, 1, 2, 2],
        'selected': [0, 1, 0, 1],
        'totalPrice': [100, 150, 200, 120],
        'taxes': [10, 15, 20, 12],
        'legs0_duration': [60, 80, 90, 100],
        'legs1_duration': [70, 60, 80, 90],
        'legs1_segments0_departureFrom_airport_iata': ['A', 'B', 'C', 'D'],
        'legs0_segments0_departureFrom_airport_iata': ['X', 'Y', 'Z', 'W'],
        'frequentFlyer': ['SU', 'S7', 'U6', 'TK'],
        'legs0_segments0_marketingCarrier_code': ['SU', 'SU', 'U6', 'TK'],
        'legs1_segments0_marketingCarrier_code': ['SU', 'SU', 'U6', 'TK'],
        'legs0_segments0_baggageAllowance_quantity': [1, 0, 1, 0],
        'legs1_segments0_baggageAllowance_quantity': [0, 1, 0, 1],
        'miniRules0_monetaryAmount': [0, 10, 0, 5],
        'miniRules1_monetaryAmount': [5, 0, 10, 0],
        'searchRoute': ['MOWLED/LEDMOW'] * 4,
        'isVip': [0, 1, 0, 1],
        'pricingInfo_isAccessTP': [1, 0, 1, 0],
        'legs0_segments0_cabinClass': [1, 1, 1, 1],
        'legs1_segments0_cabinClass': [2, 2, 2, 2],
    })

    processed = pipeline.preprocess_dataframe(df.copy(), is_train=True)
    assert 'duration_pct_rank' in processed.columns
