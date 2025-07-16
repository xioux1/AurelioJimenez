# Configuration constants for preprocessing pipeline

# Columns to exclude from processing
COL_BLACKLIST = set()

# Core columns to load from the dataset
INITIAL_CORE_COLUMNS = [
    'Id', 'ranker_id', 'selected', 'profileId', 'companyID',
    'requestDate', 'totalPrice', 'taxes',
    'legs0_departureAt', 'legs0_arrivalAt', 'legs0_duration',
    'legs1_departureAt', 'legs1_arrivalAt', 'legs1_duration',
    'legs0_segments0_departureFrom_airport_iata', 'legs0_segments0_arrivalTo_airport_iata',
    'legs0_segments0_marketingCarrier_code', 'legs0_segments0_cabinClass',
    'legs0_segments0_baggageAllowance_quantity',
    'searchRoute',
    'pricingInfo_isAccessTP', 'pricingInfo_passengerCount',
    'sex', 'nationality', 'isVip',
    'miniRules0_monetaryAmount', 'miniRules0_percentage',
    'miniRules1_monetaryAmount', 'miniRules1_percentage'
]

# Optional additional columns that may be present
EXTRA_COLS = [
    'frequentFlyer', 'corporateTariffCode',
    'legs1_segments0_baggageAllowance_quantity',
    'legs1_segments0_cabinClass',
    'legs1_segments0_departureFrom_airport_iata',
    'legs1_segments0_arrivalTo_airport_iata',
    'legs1_segments0_marketingCarrier_code',
]
