import pandas as pd
import Analysis.visualisations as viz
import Analysis.eda as eda
import Analysis.ml as ml

### 1. DATA LOAD ###

results = pd.read_csv("data/results.csv")
races = pd.read_csv("data/races.csv")
drivers = pd.read_csv("data/drivers.csv")
constructors = pd.read_csv("data/constructors.csv")
qualifying = pd.read_csv("data/qualifying.csv")
status = pd.read_csv("data/status.csv")
circuits = pd.read_csv("data/circuits.csv")

## Rows Check
print("=== ORIGINAL DATA ===")
print(f"Results rows: {results.shape[0]}")
print(f"Races rows: {races.shape[0]}")
print(f"Drivers rows: {drivers.shape[0]}")
print(f"Constructors rows: {constructors.shape[0]}")
print(f"Qualifying rows: {qualifying.shape[0]}")
print(f"Status rows: {status.shape[0]}")
print(f"Circuits rows: {circuits.shape[0]}")
print("*" * 30, '\n')

### 2. DATA PREP ###

## Data Filter and Basic Prep
races_filtered = races[
    (races['year'] >= 2014) &
    (races['year'] <= 2024)
]

hybrid_race_ids = races_filtered['raceId']

results_filtered = results[
    results['raceId'].isin(hybrid_race_ids)
]

print("=== HYBRID ERA FILTER ===")
print(f"Filtered races: {races_filtered.shape[0]}")
print(f"Filtered results: {results_filtered.shape[0]}")
print(f"Years: {races_filtered['year'].min()} - {races_filtered['year'].max()}")
print("*" * 30, '\n')

qualifying = qualifying.rename(columns={
    'position': 'qualiPosition'
})

constructors = constructors.rename(columns={
    'name': 'constructorName'
})

circuits = circuits.rename(columns={
    'name': 'circuitName'
})



## Data Frame Build

# Merge - Races
df = results_filtered.merge(
    races_filtered[['raceId', 'year', 'name', 'date', 'circuitId']],
    on='raceId',
    how='left'
)

# Merge - Drivers
df = df.merge(
    drivers[['driverId', 'surname', 'forename', 'dob', 'nationality']],
    on='driverId',
    how='left'
)

# Merge - Constructors
df = df.merge(
    constructors[['constructorId', 'constructorName']],
    on='constructorId',
    how='left'
)

# Merge - Qualifying
df = df.merge(
    qualifying[['raceId', 'driverId', 'qualiPosition', 'q1', 'q2', 'q3']],
    on=['raceId', 'driverId'],
    how='left'
)

# Merge - Status
df = df.merge(
    status[['statusId', 'status']],
    on='statusId',
    how='left'
)

# Merge - Circuits

df = df.merge(
    circuits[['circuitId', 'circuitName']],
    on='circuitId',
    how='left'
)



# Columns selection
cols = [
    'raceId',
    'year',
    'name',
    'date',

    'circuitId',
    'circuitName',

    'driverId',
    'forename',
    'surname',
    'dob',

    'constructorId',
    'constructorName',

    'grid',
    'positionOrder',
    'points',
    'laps',

    'fastestLap',
    'fastestLapTime',
    'fastestLapSpeed',

    'qualiPosition',
    'q1',
    'q2',
    'q3',

    'status',
    'statusId'
]

# MAIN DATA FRAME
df = df[cols]

pd.set_option('display.max_columns', None)

print("=== FINAL DATAFRAME ===")
print(df.shape)
print("*" * 30, '\n')

## Data Quality Check

# Missing Values
print("=== MISSING VALUES ===")
print(df.isnull().sum())
print("*" * 30, '\n')

# Duplicates
duplicates = df.duplicated(
    subset=['raceId', 'driverId']
).sum()

print("=== DUPLICATE CHECK ===")
print(f"Duplicate rows: {duplicates}")
print("*" * 30, '\n')

# Basic Statistics
print("=== BASIC STATISTICS ===")
print(df[['grid', 'positionOrder', 'points']].describe())
print("*" * 30, '\n')

# Unique Values after Merge
print("=== UNIQUE Values ===")
print(f"Drivers: {df['driverId'].nunique()}")
print(f"Constructors: {df['constructorId'].nunique()}")
print(f"Races: {df['raceId'].nunique()}")
print("*" * 30, '\n')

### 3. FEATURE ENGINEERING ###

## RACE STATUS FEATURES ##

# Finished
df['finished'] = df['status'].str.contains(
    r'Finished|\+\d+ Laps?',
    regex=True,
    na=False
)

# Mechanical DNFs
mechanicalStatuses = [
    'Wheel', 'Power loss', 'Differential',
    'Fuel system', 'Transmission',
    'Cooling system', 'Tyre', 'Throttle',
    'Brake duct', 'Hydraulics', 'Battery',
    'Puncture', 'Overheating', 'Wheel nut',
    'Vibrations', 'Driveshaft',
    'Fuel pressure', 'Spark plugs',
    'Steering', 'Out of fuel',
    'Radiator', 'Electronics',
    'Water leak', 'Undertray',
    'Fuel pump', 'Fuel leak',
    'ERS', 'Oil pressure',
    'Engine', 'Technical',
    'Gearbox', 'Electrical',
    'Power Unit', 'Brakes',
    'Clutch', 'Exhaust',
    'Water pump', 'Mechanical',
    'Turbo', 'Drivetrain',
    'Suspension', 'Oil leak',
    'Water pressure',
    'Seat', 'Retired'
]

# Incident DNFs
incidentStatuses = [
    'Spun off',
    'Collision',
    'Collision damage',
    'Accident',
    'Damage',
    'Debris',
    'Front wing',
    'Rear wing'
]

# Administrative DNFs
otherStatuses = [
    'Excluded',
    'Disqualified',
    'Withdrew',
    'Illness'
]

# DNF Flags
df['mechanicalDnf'] = (
    df['status']
    .isin(mechanicalStatuses)
)

df['incidentDnf'] = (
    df['status']
    .isin(incidentStatuses)
)

df['otherDnf'] = (
    df['status']
    .isin(otherStatuses)
)

# Overall DNF
df['dnf'] = ~df['finished']

## PERFORMANCE FEATURES ##

# Podium Finish
df['podium'] = (
    df['positionOrder'] <= 3
)

# Points Finish
df['pointsFinish'] = (
    df['points'] > 0
)

# Race Win
df['raceWin'] = (
    df['positionOrder'] == 1
)

# Pole Position
df['polePosition'] = (
    df['grid'] == 1
)

# Front Row Start
df['frontRowStart'] = (
    df['grid'] <= 2
)

# Places Gained
df['placesGained'] = (
    df['grid']
    - df['positionOrder']
)

# Absolute Position Change
df['positionChangeAbs'] = (
    abs(
        df['grid']
        - df['positionOrder']
    )
)

## EXPECTED PERFORMANCE FEATURES ##

# Expected finish by grid slot
expectedFinishByGrid = (
    df[
        df['grid'] > 0
    ]
    .groupby('grid')['positionOrder']
    .mean()
)

# Expected finish position
df['expectedFinish'] = (
    df['grid']
    .map(expectedFinishByGrid)
)

# Performance vs expectation
df['performanceDelta'] = (
    df['expectedFinish']
    - df['positionOrder']
)

## DRIVER FEATURES ##

# Driver age
df['date'] = pd.to_datetime(df['date'])

df['dob'] = pd.to_datetime(df['dob'])

df['driverAge'] = (
    (df['date'] - df['dob']).dt.days
    / 365.25
)

## ANALYTICAL DATASETS ##

# Finished-only races
df_finished = (
    df[df['finished']]
    .copy()
)
##### 6. EDA #####

### 6. EDA ###

edaResults = eda.run_eda(
    df,
    df_finished
)

##### 7. VISUALISATIONS #####

viz.plot_all_visualizations(

    trackAnalysis=
        edaResults['trackAnalysis'],

    df=df,

    trackVolatility=
        edaResults['trackVolatility'],

    seasonCorrelation=
        edaResults['seasonCorrelation'],

    constructorPerformance=
        edaResults['constructorPerformance'],

    constructorReliability=
        edaResults['constructorReliability'],

    driverRacecraft=
        edaResults['driverRacecraft'],

    driverReliability=
        edaResults['driverReliability'],

    circuitAnalysis=
        edaResults['circuitAnalysis'],

    driverRiskProfile=
        edaResults['driverRiskProfile'],

    teammateComparison=
        edaResults['teammateComparison'],

    correlationMatrix=
        edaResults['correlationMatrix'],

    season=2021
)

##### MACHINE LEARNING

ml.run_dnf_classification(df, edaResults)