import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import Analysis.visualisations as viz

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
### 6. EDA ###

### OVERALL GRID VS FINISH CORRELATION ###

print("=== GRID VS FINISH CORRELATION ===")

# Overall correlation (all drivers)
overallCorrelation = (
    df[['grid', 'positionOrder']]
    .corr()
    .iloc[0, 1]
)

# Finished-only correlation
finishedCorrelation = (
    df_finished[['grid', 'positionOrder']]
    .corr()
    .iloc[0, 1]
)

print(f"Overall correlation: {overallCorrelation:.3f}")
print(f"Finished-only correlation: {finishedCorrelation:.3f}")

print("*" * 30, '\n')

### NUMERICAL CORRELATION MATRIX ###

correlationFeatures = [
    'grid',
    'positionOrder',
    'laps',
    'driverAge',
    'placesGained',
    'performanceDelta'
]

correlationMatrix = (
    df[correlationFeatures]
    .corr()
    .round(2)
)

### CIRCUIT-LEVEL CORRELATION ANALYSIS ###

print("=== GRID VS FINISH CORRELATION BY CIRCUIT ===")

# Overall correlation by circuit
overallTrackCorrelation = (
    df.groupby('circuitName')
    .agg(
        overallCorrelation = (
            'grid',
            lambda x: x.corr(
                df.loc[x.index, 'positionOrder']
            )
        ),
        races = ('raceId', 'nunique')
    )
)

# Finished-only correlation by circuit
finishedTrackCorrelation = (
    df_finished.groupby('circuitName')
    .agg(
        finishedCorrelation = (
            'grid',
            lambda x: x.corr(
                df_finished.loc[x.index, 'positionOrder']
            )
        )
    )
)

# DNF statistics by circuit
dnfStats = (
    df.groupby('circuitName')
    .agg(
        starters = ('driverId', 'count'),
        finishers = ('finished', 'sum'),
        dnfs = ('dnf', 'sum')
    )
)

# DNF rate (%)
dnfStats['dnfRate'] = (
                              dnfStats['dnfs']
                              / dnfStats['starters']
                      ) * 100

# Merge all analyses
trackAnalysis = (
    overallTrackCorrelation
    .merge(
        finishedTrackCorrelation,
        on = 'circuitName'
    )
    .merge(
        dnfStats,
        on = 'circuitName'
    )
)

# DNF impact on correlation
trackAnalysis['dnfImpact'] = (
        trackAnalysis['finishedCorrelation']
        - trackAnalysis['overallCorrelation']
)

# Final formatting
trackAnalysis = (
    trackAnalysis
    .sort_values(
        by = 'dnfImpact',
        ascending = False
    )
    .round(3)
)

# Print results
print(trackAnalysis)

print("*" * 30, '\n')

### POSITION VOLATILITY BY CIRCUIT ###

trackVolatility = (
    df.groupby('circuitName')
    .agg(
        avgPositionChange = (
            'positionChangeAbs',
            'mean'
        ),
        races = ('raceId', 'nunique')
    )
    .sort_values(
        by = 'avgPositionChange',
        ascending = False
    )
)

print(trackVolatility)



### QUALIFYING IMPORTANCE OVER TIME ###

seasonCorrelation = (
    df_finished.groupby('year')
    .apply(
        lambda x: x['grid'].corr(
            x['positionOrder']
        )
    )
)

print(seasonCorrelation)

### CONSTRUCTOR RELIABILITY ###

constructorReliability = (
    df.groupby('constructorName')
    .agg(
        mechanicalDnfRate=('mechanicalDnf', 'mean'),
        incidentDnfRate=('incidentDnf', 'mean'),
        races=('raceId', 'count')
    )
)

constructorReliability[
    ['mechanicalDnfRate', 'incidentDnfRate']
] *= 100

### CONSTRUCTOR PERFORMANCE OVER TIME ###

constructorPerformance = (
    df_finished.groupby(
        ['year', 'constructorName']
    )
    .agg(
        avgFinish=('positionOrder', 'mean'),
        points=('points', 'sum')
    )
    .reset_index()
)

### DRIVER RACECRAFT ANALYSIS ###

driverRacecraft = (
    df_finished.groupby('surname')
    .agg(
        avgGrid=('grid', 'mean'),
        avgFinish=('positionOrder', 'mean'),
        races=('raceId', 'count')
    )
)

# Minimum sample size
driverRacecraft = driverRacecraft[
    driverRacecraft['races'] >= 20
]

# Racecraft score
driverRacecraft['racecraftScore'] = (
    driverRacecraft['avgGrid']
    - driverRacecraft['avgFinish']
)

# Sort
driverRacecraft = (
    driverRacecraft
    .sort_values(
        by='racecraftScore',
        ascending=False
    )
    .round(2)
)

print(driverRacecraft)

### DRIVER DNF Rate ANALYSIS ###

driverReliability = (
    df.groupby('surname')
    .agg(
        races=('raceId', 'count'),

        finishRate=('finished', 'mean'),

        mechanicalDnfRate=('mechanicalDnf', 'mean'),

        incidentDnfRate=('incidentDnf', 'mean')
    )
)

## Minimum sample size
driverReliability = driverReliability[
    driverReliability['races'] >= 20
]

## Convert to percentages
driverReliability[
    [
        'finishRate',
        'mechanicalDnfRate',
        'incidentDnfRate'
    ]
] *= 100


driverReliability = (
    driverReliability.sort_values(
        by='incidentDnfRate',
        ascending=False
    )
    .round(2)
)

print(driverReliability)

### DRIVER RISK VS REWARD PROFILE ###

driverRiskProfile = (
    df.groupby('surname')
    .agg(
        races=('raceId', 'count'),

        avgPerformanceDelta=(
            'performanceDelta',
            'mean'
        ),

        incidentDnfRate=(
            'incidentDnf',
            'mean'
        ),

        finishRate=(
            'finished',
            'mean'
        )
    )
)

# Minimum sample size
driverRiskProfile = driverRiskProfile[
    driverRiskProfile['races'] >= 20
]

# Convert to percentages
driverRiskProfile[
    [
        'incidentDnfRate',
        'finishRate'
    ]
] *= 100

# Sorting
driverRiskProfile = (
    driverRiskProfile
    .sort_values(
        by='avgPerformanceDelta',
        ascending=False
    )
    .round(2)
)

print(driverRiskProfile)

### CIRCUIT DNF ANALYSIS ###

circuitAnalysis = (
    df.groupby('circuitName')
    .agg(
        starters=('driverId', 'count'),

        races=('raceId', 'nunique'),

        mechanicalDnfs=('mechanicalDnf', 'sum'),

        incidentDnfs=('incidentDnf', 'sum'),

        finishers=('finished', 'sum')
    )
)

# Rates
circuitAnalysis['mechanicalDnfRate'] = (
    circuitAnalysis['mechanicalDnfs']
    / circuitAnalysis['starters']
) * 100

circuitAnalysis['incidentDnfRate'] = (
    circuitAnalysis['incidentDnfs']
    / circuitAnalysis['starters']
) * 100

circuitAnalysis['finishRate'] = (
    circuitAnalysis['finishers']
    / circuitAnalysis['starters']
) * 100

circuitAnalysis = (
    circuitAnalysis
    .sort_values(
        by='incidentDnfRate',
        ascending=False
    )
    .round(2)
)

print(circuitAnalysis)


### TEAMMATE COMPARISON TABLE ###

teammateComparison = (
    df.groupby(
        ['year', 'constructorName', 'surname']
    )
    .agg(
        avgFinish=('positionOrder', 'mean'),
        medianFinish=('positionOrder', 'median'),
        avgGrid=('grid', 'mean'),
        races=('raceId', 'count')
    )
    .reset_index()
)

# Remove small samples
teammateComparison = teammateComparison[
    teammateComparison['races'] >= 8
]

# Position gain relative to starting position
teammateComparison['positionsGained'] = (
    teammateComparison['avgGrid']
    - teammateComparison['avgFinish']
)

# Difference to teammate average
teamAverage = (
    teammateComparison.groupby(
        ['year', 'constructorName']
    )['avgFinish']
    .transform('mean')
)

# Positive = better than teammate
teammateComparison['vsTeammate'] = (
    teamAverage
    - teammateComparison['avgFinish']
)

# Round values
teammateComparison = teammateComparison.round(2)

# Better readability
teammateComparison = teammateComparison.rename(
    columns={
        'surname': 'driver'
    }
)

# Sort INSIDE seasons and teams
teammateComparison = teammateComparison.sort_values(
    ['year', 'constructorName', 'avgFinish']
)

# Reorder columns
teammateComparison = teammateComparison[
    [
        'year',
        'constructorName',
        'driver',
        'avgFinish',
        'medianFinish',
        'avgGrid',
        'positionsGained',
        'vsTeammate',
        'races'
    ]
]

# Reset index for clean display
teammateComparison = teammateComparison.reset_index(drop=True)

print(teammateComparison)

##### 7. VISUALISATIONS #####

viz.plot_all_visualizations(
    trackAnalysis=trackAnalysis,
    df=df,
    trackVolatility=trackVolatility,
    seasonCorrelation=seasonCorrelation,
    constructorReliability=constructorReliability,
    constructorPerformance=constructorPerformance,
    driverRacecraft=driverRacecraft,
    driverReliability=driverReliability,
    circuitAnalysis=circuitAnalysis,
    driverRiskProfile=driverRiskProfile,
    teammateComparison=teammateComparison,
    season=2021,
    correlationMatrix = correlationMatrix
)