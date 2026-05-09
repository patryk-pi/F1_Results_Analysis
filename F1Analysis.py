import numpy as np
import pandas as pd

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

## Preprocessing

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

### 3. Feature engineering

# Finished
df['finished'] = (
    df['status'].str.contains(r'Finished|\+\d+ Laps?', regex=True)
)

# DNF
df['dnf'] = ~df['finished']

# Values Check
pd.set_option('display.max_rows', None)
print(df[['status', 'finished']]
      .drop_duplicates()
      .sort_values('finished'))

# Podium Finish
df['podium'] = df['positionOrder'] <= 3

# Points Finish
df['pointsFinish'] = df['points'] > 0

# Places Gained
df['placesGained'] = df['grid'] - df['positionOrder']

# Absolute Position Change
df['positionChangeAbs'] = (
    abs(df['grid'] - df['positionOrder'])
)

# Pole Position
df['polePosition'] = df['grid'] == 1

# Win
df['raceWin'] = df['positionOrder'] == 1

# Front Row Start
df['frontRowStart'] = df['grid'] <= 2

# Age of Driver
df['date'] = pd.to_datetime(df['date'])
df['dob'] = pd.to_datetime(df['dob'])

df['driverAge'] = (
    (df['date'] - df['dob']).dt.days / 365.25
)

# Data Frame for Finishers
df_finished = df[df['finished']]

### 4. EDA

## Grid vs Finish Correlation

# print("=== GRID VS FINISH CORRELATION ===")
#
# correlation = df[['grid', 'positionOrder']].corr()
#
# print(correlation)
# print("*" * 30, '\n')
#
# ## Grid vs Finish Correlation (circuits)
# print("=== GRID VS FINISH CORRELATION ON DIFFERENT CIRCUITS ===")
#
# trackCorrelation = (
#     df.groupby('circuitName')
#     .agg(
#         correlation=('grid', lambda x:
#             x.corr(df.loc[x.index, 'positionOrder'])
#         ),
#         races=('raceId', 'nunique')
#     )
#     .sort_values(by='correlation', ascending=False)
# )
#
# print(trackCorrelation)

### GRID VS FINISH CORRELATION ###

print("=== GRID VS FINISH CORRELATION ===")

# Overall correlation (all results)
overallCorrelation = (
    df[['grid', 'positionOrder']]
    .corr()
    .iloc[0, 1]
)

# Finished-only correlation
finishedCorrelation = (
    df[df['finished']][['grid', 'positionOrder']]
    .corr()
    .iloc[0, 1]
)

print(f"Overall correlation: {overallCorrelation:.3f}")
print(f"Finished-only correlation: {finishedCorrelation:.3f}")

print("*" * 30, '\n')


### GRID VS FINISH CORRELATION BY CIRCUIT ###

print("=== GRID VS FINISH CORRELATION BY CIRCUIT ===")

import matplotlib.pyplot as plt

# Finished-only dataframe
df_finished = df[df['finished']].copy()

# Overall correlation by circuit
overallTrackCorrelation = (
    df.groupby('circuitName')
    .agg(
        overallCorrelation=(
            'grid',
            lambda x: x.corr(
                df.loc[x.index, 'positionOrder']
            )
        ),
        races=('raceId', 'nunique')
    )
)

# Finished-only correlation by circuit
finishedTrackCorrelation = (
    df_finished.groupby('circuitName')
    .agg(
        finishedCorrelation=(
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
        starters=('driverId', 'count'),
        finishers=('finished', 'sum'),
        dnfs=('dnf', 'sum')
    )
)

# DNF rate
dnfStats['dnfRate'] = (
    dnfStats['dnfs']
    / dnfStats['starters']
) * 100

# Merge all analyses

trackCorrelation = (
    overallTrackCorrelation
    .merge(
        finishedTrackCorrelation,
        on='circuitName'
    )
    .merge(
        dnfStats,
        on='circuitName'
    )
)

# Difference between finished-only and overall correlation
trackCorrelation['dnfImpact'] = (
    trackCorrelation['finishedCorrelation']
    - trackCorrelation['overallCorrelation']
)

# Sort by DNF impact
trackCorrelation = (
    trackCorrelation
    .sort_values(
        by='dnfImpact',
        ascending=False
    )
    .round(3)
)

# Print results
print(trackCorrelation)

print("*" * 30, '\n')


### VISUALIZATION ###

# Sort ascending for horizontal plot readability
plot_data = trackCorrelation.sort_values(
    by='dnfImpact',
    ascending=True
)

# Labels with race count
# Labels with correct singular/plural

plot_labels = [
    f"{circuit} ({races} race)"
    if races == 1
    else f"{circuit} ({races} races)"

    for circuit, races in zip(
        plot_data.index,
        plot_data['races']
    )
]

# Figure
plt.figure(figsize=(12, 10))

# Horizontal bar chart
plt.barh(
    plot_labels,
    plot_data['dnfImpact']
)

# Axis labels
plt.xlabel(
    'Finished vs Overall Correlation Difference'
)

plt.ylabel('Circuit')

# Title
plt.title(
    'Impact of DNFs on Grid-to-Finish Correlation'
)

# Layout
plt.tight_layout()

# Show plot
plt.show()