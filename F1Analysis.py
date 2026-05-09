import numpy as np
import pandas as pd

### 1. DATA LOAD ###

results = pd.read_csv("data/results.csv")
races = pd.read_csv("data/races.csv")
drivers = pd.read_csv("data/drivers.csv")
constructors = pd.read_csv("data/constructors.csv")
qualifying = pd.read_csv("data/qualifying.csv")
status = pd.read_csv("data/status.csv")

## Rows Check
print("=== ORIGINAL DATA ===")
print(f"Results rows: {results.shape[0]}")
print(f"Races rows: {races.shape[0]}")
print(f"Drivers rows: {drivers.shape[0]}")
print(f"Constructors rows: {constructors.shape[0]}")
print(f"Qualifying rows: {qualifying.shape[0]}")
print(f"Status rows: {status.shape[0]}")
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

## Data Frame Build

# Merge - Races
df = results_filtered.merge(
    races_filtered[['raceId', 'year', 'name', 'date']],
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

# Columns selection
cols = [
    'raceId',
    'year',
    'name',
    'date',

    'driverId',
    'forename',
    'surname',

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

df = df[cols]

pd.set_option('display.max_columns', None)

print("=== FINAL DATAFRAME ===")
print(df.shape)
print("*" * 30, '\n')

### 3. EDA


