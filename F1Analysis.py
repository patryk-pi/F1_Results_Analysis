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

# Podium finish
df['podium'] = df['positionOrder'] <= 3

# Points finish
df['pointsFinish'] = df['points'] > 0

# Places gained
df['placesGained'] = df['grid'] - df['positionOrder']

# Pole Position
df['polePosition'] = df['grid'] == 1

# Win
df['raceWin'] = df['positionOrder'] == 1

# Front Row Start
df['frontRowStart'] = df['grid'] <= 2

# Age of driver
df['date'] = pd.to_datetime(df['date'])
df['dob'] = pd.to_datetime(df['dob'])

df['driverAge'] = (
    (df['date'] - df['dob']).dt.days / 365.25
)

### 4. EDA
