import numpy as np
import pandas as pd

### 1. DATA LOAD ###

results = pd.read_csv("data/results.csv")
races = pd.read_csv("data/races.csv")
drivers = pd.read_csv("data/drivers.csv")
constructors = pd.read_csv("data/constructors.csv")
qualifying = pd.read_csv("data/qualifying.csv")
status = pd.read_csv("data/status.csv")

### 2. DATA PREP ###

## Data Filter
races_filtered = races[
    (races['year'] >= 2014) &
    (races['year'] <= 2024)
]

hybrid_race_ids = races_filtered['raceId']

results_filtered = results[
    results['raceId'].isin(hybrid_race_ids)
]

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
    constructors[['constructorId', 'name']],
    on='constructorId',
    how='left',
    suffixes=('', '_constructor')
)

# Merge - Qualifying
df = df.merge(
    qualifying[['raceId', 'driverId', 'position', 'q1', 'q2', 'q3']],
    on=['raceId', 'driverId'],
    how='left',
    suffixes=('', '_quali')
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
    'name_constructor',

    'grid',
    'positionOrder',
    'points',
    'laps',

    'fastestLap',
    'fastestLapTime',
    'fastestLapSpeed',

    'position_quali',
    'q1',
    'q2',
    'q3',

    'status',
    'statusId'
]

pd.set_option('display.max_columns', None)
df = df[cols]
print(df)
