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

races_filtered = races[
    (races['year'] >= 2014) &
    (races['year'] <= 2024)
]

print(races_filtered)