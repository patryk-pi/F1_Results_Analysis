# analysis/visualizations.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


### DNF IMPACT BY CIRCUIT ###

def plot_dnf_impact_by_circuit(trackAnalysis):

    plot_data = trackAnalysis.sort_values(
        by='dnfImpact',
        ascending=True
    )

    plot_labels = [
        f"{circuit} ({races} race)"
        if races == 1
        else f"{circuit} ({races} races)"

        for circuit, races in zip(
            plot_data.index,
            plot_data['races']
        )
    ]

    plt.figure(figsize=(12, 10))

    plt.barh(
        plot_labels,
        plot_data['dnfImpact']
    )

    plt.xlabel(
        'Finished vs Overall Correlation Difference'
    )

    plt.ylabel('Circuit')

    plt.title(
        'Impact of DNFs on Grid-to-Finish Correlation'
    )

    plt.tight_layout()

    plt.show()

### CORRELATION HEATMAP ###

def plot_correlation_heatmap(correlationMatrix):

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlationMatrix,
        annot=True,
        cmap='coolwarm',
        center=0
    )

    plt.title(
        'Feature Correlation Matrix'
    )

    plt.tight_layout()

    plt.show()


### FINISHED-ONLY CORRELATION BY CIRCUIT ###

def plot_finished_correlation_by_circuit(trackAnalysis):

    plot_data = trackAnalysis.sort_values(
        by='finishedCorrelation',
        ascending=True
    )

    plot_labels = [
        f"{circuit} ({races} races)"

        for circuit, races in zip(
            plot_data.index,
            plot_data['races']
        )
    ]

    plt.figure(figsize=(12, 10))

    plt.barh(
        plot_labels,
        plot_data['finishedCorrelation']
    )

    plt.xlabel('Grid vs Finish Correlation')

    plt.ylabel('Circuit')

    plt.title(
        'How Strongly Qualifying Determines Race Result'
    )

    plt.tight_layout()

    plt.show()


### GRID VS FINISH POSITION DENSITY ###

def plot_grid_vs_finish_density(df):

    heatmap_df = df[
        (df['grid'] > 0) &
        (df['grid'] <= 20) &
        (df['positionOrder'] <= 20)
    ]

    heatmapData = pd.crosstab(
        heatmap_df['positionOrder'],
        heatmap_df['grid']
    )

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        heatmapData,
        cmap='inferno'
    )

    plt.title("Grid Position vs Finish Position")

    plt.xlabel("Grid Position")

    plt.ylabel("Finish Position")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.show()


### WIN RATE ###

def plot_win_rate(df):

    winRate = (
        df.groupby('grid')['positionOrder']
          .apply(lambda x: (x == 1).mean() * 100)
          .sort_index()
    )

    winRate = winRate[
        winRate.index > 0
    ]

    plt.figure(figsize=(10, 6))

    winRate.plot(kind='bar')

    plt.title("Win Rate by Grid Position")

    plt.xlabel("Grid Position")

    plt.ylabel("Win Rate (%)")

    plt.tight_layout()

    plt.show()


### POSITION VOLATILITY BY CIRCUIT ###

def plot_position_volatility(trackVolatility):

    plot_data = trackVolatility.sort_values(
        by='avgPositionChange',
        ascending=True
    )

    plot_labels = [
        f"{circuit} ({races} races)"

        for circuit, races in zip(
            plot_data.index,
            plot_data['races']
        )
    ]

    plt.figure(figsize=(12, 10))

    plt.barh(
        plot_labels,
        plot_data['avgPositionChange']
    )

    plt.xlabel(
        'Average Absolute Position Change'
    )

    plt.ylabel('Circuit')

    plt.title(
        'Race Position Volatility by Circuit'
    )

    plt.tight_layout()

    plt.show()

### POSITION CHANGE DISTRIBUTION BY CIRCUIT ###

def plot_position_change_boxplot(df):

    # Minimum sample size
    circuitCounts = (
        df.groupby('circuitName')['raceId']
        .nunique()
    )

    validCircuits = circuitCounts[
        circuitCounts >= 5
    ].index

    # Filter dataframe
    plot_df = df[
        df['circuitName']
        .isin(validCircuits)
    ]

    # Sort by median volatility
    circuitOrder = (
        plot_df.groupby('circuitName')['positionChangeAbs']
        .median()
        .sort_values()
        .index
    )

    plt.figure(figsize=(16, 8))

    sns.boxplot(
        data=plot_df,
        x='circuitName',
        y='positionChangeAbs',
        order=circuitOrder
    )

    plt.xticks(
        rotation=45,
        ha='right'
    )

    plt.xlabel('Circuit')

    plt.ylabel(
        'Absolute Position Change'
    )

    plt.title(
        'Race Position Volatility by Circuit'
    )

    plt.tight_layout()

    plt.show()


### QUALIFYING IMPORTANCE OVER TIME ###

def plot_qualifying_importance_over_time(
    seasonCorrelation
):

    plt.figure(figsize=(10, 6))

    plt.plot(
        seasonCorrelation.index,
        seasonCorrelation.values,
        marker='o'
    )

    plt.xlabel('Season')

    plt.ylabel(
        'Grid vs Finish Correlation'
    )

    plt.title(
        'Importance of Qualifying Over Time'
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.show()

### CONSTRUCTOR RELIABILITY ###

def plot_constructor_reliability(
    constructorReliability
):

    plot_data = constructorReliability.sort_values(
        by='mechanicalDnfRate',
        ascending=True
    )

    plt.figure(figsize=(12, 8))

    plt.barh(
        plot_data.index,
        plot_data['mechanicalDnfRate'],
        label='Mechanical DNF Rate'
    )

    plt.barh(
        plot_data.index,
        plot_data['incidentDnfRate'],
        left=plot_data['mechanicalDnfRate'],
        label='Incident DNF Rate'
    )

    plt.xlabel('DNF Rate (%)')

    plt.ylabel('Constructor')

    plt.title(
        'Constructor Reliability and Incident Exposure'
    )

    plt.legend()

    plt.tight_layout()

    plt.show()

### CONSTRUCTOR PERFORMANCE TREND ###

def plot_constructor_performance_trend(
    constructorPerformance
):

    topConstructors = (
        constructorPerformance.groupby(
            'constructorName'
        )['points']
        .sum()
        .sort_values(ascending=False)
        .head(6)
        .index
    )

    plot_df = constructorPerformance[
        constructorPerformance['constructorName']
        .isin(topConstructors)
    ]

    plt.figure(figsize=(12, 8))

    sns.lineplot(
        data=plot_df,
        x='year',
        y='avgFinish',
        hue='constructorName',
        marker='o'
    )

    plt.gca().invert_yaxis()

    plt.title(
        'Constructor Performance Over Time'
    )

    plt.ylabel('Average Finish Position')

    plt.tight_layout()

    plt.show()


### DRIVER RACECRAFT ANALYSIS ###

def plot_driver_racecraft(driverRacecraft):

    plot_data = (
        driverRacecraft
        .head(15)
        .sort_values(
            by='racecraftScore'
        )
    )

    plt.figure(figsize=(10, 8))

    plt.barh(
        plot_data.index,
        plot_data['racecraftScore']
    )

    plt.xlabel('Racecraft Score')

    plt.ylabel('Driver')

    plt.title(
        'Drivers Gaining Most Positions Relative to Grid'
    )

    plt.tight_layout()

    plt.show()


### DRIVER INCIDENT DNF RATE ###

def plot_driver_incident_dnf_rate(
    driverReliability
):

    plot_data = (
        driverReliability
        .sort_values(
            by='incidentDnfRate',
            ascending=True
        )
        .tail(15)
    )

    plt.figure(figsize=(12, 8))

    plt.barh(
        plot_data.index,
        plot_data['incidentDnfRate']
    )

    plt.xlabel('Incident DNF Rate (%)')

    plt.ylabel('Driver')

    plt.title(
        'Drivers Most Frequently Involved in Incidents'
    )

    plt.tight_layout()

    plt.show()


### MOST CHAOTIC CIRCUITS ###

def plot_most_chaotic_circuits(
    circuitAnalysis
):

    plot_data = (
        circuitAnalysis
        .sort_values(
            by='incidentDnfRate',
            ascending=True
        )
    )

    plot_labels = [
        f"{circuit} ({races} races)"

        for circuit, races in zip(
            plot_data.index,
            plot_data['races']
        )
    ]

    plt.figure(figsize=(12, 10))

    plt.barh(
        plot_labels,
        plot_data['incidentDnfRate']
    )

    plt.xlabel('Incident DNF Rate (%)')

    plt.ylabel('Circuit')

    plt.title(
        'Most Incident-Prone Circuits'
    )

    plt.tight_layout()

    plt.show()


### DRIVER RISK VS REWARD PROFILE ###

def plot_driver_risk_vs_reward(
    driverRiskProfile
):

    plot_data = driverRiskProfile.copy()

    plt.figure(figsize=(14, 10))

    plt.scatter(
        plot_data['incidentDnfRate'],
        plot_data['avgPerformanceDelta']
    )

    for driver in plot_data.index:

        plt.text(
            plot_data.loc[
                driver,
                'incidentDnfRate'
            ],

            plot_data.loc[
                driver,
                'avgPerformanceDelta'
            ],

            driver,
            fontsize=8
        )

    plt.xlabel('Incident DNF Rate (%)')

    plt.ylabel(
        'Average Performance Delta'
    )

    plt.title(
        'Driver Risk vs Reward Profile'
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.show()


### TEAMMATE COMPARISON ###

def plot_teammate_comparison(
    teammateComparison,
    season
):

    comparisonSeason = teammateComparison[
        teammateComparison['year'] == season
    ]

    plt.figure(figsize=(14, 8))

    sns.barplot(
        data=comparisonSeason,
        x='constructorName',
        y='avgFinish',
        hue='driver'
    )

    plt.title(
        f'Average Finish Position by Team - {season}'
    )

    plt.xlabel('Constructor')

    plt.ylabel(
        'Average Finish Position'
    )

    # Lower finish position = better
    plt.gca().invert_yaxis()

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()


def plot_all_visualizations(
    trackAnalysis,
    df,
    trackVolatility,
    seasonCorrelation,
    constructorPerformance,
    constructorReliability,
    driverRacecraft,
    driverReliability,
    circuitAnalysis,
    driverRiskProfile,
    teammateComparison,
    correlationMatrix,
    season=2021

):

    plot_dnf_impact_by_circuit(
        trackAnalysis
    )

    plot_finished_correlation_by_circuit(
        trackAnalysis
    )

    plot_grid_vs_finish_density(
        df
    )

    plot_correlation_heatmap(
        correlationMatrix
    )

    plot_win_rate(
        df
    )

    plot_position_volatility(
        trackVolatility
    )

    plot_qualifying_importance_over_time(
        seasonCorrelation
    )

    plot_constructor_reliability(
        constructorReliability
    )

    plot_constructor_performance_trend(
        constructorPerformance
    )

    plot_driver_racecraft(
        driverRacecraft
    )

    plot_driver_incident_dnf_rate(
        driverReliability
    )

    plot_most_chaotic_circuits(
        circuitAnalysis
    )

    plot_driver_risk_vs_reward(
        driverRiskProfile
    )

    plot_teammate_comparison(
        teammateComparison,
        season
    )

    plot_position_change_boxplot (
        df
    )

