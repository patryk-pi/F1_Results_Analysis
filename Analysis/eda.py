def run_eda(df, df_finished):

    print("=== GRID VS FINISH CORRELATION ===")

    overallCorrelation = (
        df[['grid', 'positionOrder']]
        .corr()
        .iloc[0, 1]
    )

    finishedCorrelation = (
        df_finished[['grid', 'positionOrder']]
        .corr()
        .iloc[0, 1]
    )

    print(
        f"Overall correlation: "
        f"{overallCorrelation:.3f}"
    )

    print(
        f"Finished-only correlation: "
        f"{finishedCorrelation:.3f}"
    )

    print("*" * 30, '\n')


    ### CORRELATION MATRIX ###

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
        .dropna()
        .corr()
        .round(2)
    )


    ### TRACK ANALYSIS ###

    overallTrackCorrelation = (
        df.groupby('circuitName')
        .agg(
            overallCorrelation=(
                'grid',
                lambda x: x.corr(
                    df.loc[
                        x.index,
                        'positionOrder'
                    ]
                )
            ),

            races=('raceId', 'nunique')
        )
    )

    finishedTrackCorrelation = (
        df_finished.groupby('circuitName')
        .agg(
            finishedCorrelation=(
                'grid',
                lambda x: x.corr(
                    df_finished.loc[
                        x.index,
                        'positionOrder'
                    ]
                )
            )
        )
    )

    dnfStats = (
        df.groupby('circuitName')
        .agg(
            starters=('driverId', 'count'),

            finishers=('finished', 'sum'),

            dnfs=('dnf', 'sum')
        )
    )

    dnfStats['dnfRate'] = (
        dnfStats['dnfs']
        / dnfStats['starters']
    ) * 100

    trackAnalysis = (
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

    trackAnalysis['dnfImpact'] = (
        trackAnalysis['finishedCorrelation']
        - trackAnalysis['overallCorrelation']
    )

    trackAnalysis = (
        trackAnalysis
        .sort_values(
            by='dnfImpact',
            ascending=False
        )
        .round(3)
    )


    ### POSITION VOLATILITY ###

    trackVolatility = (
        df.groupby('circuitName')
        .agg(
            avgPositionChange=(
                'positionChangeAbs',
                'mean'
            ),

            races=('raceId', 'nunique')
        )
        .sort_values(
            by='avgPositionChange',
            ascending=False
        )
    )


    ### SEASON CORRELATION ###

    seasonCorrelation = (
        df_finished.groupby('year')
        [['grid', 'positionOrder']]
        .apply(
            lambda x: x['grid'].corr(
                x['positionOrder']
            )
        )
    )


    ### CONSTRUCTOR RELIABILITY ###

    constructorReliability = (
        df.groupby('constructorName')
        .agg(
            mechanicalDnfRate=(
                'mechanicalDnf',
                'mean'
            ),

            incidentDnfRate=(
                'incidentDnf',
                'mean'
            ),

            races=('raceId', 'count')
        )
    )

    constructorReliability[
        [
            'mechanicalDnfRate',
            'incidentDnfRate'
        ]
    ] *= 100

    constructorReliability = (
        constructorReliability
        .round(2)
    )


    ### CONSTRUCTOR PERFORMANCE ###

    constructorPerformance = (
        df_finished.groupby(
            ['year', 'constructorName']
        )
        .agg(
            avgFinish=(
                'positionOrder',
                'mean'
            ),

            points=('points', 'sum')
        )
        .reset_index()
    )


    ### DRIVER RACECRAFT ###

    driverRacecraft = (
        df_finished.groupby('surname')
        .agg(
            avgGrid=('grid', 'mean'),

            avgFinish=(
                'positionOrder',
                'mean'
            ),

            races=('raceId', 'count')
        )
    )

    driverRacecraft = driverRacecraft[
        driverRacecraft['races'] >= 20
    ]

    driverRacecraft['racecraftScore'] = (
        driverRacecraft['avgGrid']
        - driverRacecraft['avgFinish']
    )

    driverRacecraft = (
        driverRacecraft
        .sort_values(
            by='racecraftScore',
            ascending=False
        )
        .round(2)
    )


    ### DRIVER RELIABILITY ###

    driverReliability = (
        df.groupby('surname')
        .agg(
            races=('raceId', 'count'),

            finishRate=('finished', 'mean'),

            mechanicalDnfRate=(
                'mechanicalDnf',
                'mean'
            ),

            incidentDnfRate=(
                'incidentDnf',
                'mean'
            )
        )
    )

    driverReliability = (
        driverReliability[
            driverReliability['races'] >= 20
        ]
    )

    driverReliability[
        [
            'finishRate',
            'mechanicalDnfRate',
            'incidentDnfRate'
        ]
    ] *= 100

    driverReliability = (
        driverReliability
        .sort_values(
            by='incidentDnfRate',
            ascending=False
        )
        .round(2)
    )


    ### DRIVER RISK PROFILE ###

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

            finishRate=('finished', 'mean')
        )
    )

    driverRiskProfile = (
        driverRiskProfile[
            driverRiskProfile['races'] >= 20
        ]
    )

    driverRiskProfile[
        [
            'incidentDnfRate',
            'finishRate'
        ]
    ] *= 100

    driverRiskProfile = (
        driverRiskProfile
        .sort_values(
            by='avgPerformanceDelta',
            ascending=False
        )
        .round(2)
    )


    ### CIRCUIT ANALYSIS ###

    circuitAnalysis = (
        df.groupby('circuitName')
        .agg(
            starters=('driverId', 'count'),

            races=('raceId', 'nunique'),

            mechanicalDnfs=(
                'mechanicalDnf',
                'sum'
            ),

            incidentDnfs=(
                'incidentDnf',
                'sum'
            ),

            finishers=('finished', 'sum')
        )
    )

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


    ### TEAMMATE COMPARISON ###

    teammateComparison = (
        df.groupby(
            [
                'year',
                'constructorName',
                'surname'
            ]
        )
        .agg(
            avgFinish=(
                'positionOrder',
                'mean'
            ),

            medianFinish=(
                'positionOrder',
                'median'
            ),

            avgGrid=('grid', 'mean'),

            races=('raceId', 'count')
        )
        .reset_index()
    )

    teammateComparison = (
        teammateComparison[
            teammateComparison['races'] >= 8
        ]
    )

    teammateComparison['positionsGained'] = (
        teammateComparison['avgGrid']
        - teammateComparison['avgFinish']
    )

    teamAverage = (
        teammateComparison.groupby(
            ['year', 'constructorName']
        )['avgFinish']
        .transform('mean')
    )

    teammateComparison['vsTeammate'] = (
        teamAverage
        - teammateComparison['avgFinish']
    )

    teammateComparison = (
        teammateComparison
        .round(2)
    )

    teammateComparison = (
        teammateComparison.rename(
            columns={
                'surname': 'driver'
            }
        )
    )

    teammateComparison = (
        teammateComparison.sort_values(
            [
                'year',
                'constructorName',
                'avgFinish'
            ]
        )
    )

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

    teammateComparison = (
        teammateComparison
        .reset_index(drop=True)
    )


    return {

        'correlationMatrix': correlationMatrix,

        'trackAnalysis': trackAnalysis,

        'trackVolatility': trackVolatility,

        'seasonCorrelation': seasonCorrelation,

        'constructorReliability':
            constructorReliability,

        'constructorPerformance':
            constructorPerformance,

        'driverRacecraft':
            driverRacecraft,

        'driverReliability':
            driverReliability,

        'driverRiskProfile':
            driverRiskProfile,

        'circuitAnalysis':
            circuitAnalysis,

        'teammateComparison':
            teammateComparison
    }