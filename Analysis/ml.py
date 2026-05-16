import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score
)

import matplotlib.pyplot as plt
import seaborn as sns


def run_dnf_classification(
        df,
        edaResults
):

    print("\n" + "=" * 50)
    print("DNF CLASSIFICATION MODELS")
    print("=" * 50)

    ml_df = df.copy()

    ### REMOVE INVALID GRID ###

    ml_df = ml_df[
        ml_df['grid'] > 0
    ]

    ### ========================================= ###
    ### FEATURE ENGINEERING FROM EDA RESULTS ###

    # CIRCUIT DNF RATE

    circuitDnfRates = (
        edaResults['circuitAnalysis']
        ['incidentDnfRate']
        .to_dict()
    )

    ml_df['circuitIncidentDnfRate'] = (
        ml_df['circuitName']
        .map(circuitDnfRates)
    )

    # CONSTRUCTOR RELIABILITY

    constructorReliability = (
        edaResults['constructorReliability']
        ['mechanicalDnfRate']
        .to_dict()
    )

    ml_df['constructorMechanicalDnfRate'] = (
        ml_df['constructorName']
        .map(constructorReliability)
    )

    # DRIVER DNF HISTORY

    driverDnfHistory = (
        edaResults['driverReliability']
        ['mechanicalDnfRate']
        .to_dict()
    )

    ml_df['driverMechanicalDnfRate'] = (
        ml_df['surname']
        .map(driverDnfHistory)
    )

    ### ========================================= ###
    ### FEATURES ###

    features = [

        # Base Features
        'grid',
        'qualiPosition',
        'driverAge',

        # Categorical
        'constructorName',
        'circuitName',

        # Engineered Features
        'circuitIncidentDnfRate',
        'constructorMechanicalDnfRate',
        'driverMechanicalDnfRate'
    ]

    target = 'dnf'

    ### REMOVE MISSING ###

    ml_df = ml_df.dropna(
        subset=features + [target]
    )

    X = ml_df[features]

    y = ml_df[target]

    print(f"\nDataset size: {ml_df.shape[0]}")

    print(
        f"DNF rate: "
        f"{y.mean() * 100:.2f}%"
    )

    ### ========================================= ###
    ### FEATURE GROUPS ###

    categorical_features = [
        'constructorName',
        'circuitName'
    ]

    numeric_features = [

        'grid',
        'qualiPosition',
        'driverAge',

        'circuitIncidentDnfRate',
        'constructorMechanicalDnfRate',
        'driverMechanicalDnfRate'
    ]

    ### ========================================= ###
    ### PREPROCESSOR ###

    preprocessor = ColumnTransformer(
        transformers=[

            (
                'cat',

                OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False
                ),

                categorical_features
            ),

            (
                'num',
                StandardScaler(),
                numeric_features
            )
        ]
    )

    ### ========================================= ###
    ### TRAIN / TEST SPLIT ###

    X_train, X_test, y_train, y_test = (
        train_test_split(

            X,
            y,

            test_size=0.2,

            random_state=42,

            stratify=y
        )
    )

    print("\nTrain/Test split:")

    print(
        f"Train rows: "
        f"{X_train.shape[0]}"
    )

    print(
        f"Test rows: "
        f"{X_test.shape[0]}"
    )

    ### ========================================= ###
    ### MODELS ###

    models = {

        'Logistic Regression':

            LogisticRegression(
                class_weight='balanced',
                max_iter=2000,
                random_state=42
            ),

        'Random Forest':

            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),

        'Histogram Gradient Boosting':

            HistGradientBoostingClassifier(
                class_weight = 'balanced',
                max_iter=300,
                learning_rate=0.03,
                max_depth=8,
                random_state=42
            )
    }

    ### ========================================= ###
    ### TRAINING + EVALUATION ###

    results = []

    trainedPipelines = {}

    for modelName, model in models.items():

        print("\n" + "-" * 50)

        print(modelName)

        print("-" * 50)

        pipeline = Pipeline([

            (
                'preprocessor',
                preprocessor
            ),

            (
                'model',
                model
            )
        ])

        ### TRAIN ###

        pipeline.fit(
            X_train,
            y_train
        )

        ### PREDICT ###

        y_pred = pipeline.predict(X_test)

        ### PROBABILITIES ###

        if hasattr(
                pipeline.named_steps['model'],
                'predict_proba'
        ):

            y_prob = (
                pipeline.predict_proba(X_test)
                [:, 1]
            )

            roc_auc = roc_auc_score(
                y_test,
                y_prob
            )

        else:

            roc_auc = np.nan

        ### STORE ###

        trainedPipelines[modelName] = pipeline

        ### METRICS ###

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        f1_macro = f1_score(
            y_test,
            y_pred,
            average='macro'
        )

        f1_weighted = f1_score(
            y_test,
            y_pred,
            average='weighted'
        )

        results.append({

            'Model': modelName,

            'Accuracy': accuracy,

            'F1 Macro': f1_macro,

            'F1 Weighted': f1_weighted,

            'ROC AUC': roc_auc
        })

        ### PRINT ###

        print(
            f"\nAccuracy: "
            f"{accuracy:.4f}"
        )

        print(
            f"F1 Macro: "
            f"{f1_macro:.4f}"
        )

        print(
            f"F1 Weighted: "
            f"{f1_weighted:.4f}"
        )

        if not np.isnan(roc_auc):

            print(
                f"ROC AUC: "
                f"{roc_auc:.4f}"
            )

        print("\nClassification Report:")

        print(
            classification_report(
                y_test,
                y_pred
            )
        )

    ### ========================================= ###
    ### RESULTS TABLE ###

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            by='F1 Macro',
            ascending=False
        )
        .round(4)
    )

    print("\n" + "=" * 50)

    print("MODEL COMPARISON")

    print("=" * 50)

    print(results_df)

    ### ========================================= ###
    ### BEST MODEL ###

    bestModelName = (
        results_df.iloc[0]['Model']
    )

    bestPipeline = (
        trainedPipelines[bestModelName]
    )

    bestPredictions = (
        bestPipeline.predict(X_test)
    )

    print("\nBest model:")

    print(bestModelName)

    ### ========================================= ###
    ### CONFUSION MATRIX ###

    cm = confusion_matrix(
        y_test,
        bestPredictions
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title(
        f'Confusion Matrix - '
        f'{bestModelName}'
    )

    plt.xlabel('Predicted')

    plt.ylabel('Actual')

    plt.tight_layout()

    plt.show()

    ### ========================================= ###
    ### FEATURE IMPORTANCE ###

    bestModel = (
        bestPipeline.named_steps['model']
    )

    featureNames = (
        bestPipeline
        .named_steps['preprocessor']
        .get_feature_names_out()
    )

    if hasattr(bestModel, 'feature_importances_'):

        importanceValues = (
            bestModel.feature_importances_
        )

    elif hasattr(bestModel, 'coef_'):

        importanceValues = np.abs(
            bestModel.coef_[0]
        )

    else:

        importanceValues = None

    if importanceValues is not None:

        importanceDf = pd.DataFrame({

            'feature': featureNames,

            'importance': importanceValues
        })

        importanceDf = (
            importanceDf
            .sort_values(
                by='importance',
                ascending=False
            )
            .head(20)
        )

        plt.figure(figsize=(12, 8))

        sns.barplot(

            data=importanceDf,

            x='importance',

            y='feature'
        )

        plt.title(
            f'Feature Importance - '
            f'{bestModelName}'
        )

        plt.tight_layout()

        plt.show()

        print("\nTop Important Features:")

        print(importanceDf)

    ### ========================================= ###

    print("\n" + "=" * 50)

    print("ML ANALYSIS FINISHED")

    print("=" * 50)

    return {

        'results': results_df,

        'bestModel': bestModelName,

        'pipelines': trainedPipelines
    }