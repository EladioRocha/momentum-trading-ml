import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV

def create_model(model_name):
    if model_name == "random_forest":
        clf = create_random_forest_model()
    elif model_name == "svc":
        clf = create_svc_model()
    elif model_name == "logistic_regression":
        clf = logistic_regression_model()
    elif model_name == "neural_network":
        return create_neural_network_model()
    
    clf = BaggingClassifier(base_estimator=clf, n_estimators=100, n_jobs=-1, verbose=0)
    return clf

def create_random_forest_model():
    clf = RandomForestClassifier()
    return clf

def create_svc_model():
    clf = SVC(
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="auto",
        coef0=0.0,
        shrinking=True,
        probability=True,
        tol=0.001,
        cache_size=200,
        class_weight="balanced",
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        random_state=None,
    )
    return clf

def logistic_regression_model():
    clf = LogisticRegression(
        penalty="l2",
        dual=False,
        tol=0.0001,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="liblinear",
        max_iter=100,
        multi_class="ovr",
        verbose=0,
        warm_start=False,
        n_jobs=1,
    )
    return clf

def create_neural_network_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def walkforward_with_purging_and_embargos(
    model,
    X,
    y,
    purged_window_size=0,
    embargo_period=0,
    lookback=50,
    test_size=10,
    scaler=None,
    scorer=None,
    model_name=None,
):
    tscv = TimeSeriesSplit(n_splits=int((len(X) - lookback - test_size) / test_size))
    metric_scores = []
    prediction_probabilities = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = (
            y.iloc[train_index],
            y.iloc[test_index],
        )

        purged_start = max(0, len(y_train) - (len(y_train) - purged_window_size))
        embargo_start = max(0, len(y_train) - embargo_period)

        if scaler:
            # Get the columns that start with xs_ and scale them
            X_train_to_scale = X_train.filter(regex="xs_")
            X_test_to_scale = X_test.filter(regex="xs_")

            scaler_fit = scaler.fit(X_train_to_scale.iloc[purged_start:embargo_start])
            X_train_scaled = scaler_fit.transform(X_train_to_scale.iloc[purged_start:embargo_start])
            X_test_scaled = scaler_fit.transform(X_test_to_scale.iloc[purged_start:embargo_start])

            X_train_categorical = X_train.filter(regex="x_")
            X_test_categorical = X_test.filter(regex="x_")

            X_train_scaled = np.concatenate((X_train_scaled, X_train_categorical.iloc[purged_start:embargo_start]), axis=1)
            X_test_scaled = np.concatenate((X_test_scaled, X_test_categorical.iloc[purged_start:embargo_start]), axis=1)

        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        if model_name != "neural_network":
            model.fit(
                X_train_scaled[purged_start:embargo_start],
                y_train[purged_start:embargo_start].values.ravel(),
            )
        else:
            model.fit(
                X_train_scaled[purged_start:embargo_start],
                y_train[purged_start:embargo_start].values.ravel(),
                epochs=150,
                batch_size=10,
                verbose=0,
            )
        predictions = model.predict(X_test_scaled)
        prediction_proba = model.predict_proba(X_test_scaled)

        if prediction_proba.shape[1] == 1:
            # Model is returning only one prediction
            prediction_proba = prediction_proba[:, 0]
        else:
            # Model is returning two predictions, take the second one
            prediction_proba = prediction_proba[:, 1]

        metric_scores.append(scorer(y_test, predictions))
        prediction_proba_series = pd.Series(prediction_proba, index=y_test.index)
        prediction_probabilities.append(prediction_proba_series)

    return metric_scores, prediction_probabilities