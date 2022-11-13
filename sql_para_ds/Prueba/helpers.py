#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import pickle, datetime, os
from pathlib import Path
import os.path

def report_performance(model, model_name, target_name, X_train, X_test, y_train, y_test, pickle_it=True, force_retrain=False):
    """Given a sklearn model class, a partitioned database in train and test,
    train the model, print a classification_report and pickle the trained model.

    :model: a sklearn model class
    :X_train: Feat training matrix
    :X_test: Feat testing matrix
    :y_train: Objective vector training
    :y_test: Objective vector testing
    :pickle_it: If true, store model with an specific tag.
    :returns: TODO

    """
    print(model_name)
    time_stamp = datetime.datetime.now().strftime('%d%m-%H')
    if not os.path.isfile(f"./models/{y_train.name}__{model_name}__{time_stamp}.pkl") or force_retrain:
        tmp_model_train = model.fit(X_train, y_train)
        if pickle_it is True:
            Path("./models").mkdir(parents=True, exist_ok=True)
            pickle.dump(
                {'model': tmp_model_train, 'target': target_name, 'model_name': model_name},
                open(f"./models/{y_train.name}__{model_name}__{time_stamp}.pkl", 'wb')
            )
        print(classification_report(y_test, tmp_model_train.predict(X_test)))
    else:
        print(f'El modelo no se entrenó porque ya existía el archivo {f"./models/{y_train.name}__{model_name}__{time_stamp}.pkl"}')

def create_crosstab(pickled_model, X_test, y_test, variables):
    """Returns a pd.DataFrame with k-variable defined crosstab and its prediction on hold out test

    :pickled_model: TODO
    :X_test: TODO
    :y_test: TODO
    :variables: TODO
    :returns: TODO

    """
    tmp_training = X_test.copy()
    unpickle_obj = pickle.load(open(pickled_model, 'rb'))
    tmp_training[f"{y_test.name}_yhat"] = unpickle_obj['model'].predict(X_test)

    if isinstance(variables, list) is True:
        tmp_query = tmp_training.groupby(variables)[f"{y_test.name}_yhat"].mean()
    else:
        raise TypeError('Variables argument must be a list object')

    del tmp_training
    return tmp_query, unpickle_obj['target'], unpickle_obj['model_name']

def pipeline_maker(**kwargs):
    return Pipeline(steps=[(key,value) for key,value in kwargs.items()])

