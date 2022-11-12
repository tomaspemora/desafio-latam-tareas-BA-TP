#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle, datetime, os, glob
import pandas as pd
import numpy as np
import psycopg2

def report_performance(model, X_train, X_test, y_train, y_test, pickle_it=True):
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
    tmp_model_train = model.fit(X_train, y_train)

    if pickle_it is True:
        model_name = str(model.__class__).replace("'>", '').split('.')[-1]
        time_stamp = datetime.datetime.now().strftime('%d%m-%H')
        pickle.dump(tmp_model_train,
                    open(f"./{y_train.name}_{model_name}_{time_stamp}.pkl", 'wb')
                    )
    print(classification_report(y_test, tmp_model_train.predict(X_test)))

def create_crosstab(pickled_model, X_test, y_test, variables):
    """Returns a pd.DataFrame with k-variable defined crosstab and its prediction on hold out test

    :pickled_model: TODO
    :X_test: TODO
    :y_test: TODO
    :variables: TODO
    :returns: TODO

    """
    tmp_training = X_test.copy()
    unpickle_model = pickle.load(open(pickled_model, 'rb'))
    tmp_training[f"{y_test.name}_yhat"] = unpickle_model.predict(X_test)

    if isinstance(variables, list) is True:
        tmp_query = tmp_training.groupby(variables)[f"{y_test.name}_yhat"].mean()
    else:
        raise TypeError('Variables argument must be a list object')

    del tmp_training, unpickle_model
    return tmp_query



