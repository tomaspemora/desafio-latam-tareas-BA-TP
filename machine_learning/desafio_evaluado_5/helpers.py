#!/usr/bin/env python
# -*- coding: utf-8 -*-

from decimal import Underflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve, auc

def plot_classification_report(y_true, y_hat, method):
    """
    plot_classification_report: Genera una visualización de los puntajes reportados con la función `sklearn.metrics.classification_report`.

    Parámetros de ingreso:
        - y_true: Un vector objetivo de validación.
        - y_hat: Un vector objetivo estimado en función a la matriz de atributos de validación y un modelo entrenado.

    Retorno:
        - Un gráfico generado con matplotlib.pyplot

    """
    # process string and store in a list
    report = classification_report(y_true, y_hat).split()
    # keep values
    report = [i for i in report if i not in ['precision', 'recall', 'f1-score', 'support', 'weighted avg']]
    # transfer to a DataFrame
    report = pd.DataFrame(np.array(report).reshape(len(report) // 5, 5))
    # asign columns labels
    report.columns = ['idx', 'prec', 'rec', 'f1', 'n']
    # preserve class labels
    class_labels = report.iloc[:np.unique(y_true).shape[0]].pop('idx').apply(int)
    # separate values
    class_report = report.iloc[:np.unique(y_true).shape[0], 1:4]
    # convert from str to float
    class_report = class_report.applymap(float)
    # convert to float average report
    average_report = report.iloc[-1, 1: 4].apply(float)
    if method == 'soft' or method == '':
        colors = ['dodgerblue', 'tomato', 'purple', 'orange']
        markers = 'x'
    else:
        colors = ['green', 'royalblue','khaki','darkviolet']
        markers = '+'

    for i in class_labels:
        plt.plot(class_report['prec'][i], [1], marker=markers, color=colors[i])
        plt.plot(class_report['rec'][i], [2], marker=markers, color=colors[i])
        plt.plot(class_report['f1'][i], [3], marker=markers, color=colors[i], label=f'Class: {i} - {method}')

    plt.scatter(average_report, [1, 2, 3], marker='o', color=colors[3], label=f'Avg - {method}')
    plt.yticks([1.0, 2.0, 3.0], ['Precision', 'Recall', 'f1-Score'])
    plt.legend(bbox_to_anchor=(1.1, 1.05))

def grid_plot_batch(df, cols, plot_type):

    """
    grid_plot_batch: Genera una grilla matplotlib para cada conjunto de variables.

    Parámetros de ingreso:
        - df: un objeto pd.DataFrame
        - cols: cantidad de columnas en la grilla.
        - plot_type: tipo de gráfico a generar. Puede ser una instrucción genérica de matplotlib o seaborn.

    Retorno:
        - Una grilla generada con plt.subplots y las instrucciones dentro de cada celda.

    """
    # calcular un aproximado a la cantidad de filas
    rows = int(np.ceil(df.shape[1] / cols))

    # para cada columna
    for index, (colname, serie) in enumerate(df.iteritems()):
        plt.subplot(rows, cols, index + 1)
        plot_type(serie)
        plt.title(colname)
        plt.tight_layout()

def identify_high_correlations(df, threshold=.7):
    """
    identify_high_correlations: Genera un reporte sobre las correlaciones existentes entre variables, condicional a un nivel arbitrario.

    Parámetros de ingreso:
        - df: un objeto pd.DataFrame, por lo general es la base de datos a trabajar.
        - threshold: Nivel de correlaciones a considerar como altas. Por defecto es .7.

    Retorno:
        - Un pd.DataFrame con los nombres de las variables y sus correlaciones
    """

    # extraemos la matriz de correlación con una máscara booleana
    tmp = df.corr().mask(abs(df.corr()) < threshold, df)
    # convertimos a long format
    tmp = pd.melt(tmp)
    # agregamos una columna extra que nos facilitará los cruces entre variables
    tmp['var2'] = list(df.columns) * len(df.columns)
    # reordenamos
    tmp = tmp[['variable', 'var2', 'value']].dropna()
    # eliminamos valores duplicados
    tmp = tmp[tmp['value'].duplicated()]
    # eliminamos variables con valores de 1 
    return tmp[tmp['value'] < 1.00]

def plot_roc(model, y_true, X_test, model_label=None):
    """TODO: Docstring for plot_roc.

    :model: TODO
    :y_true: TODO
    :X_test: TODO
    :model_label: TODO
    :returns: TODO

    """
    class_pred = model.predict_proba(X_test)[:1]
    false_positive_rates, true_positive_rates, _ = roc_curve(y_true, class_pred)
    store_auc = auc(false_positive_rates, true_positive_rates)

    if model_label is not None:
        tmp_label = f'{model_label}: {round(store_auc, 3)}'
    else:
        tmp_label = None
    plt.plot(false_positive_rates, true_positive_rates, label=tmp_label)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


class PreprocBCT():
    def __init__(self, under=None, upper=None, threshold=.8):
        self.selector = None
        self.under = under
        self.upper = upper
        self.threshold = threshold
        self.variables_eliminidas = []

    def fit(self, X, Y):
        # Punto 2
        NX = X.copy()
        corrs = identify_high_correlations(NX, self.threshold)
        if(len(corrs) > 0):
            while(abs(corrs.value.iloc[0]) > self.threshold):
                NX = NX.drop(columns = corrs.var2.iloc[0])
                self.variables_eliminidas.append(corrs.var2.iloc[0])
                corrs = identify_high_correlations(NX, self.threshold)
                if (len(corrs) == 0):
                    break
        return self

    def transform(self, X, Y=None):
        NX = X.copy()
        try:
            # Punto 2
            NX = NX.drop(columns = self.variables_eliminidas)

        except Exception as err:
            print('MyFeatureSelector.transform(): {}'.format(err))
        return NX

def fit_transform(self, X, Y=None):
    self.fit(X, Y)
    return self.transform(X, Y)