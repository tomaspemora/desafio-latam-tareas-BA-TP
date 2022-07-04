import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def null_analyzer(dataframe, var, print_list =False):
    totales = dataframe[var].isnull().sum()
    print(var)
    print(f"La cantidad de registros perdidos es {totales}, {totales/len(dataframe[var])}")
    print(f"*************************\n")


def puntaje_z(dataframe, col):
    media = dataframe[col].mean()
    desv_est = dataframe[col].std()

    ptje_z = (dataframe[col] - media) / desv_est
    return ptje_z




#def puntaje_z(x):
    #return (x - x.mean()) / x.std()


#def puntaje_z_norm(x):
    #return (x - x.mean()) / x.std()