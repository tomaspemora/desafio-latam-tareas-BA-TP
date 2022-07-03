import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

def puntaje_z(x):
    return (x - x.mean()) / x.std()


def description(df, lista_columnas_descripcion):
    # Para seleccionar variables numéricas, usamos .select_dtypes(np.number)
    print("_________________________________________________")
    print("_________________________________________________")
    print('Se recomienda ver output completo.')
    print("_________________________________________________")
    print("_________________________________________________")
    print(f'Medias para variables numéricas')
    print(df.select_dtypes(np.number).mean())
    print("_________________________________________________")
    # Para seleccionar variables categóricas, usamos .select_dtypes(object)
    print(f'Frecuencias para variables categóricas')
    print("*************************")
    df.select_dtypes(object).apply(lambda x: print(
        x.value_counts(), '\n*************************'))
    print("_________________________________________________")
    print(f"Descripción variables {','.join(lista_columnas_descripcion)}")
    print(df[lista_columnas_descripcion].describe())

def null_analyzer(dataframe, var, print_list=False):
    null_values = dataframe[var].isnull().sum()
    porcentaje_nulos = round(100 * null_values / len(dataframe),2)
    print(f"La columna {var} tiene {null_values} datos nulos de un total de {len(dataframe)}, correspondientes al {porcentaje_nulos}%")
    if print_list:
        print(f"Los datos nulos en la columna {var} son los siguientes")
        nulos = dataframe[var].isnull()
        display(dataframe[nulos])
    return null_values, porcentaje_nulos

def graficar_histograma(sample_df, full_df, var, sample_mean, true_mean):
    plt.figure(1,figsize=(10,5))
    plt.title(f"Histograma de {var}")
    plt.xlabel(f"{var}")
    plt.ylabel("Frecuencia")
    plt.hist(sample_df[var], bins=20, alpha=0.5, label="Muestra")
    plt.hist(full_df[var], bins=20, alpha=0.5, label="Todo el conjunto")
    if sample_mean:
        plt.axvline(sample_df[var].mean(), color='r', linestyle='dashed',
                    linewidth=2, label=f"Media muestral: {round(sample_df[var].mean(),2)}")
    if true_mean:
        plt.axvline(full_df[var].mean(), color='b', linestyle='dashed', linewidth=2, label=f"Media real: {round(full_df[var].mean(),2)}")
    plt.legend()
    # plt.show()


def graficar_dotplot(dataframe, plot_var, plot_by, statistic='mean', global_stat=False):
    plt.figure(2,figsize=(10, 5))
    plt.title(f"Dotplot de {plot_var} v/s {plot_by}")
    plt.xlabel(f"{plot_by}")
    plt.ylabel(f"{plot_var}")
    if statistic == 'mean':
        y_data = dataframe.groupby(plot_by)[plot_var].mean()
    elif statistic == 'median':
        y_data = dataframe.groupby(plot_by)[plot_var].median()
    else:
        raise ValueError("statistic must be 'mean' or 'median'")
    x_data = dataframe[plot_by].unique()
    if global_stat:
        if statistic == 'mean':
            plt.axhline(dataframe[plot_var].mean(), color='r', linestyle='dashed',
                        linewidth=2, label=f"Media: {round(dataframe[plot_var].mean(),2)}")
        elif statistic == 'median':
            plt.axhline(dataframe[plot_var].median(), color='r', linestyle='dashed',
                        linewidth=2, label=f"Mediana: {round(dataframe[plot_var].median(),2)}")
        else:
            raise ValueError("statistic must be 'mean' or 'median'")
        plt.legend()
    plt.plot(x_data, y_data, 'o')
    plt.xticks(rotation=45)
    # plt.show()

if __name__ == '__main__':

    df = pd.read_csv('subsample_tpm_demo.csv')
    df_sample = df.sample(frac = 0.5)
    print("********************************************************")
    print('Uso de description')
    print(f'Ej: description(df, ["gle_cgdpc", "undp_hdi", "imf_pop"])')
    description(df, ["gle_cgdpc", "undp_hdi", "imf_pop"])
    print("********************************************************")
    print('Uso de null_analyzer')
    print(f'Ej: null_analyzer(df, "undp_hdi", print_list=True)')
    null_analyzer(df, "undp_hdi", print_list=True)
    print("********************************************************")
    print('Uso de graficar_histograma')
    print(f'Ej: graficar_histograma(df_sample, df, "undp_hdi", True, True)')
    graficar_histograma(df_sample, df, "undp_hdi", True, True)
    print("********************************************************")
    print('Uso de graficar_dotplot')
    print(f'Ej: graficar_dotplot(df_sample, "undp_hdi", "ht_region", statistic="mean", global_stat=False)')
    graficar_dotplot(df_sample, "undp_hdi", "ht_region", statistic='mean', global_stat=False)
    print("********************************************************")
    print('Uso de puntaje_z')
    print(f'Ej: puntaje_z(df_sample["undp_hdi"])')
    Z = puntaje_z(df_sample["undp_hdi"])
    print(Z)
    print("********************************************************")
    plt.show()