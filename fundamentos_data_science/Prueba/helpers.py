import numpy as np
from IPython.display import display
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer import calculate_bartlett_sphericity

## AGREGAR DOCSTRING

#ELIMINAR PRETTY
def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            pretty(value, indent+1)
            
        else:
            print(':' * (1) + str(value))

def encoding_feature(x): return dict(zip(range(len(x)),x))

def make_pretty(styler, num_format='{:.2f}'):
    d1 = dict(selector="td",props=[('text-align', 'center')])
    d2 = dict(selector="th",props=[('text-align', 'center')])
    d3 = dict(selector=".index_name",props=[('text-align', 'center')])
    d4 = dict(selector="th.col_heading",props=[('text-align', 'center')])
    styler.format(num_format)
    #styler.format_index(lambda v: v.upper())
    styler.background_gradient(axis=None, cmap="YlGnBu")
    styler.set_table_styles([d1,d2,d3,d4])
    styler.set_properties(**{'border': '1px black solid !important', 'text-align': 'center'})
    styler.set_table_styles([{'selector': 'th','props': [('border', '2px black solid !important'), ('min-width','90px'), ('max-width','90px'), ('width','90px'), ('text-align','center')]}])
    styler.set_table_styles([{'selector': 'td','props': [('border', '2px black solid !important'), ('min-width','90px'), ('max-width','90px'), ('width','90px'), ('text-align','center')]}])
    styler.set_table_styles([{'selector': '.index_name','props': [('border', '2px black solid !important'), ('min-width','90px'), ('max-width','90px'), ('width','90px'), ('text-align','center')]}])
    styler.set_table_styles([{'selector': 'th.col_heading','props': [('border', '2px black solid !important'), ('min-width','90px'), ('max-width','90px'), ('width','90px'), ('text-align','center')]}])
    #styler.applymap_index(lambda v: "min-width:90px;max-width:90px;width:90px", axis=0)
    return styler

def get_type_vars(df):
    cat_variables = df.select_dtypes(include=['object','string']).columns # Variables categóricas
    num_variables = df.select_dtypes(include=['number']).columns # Variables numéricas
    return (cat_variables, num_variables)

def describe_variables(df):
    '''describe_variables(df)
    Realiza una iteración de las columnas de un dataframe (df), permitiendo visualizar 
    de manera individual cada una de éstas junto al número de ocurrencias por cada variable
    ordenadas de manera descendente.   
    
    Parametros:
        df: DataFrame
            Ingresar el dataframe del cual se quiere visualizar la información contenida
            en sus columnas
    Retorno:
        NoneType
        Devuelve una tabla por cada columna iterada mostrando el número de ocurrencias de las 
        variables contenidas en ella, formateadas de manera descendente, destacando las o las 
        de mayor relevancia. Agrupadas por variables categóricas y numéricas.
    '''
    cat_variables,_ = get_type_vars(df)    
    print("------------------------------------------------------------")
    print("-------------------Variables Categóricas--------------------")
    print("------------------------------------------------------------")
    for col in cat_variables:
        display(pd.DataFrame(df[col].value_counts()).T.style.pipe(make_pretty, num_format='{:d}'))
    print("------------------------------------------------------------")
    print("-------------------Variables Numéricas----------------------")
    print("------------------------------------------------------------")
    display(df.select_dtypes(include=['number']).describe().T.style.pipe(make_pretty, num_format='{:.1f}'))

def evaluation(model, real, preds):
    print(f"AIC es : {model.aic}")
    print(f"BIC es : {model.bic}")
    print(f"Condition Number: {model.condition_number}")
    print(f"R2: {r2_score(real, preds)}")
    print(f"RMSE: {mean_squared_error(real, preds, squared=False)} ")
    
def OrdinalEncoderListCategories(df, direction = 'ascending', bin_or_num = 'bin'):
    if direction == 'ascending':
        if bin_or_num == 'bin':
            return [df[col].value_counts(sort=True, ascending = True).index.to_list() for col in df.select_dtypes(np.object_).columns.to_list() if len(df[col].value_counts()) == 2]
        else:
            return [df[col].value_counts(sort=True, ascending = True).index.to_list() for col in df.select_dtypes(np.object_).columns.to_list() if len(df[col].value_counts()) > 2]
    if bin_or_num == 'bin':
        return [df[col].value_counts(sort=True, ascending = False).index.to_list() for col in df.select_dtypes(np.object_).columns.to_list() if len(df[col].value_counts()) == 2]
    return [df[col].value_counts(sort=True, ascending = False).index.to_list() for col in df.select_dtypes(np.object_).columns.to_list() if len(df[col].value_counts()) > 2]



def test_factor_analyzer(dataf):
    data_np = dataf.values
    _, p_value = calculate_bartlett_sphericity(data_np)
    p_value #p_value tiene que ser menor que un nivel de significancia 0.05
    print(f'p_value: {p_value}. Tiene que ser menor que un nivel de significancia 0.05, OK para poder usar factor analyzer')
    kmo_all, kmo_model = calculate_kmo(data_np)
    kmo_model  # si kmo_model es menor a 0.6 el factor analyzer no se puede hacer
    print(f'El valor de kmo es {kmo_model}. Si kmo_model es menor a 0.6 el factor analyzer no se puede hacer... 0.7 dice la lectura ')
    display(pd.DataFrame({"KMO_ALL":kmo_all},index = dataf.columns))
    