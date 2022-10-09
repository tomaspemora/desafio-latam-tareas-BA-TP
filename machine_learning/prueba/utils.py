import numpy as np
from IPython.display import display
import pandas as pd
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer import calculate_bartlett_sphericity
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from IPython.display import HTML
import pickle
from sklearn.metrics import classification_report
import inspect, re
import nltk
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob import TextBlob
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from datetime import datetime
from feature_engine.encoding import OrdinalEncoder


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

def report_regression_metrics(model, X_test, y_test, metrics):
    y_pred = model.predict(X_test)
    metrics_results = {}
    for metric_name,metric_function in metrics.items():
        metrics_results[metric_name] = metric_function(y_test,y_pred).round(3)
    return metrics_results

def reporte_modelos(models_dict):
    # models_dict
    models = list(models_dict.keys())
    metrics = models_dict[models[0]].keys()
    table_str = '<table><tr><th>Models</th><th>' + '</th><th>'.join(metrics) + '</th></tr>'
    for model in models:
        table_str += '<tr>'
        table_str += f"<td>{model}</td>"
        for metric in metrics:
            table_str += f"<td>{models_dict[model][metric]:.4f}</td>"
        table_str += '</tr>'
    display(HTML(table_str))

def save_bytes_variable(variable_dict, nombre_archivo):
    file = open(nombre_archivo, 'wb')
    pickle.dump(variable_dict, file)
    file.close()

def load_bytes_variable(nombre_archivo):
    with open(nombre_archivo, 'rb') as f:
        return pickle.load(f)

def cat_num_rate_analysis(df):
    cat_num_rate = df.apply(lambda col: (len(col.unique())/len(col), len(col.unique()), len(col),col.dtype ,  col.unique()))
    cmr = pd.DataFrame(cat_num_rate.T)
    cmr.columns=["num_to_cat_rate", "len of unique", "len of data", "col type", "unique of col"]
    max_rows = pd.get_option('display.max_rows')
    max_width = pd.get_option('display.max_colwidth')
    pd.set_option('display.max_colwidth', 150)
    pd.set_option('display.max_rows', None)
    display(cmr.sort_values(by="num_to_cat_rate",ascending=False))
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_colwidth', max_width)
    return cmr

def train_function(pipe, X_train, X_test, y_train, y_test):
    pipe.fit(X_train, y_train)
    y_pred_train = pipe.predict(X_train)
    y_pred = pipe.predict(X_test)
    print('train')
    print(classification_report(y_train, y_pred_train, digits=4))
    print('test')
    print(classification_report(y_test, y_pred, digits=4))
    return pipe

def test_function(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    print('test')
    print(classification_report(y_test, y_pred, digits=4))
    return pipe

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

### Especificas para prueba Machine Learning twitter.
def columns_reorder(X, new_columns_ordered):
    if sorted(X.columns.to_list()) == sorted(new_columns_ordered):
        return X[new_columns_ordered]
    return X

def multi_class_remapping(X,group_classes = {}, var_name='sentiment', neutral_class='neutral', random_state=42):
    list_sentiments = list(set(group_classes.values()))
    list_sentiments.remove(neutral_class)
    random.seed(random_state)
    X[f'{var_name}_remapped'] = X[var_name].map(group_classes).apply(lambda s: 
    random.choice(list_sentiments) if s == neutral_class else s)
    return X

def remove_arrobas(X, var_name='content'):
    X[f'{var_name}_remarroba']= X[var_name].apply(
        lambda s: re.sub(r'(\@[a-zA-Z0-9\-\_]*)', '', s))
    return X

def match_regex_exp(string='',exp=''):
    string_found = re.findall(exp, string+" ")
    if len(string_found)==0:
        return ""
    if len(string_found)>1:
        return '_____________'.join(string_found)
    return string_found[0]

def remove_links(X, var_name='content'):
    exps = [
            r'(https?\s*?\:\s*?.*?(?=\s|\,|\"|\)|\]))',
            r'(www\.(?!\.|\s|\,).*?(?=\s|\,|\"|\)|\]))',
            r'[^\s|\,]*[^\.]\.(?:com|org|uk)'
            ]

    NX = X.copy()
    XRL = NX[var_name]
    XAL = pd.Series([], dtype='O')
    count = 0
    for exp in exps:
        XL = XRL.apply(lambda s: match_regex_exp(s, exp))
        XL.index = XRL.index.astype(str) + f'_{count}'
        XL = XL.replace("", float("NaN")).dropna()
        XAL = pd.concat([XAL, XL], axis=0)
        XRL = XRL.apply(lambda s: re.sub(exp, '', s+" "))
        count+=1
        
    X[f'{var_name}_remlinks'] = XRL
    XAL.to_csv('http_links_removed.csv')
    XRL.to_csv('clean_twits.csv')
    return X

def remove_chars(X, var_name='content',char_list=[('','')] ):
    NX = X.copy()
    XR = NX[var_name]
    for char in char_list:
        XR = XR.apply(lambda s: re.sub(char[0],char[1],s))
    X[f'{var_name}_remchars'] = XR.str.lower()

    return X

def target_encoding(X,mapping,column_to_encode='sentiment_remapped'):
    NX = X.copy()
    XR = NX[column_to_encode]
    for cl,nc in mapping:
        XR = XR.mask(XR == cl,nc)
    X[column_to_encode] = XR.astype('int32')
    return X


class RemoveStopWords(BaseEstimator,TransformerMixin):
    def __init__(self, text_columns=[], bool_trans=True):
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('words',download_dir='.')

        self.dictionary = stopwords.words('english')
        self.text_columns = text_columns
        self.bool_trans = bool_trans

    def fit(self, X, Y):
        return self

    def create_clean_column(self,twitt):
        return " ".join([word for word in word_tokenize(twitt) if word not in self.dictionary]).lower()


    def transform(self, X, Y=None):
        if self.bool_trans:
            NX = X.copy()
            try:
                for col in self.text_columns:
                    NX[f"{col}_sw"] = NX[col].apply(self.create_clean_column)
            except Exception as err:
                print('RemoveStopWords.transform(): {}'.format(err))
            return NX
        return X

    def fit_transform(self, X, Y=None):
        self.fit(X, Y)
        return self.transform(X, Y)

class LemmantizerTransformer(BaseEstimator,TransformerMixin):
    def __init__(self, text_columns=[], stemmer='ps'):
        try:
            nltk.data.find('corpora/wordnet.zip')
            nltk.data.find('corpora/omw-1.4.zip/omw-1.4/')
        except LookupError:
            nltk.download('wordnet')
            nltk.download('omw-1.4')
        self.wnl = nltk.wordnet.WordNetLemmatizer()
        self.sno = nltk.stem.SnowballStemmer('english')
        self.ps = nltk.stem.PorterStemmer()
        self.text_columns = text_columns
        self.stemmer = stemmer

    def fit(self, X, Y):
        return self

    def stemSentence(self, sentence, stemmer):
        token_words=word_tokenize(sentence)
        token_words
        stem_sentence=[]
        for word in token_words:
            stem_sentence.append(stemmer(word))
            stem_sentence.append(" ")
        return "".join(stem_sentence)

    def create_lemma_column(self,twitt, method):
        if method == 'ps':
            return self.stemSentence(twitt, self.ps.stem)
        if method == 'sno':
            return self.stemSentence(twitt, self.sno.stem)
        if method == 'lemma':
            return self.stemSentence(twitt, self.wnl.lemmatize)
        return twitt


    def transform(self, X, Y=None):
        NX = X.copy()
        try:
            for col in self.text_columns:
                if 'ps' == self.stemmer:
                    NX[f"{col}_stemmer"] = NX[col].apply(self.create_lemma_column, method='ps')
                elif 'sno' == self.stemmer:
                    NX[f"{col}_stemmer"] = NX[col].apply(self.create_lemma_column, method = 'sno')
                elif 'wnl' == self.stemmer:
                    NX[f"{col}_stemmer"] = NX[col].apply(self.create_lemma_column, method = 'wnl')
        except Exception as err:
            print('LemmantizerTransformer.transform(): {}'.format(err))
        return NX

    def fit_transform(self, X, Y=None):
        self.fit(X, Y)
        return self.transform(X, Y)

class FeatureExtractionTwitts(BaseEstimator,TransformerMixin):
    def __init__(self, text_column="content_min", features_to_extract = []):
        
        self.text_column = text_column
        self.features_to_extract = features_to_extract
        
    def fit(self, X, Y):
        return self

    def regex_count(self,twitt,patt=r'(\#[a-zA-Z0-9\-\_]*)',threshold=3):
        pattern = re.compile(patt, re.IGNORECASE)
        res = re.findall(pattern, twitt)
        return min(len(res),threshold)
        
    def is_reply(self,twitt):
        if self.regex_count(twitt,patt=r'(^\@[a-zA-Z0-9\-\_]*)')>0:
            return 1
        return 0
                
    def is_rt(self,twitt):
        if self.regex_count(twitt,patt=r'(^RT*)')>0:
            return 1
        return 0
    
    def getSubjectivity(self, text):
        return TextBlob(text).sentiment.subjectivity

    def getPolarity(self, text):
        return TextBlob(text).sentiment.polarity

    def transform(self, X, Y=None):
        NX = X.copy()
        try:
            if "arrobas_count" in self.features_to_extract:
                NX[f"var_arrobas_count"] = NX[self.text_column].apply(self.regex_count, patt= r'(\@[a-zA-Z0-9\-\_]*)', threshold=3)
            if "hashtag_count" in self.features_to_extract:
                NX[f"var_hashtag_count"] = NX[self.text_column].apply(self.regex_count, patt= r'(\#[a-zA-Z0-9\-\_]*)', threshold=1)
            if "is_reply" in self.features_to_extract:
                NX[f"var_is_reply"] = NX[self.text_column].apply(self.is_reply)
            if "is_rt" in self.features_to_extract:
                NX[f"var_is_rt"] = NX[self.text_column].apply(self.is_rt)
            if "subjectivity" in self.features_to_extract:
                NX[f"var_subjectivity"] = NX[self.text_column].apply(self.getSubjectivity) 
            if "polarity" in self.features_to_extract:
                NX[f"var_polarity"] = NX[self.text_column].apply(self.getPolarity)
            if "twitt_length" in self.features_to_extract:
                NX[f"var_twit_length"] = NX[self.text_column].apply(lambda x: len(x))
        except Exception as err:
            print('FeatureExtractionTwitts.transform(): {}'.format(err))
        return NX

    def fit_transform(self, X, Y=None):
        self.fit(X, Y)
        return self.transform(X, Y)

class Vectorizer(BaseEstimator,TransformerMixin):
    def __init__(self, vect_type='count', text_column='content', min_df=.1, max_df=.8,ngram_range=(1,1)):
        self.vect_type = vect_type
        self.text_column = text_column
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range

    def fit(self, X, Y):
        NX = X.copy()
        Txt_sel = NX[self.text_column]
        if self.vect_type == 'count':
            self.cvec = CountVectorizer(min_df=self.min_df, max_df=self.max_df, ngram_range = self.ngram_range)
        elif self.vect_type == 'tfid':
            self.cvec = TfidfVectorizer(min_df = self.min_df, max_df = self.max_df, ngram_range = self.ngram_range)
        else:
            raise Exception('Solo se acepta "count" y "tfid" para min_df')
        self.cvec.fit(Txt_sel)
        self.tokens = ['var_token_' + sub for sub in list(self.cvec.get_feature_names_out())]
        return self

    def transform(self, X, Y=None):
        NX = X.copy()
        Txt_sel = NX[self.text_column]
        features = self.cvec.transform(Txt_sel)
        count_vect_df = pd.DataFrame(features.todense(), columns=self.tokens, index=Txt_sel.index)
        NX = pd.concat([NX, count_vect_df], axis=1)
        return NX

    def fit_transform(self, X, Y=None):
        self.fit(X, Y)
        return self.transform(X, Y)

class ColumnSelectedTransformer():
    def __init__(self, vars_prefix='var_'):
        self.vars_prefix = vars_prefix

    def fit(self, X, y=None):
        return self 

    def transform(self,X,y=None):
        filter_col = [col for col in X.columns if col.startswith(self.vars_prefix)]
        return X[filter_col]

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

class KerasCustomClassifier(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def make_model(self,n_features, hidden_1=32, drop=0.2, optimizer="Adam"):
        model = Sequential(name = 'titanic')
        model.add(Dense(name = 'hidden_1', units = hidden_1, activation = 'relu' ,input_shape=(n_features, )))
        model.add(Dropout(name ='drop', rate = drop))
        model.add(Dense(name = 'output', units = 1, activation = 'sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer = optimizer, metrics = 'accuracy')
        return model

    def predict(self, X):
        return self.base_estimator.predict(X)

    def classify(self, inputs):
        return self.base_estimator.classify(inputs)

    def fit(self, X, y):
        n_features = X.shape[1]
        self.base_estimator = KerasClassifier(self.make_model, n_features=n_features)
        self.base_estimator.fit(X,y)

## Aqui parte para crimenes
class CreateSuitableDataframeTransformer(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.preserve_vars = []
        self.suitable_categorical_attributes = []

    def infer_datatype(self,df, datatype, drop_none=True):
        """ A partir de un dataset y un tipo de datos entregado, devuelve los nombres de las columnas
            del dataset que tienen el correspondiente tipo de dato.
            
            Argumentos:
            - df: Dataframe de pandas.
            - datatype: String con el tipo de dato que se desea consultar a las columnas del dataframe.
            - drop_none: Filtra las columnas cuyo tipo de dato no esté especificado. default = True.
        """
        tmp_list = [i if df[i].dtype == datatype else None for i in df.columns]
        if drop_none is True:
            tmp_list = list(filter(lambda x: x != None, tmp_list))

        return tmp_list


    def return_time_string(self,var, date_format='%m%d%Y'):
        return var.apply(lambda x: datetime.strptime(str(x), date_format))


    def count_freq(self,df, selected_columns):
        """ Cuenta la cantidad de valores únicos y la frecuencia de dichos valores en las columnas
            entregadas por `selected_columns`.
            
            Argumentos:
                - df: dataframe que contiene las columnas en cuestión.
                - selected_columns: Columnas del dataframe de las que se quiere saber la frecuencia de valores.
        """
        return {i: df[i].unique().shape[0] for i in selected_columns}
    
    def fit(self, X, Y):
        NX = X.copy()
        #Limpieza de variables de ubicación "xcoord" e "ycoord"
        NX = NX.loc[NX["xcoord"] != " ", :]

        NX.loc[:, "xcoord"] = NX["xcoord"].apply(lambda x: np.nan if x==' ' else float(x))
        NX.loc[:, "ycoord"] = NX["ycoord"].apply(lambda x: np.nan if x==' ' else float(x))
        self.ciudad_moda = NX['city'].mode()
        NX.loc[:, "city"] = NX["city"].apply(lambda x: self.ciudad_moda if x == ' ' else x).apply(lambda x: 'STATEN ISLAND' if x == 'STATEN IS' else x)

        # NX.loc[:, "post"] = NX["post"].apply(lambda x: np.nan if x==' ' else int(x))
        ### Obtener columnas por tipo de dato
        object_data_type = self.infer_datatype(NX, 'object')
        integer_data_type = self.infer_datatype(NX, 'int')
        float_data_type = self.infer_datatype(NX, 'float')
        
        # Quiero recuperar la lista de valores numericos tambien
        suitable_numerical_attributes = list(integer_data_type) + list(float_data_type)
        # print(suitable_numerical_attributes)
        
        ### Contar la cantidad de clases en el caso de las var. categóricas y frecuencia de valores para las numéricas
        object_unique_vals = self.count_freq(NX, object_data_type)
        int_unique_vals = self.count_freq(NX, integer_data_type)
        float_unique_vals = self.count_freq(NX, float_data_type)
        
        ### Selección de atributos categoricos que cumplen con características deseadas
        suitable_categorical_attributes = dict(filter(lambda x: x[1] < 100 and x[1] >= 2, object_unique_vals.items()))
        suitable_categorical_attributes = list(suitable_categorical_attributes.keys())
        
        preserve_vars = suitable_categorical_attributes + ['month', 'meters', "xcoord", "ycoord"]
        self.preserve_vars = preserve_vars
        self.suitable_categorical_attributes = suitable_categorical_attributes
        self.suitable_numerical_attributes = suitable_numerical_attributes

        return self
        
    def transform(self, X, Y=None):
        NX = X.copy()
        #Limpieza de variables de ubicación "xcoord" e "ycoord"
        
        NX.loc[:,"xcoord"] = NX["xcoord"].apply(lambda x: np.nan if x==' ' else float(x))
        NX.loc[:,"ycoord"] = NX["ycoord"].apply(lambda x: np.nan if x==' ' else float(x))
        NX.loc[:, "city"] = NX["city"].apply(lambda x: self.ciudad_moda if x == ' ' else x).apply(lambda x: 'STATEN ISLAND' if x == 'STATEN IS' else x)
        # NX.loc[:, "post"] = NX["post"].apply(lambda x: np.nan if x==' ' else int(x))
        
        ### Reemplazo de clases faltantes
        ### {N: No, Y: Yes, U: Unknown}

        # NX['officrid'] = np.where(NX['officrid'] == ' ', 'N', 'Y')
        NX['offshld'] = np.where(NX['offshld'] == ' ', 'N', 'Y')
        # NX['sector'] = np.where(NX['sector'] == ' ', 'U', NX['sector'])
        # NX['trhsloc'] = np.where(NX['trhsloc'] == ' ', 'U', NX['trhsloc'])
        # NX['beat'] = np.where(NX['beat'] == ' ', 'U', NX['beat'])
        # NX['offverb'] = np.where(NX['offverb'] == ' ', 'N', 'Y')

        meters = NX['ht_feet'].astype(str) + '.' + NX['ht_inch'].astype(str)
        # Conversión de distanca a sistema metrico (non retarded)
        NX['meters'] = meters.apply(lambda x: float(x) * 0.3048)
        NX['month'] = self.return_time_string(NX['datestop']).apply(
            lambda x: x.month)  # Agregación a solo meses
        NX = NX.loc[:, self.preserve_vars] # Agregar los atributos sintéticos al df
        return NX

    def fit_transform(self, X, Y=None):
        self.fit(X, Y)
        return self.transform(X, Y)


class OrdinalEncoderFixedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,encoding_method, csd):
        self.csd = csd
        self.encoding_method = encoding_method

    def fit(self, X, Y):
        self.encoded_columns = self.csd.suitable_categorical_attributes
        self.encoder = OrdinalEncoder(encoding_method=self.encoding_method, variables=self.csd.suitable_categorical_attributes)
        self.encoder.fit(X,Y)
        return self
        
    def transform(self, X, Y=None):
        return self.encoder.transform(X)

    def fit_transform(self, X, Y=None):
        self.fit(X, Y)
        return self.transform(X, Y)

class DropRowsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, Y):
        return self
        
    def transform(self, X, Y=None):
        X = X.loc[X["xcoord"] != " ", :]
        X = X.loc[X["ycoord"] != " ", :]
        X['age_individual'] = np.where(np.logical_and(X['age'] > 18, X['age'] < 100), X['age'], np.nan) # Evaluar si mantener esta condición, no la entiendo bien
        X = X.dropna()
        return X

    def fit_transform(self, X, Y=None):
        self.fit(X, Y)
        return self.transform(X, Y)

def split_features_target(df):
    ## Definición target y_1
    y_1 = (df.arstmade == 'Y').astype(int)
    
    ## Transformación target y_2
    var_pf = df.columns[np.where([i[0:2]=='pf' for i in df.columns.tolist()])]. tolist()
    u = df[var_pf]
    y_2 = pd.Series([int(np.isin(["Y"], u.iloc[i].values.tolist())[0]) for i in range(0,len(u))], name='violence')

    ## Predictores para y_1 e y_2: hay un subconjunto de potenciales predictores para y_1 y otro para y_2
    x_1 = df.drop(columns=['arstmade'])
    var_eliminar_pf = ["pf_baton", "pf_hcuff", "pf_pepsp", "pf_other", "pf_ptwep", "pf_drwep", "pf_wall", "pf_hands", "pf_grnd"] #9 variables eliminadas
    x_2 = df.drop(columns = var_eliminar_pf)
    
    return x_1, y_1, x_2, y_2

class CriterioExperto(BaseEstimator, TransformerMixin):
    def __init__(self,columns):
        self.columns = columns

    def fit(self, X, Y):
        return self
        
    def transform(self, X, Y=None):
        
        return X.drop(columns = self.columns)

    def fit_transform(self, X, Y=None):
        self.fit(X, Y)
        return self.transform(X, Y)