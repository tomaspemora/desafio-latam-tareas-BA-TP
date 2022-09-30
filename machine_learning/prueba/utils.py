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


def train_function(pipe, X_train, X_test, y_train, y_test):
    pipe.fit(X_train, y_train)
    y_pred_train = pipe.predict(X_train)
    y_pred = pipe.predict(X_test)
    print('train')
    print(classification_report(y_train, y_pred_train, digits=4))
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
    X[f'{var_name}_remarroba'] = X[var_name].apply(lambda s: re.sub(r'(\@[a-zA-Z0-9\-\_]*)', '', s))
    return X

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
                NX = NX.drop(columns=corrs.var2.iloc[0])
                self.variables_eliminidas.append(corrs.var2.iloc[0])
                corrs = identify_high_correlations(NX, self.threshold)
                if (len(corrs) == 0):
                    break
        return self

    def transform(self, X, Y=None):
        NX = X.copy()
        try:
            # Punto 2
            NX = NX.drop(columns=self.variables_eliminidas)

        except Exception as err:
            print('MyFeatureSelector.transform(): {}'.format(err))
        return NX


    def fit_transform(self, X, Y=None):
        self.fit(X, Y)
        return self.transform(X, Y)

class RemoveStopWords(BaseEstimator,TransformerMixin):
    def __init__(self, text_columns=[]):
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('words',download_dir='.')

        self.dictionary = stopwords.words('english')
        self.columns = text_columns

    def fit(self, X, Y):
        return self

    def create_clean_column(self,twitt):
        return " ".join([word for word in twitt.split(" ") if word not in self.dictionary]).lower()


    def transform(self, X, Y=None):
        NX = X.copy()
        try:
            for col in self.columns:
                NX[f"{col}_sw"] = NX[col].apply(self.create_clean_column)
        except Exception as err:
            print('RemoveStopWords.transform(): {}'.format(err))
        return NX

    def fit_transform(self, X, Y=None):
        self.fit(X, Y)
        return self.transform(X, Y)


class LemmantizerTransformer(BaseEstimator,TransformerMixin):
    def __init__(self, text_columns=[], stemmers='ps'):
        try:
            nltk.data.find('corpora/wordnet.zip')
            nltk.data.find('corpora/omw-1.4.zip/omw-1.4/')
        except LookupError:
            nltk.download('wordnet')
            nltk.download('omw-1.4')
        self.wnl = nltk.wordnet.WordNetLemmatizer()
        self.sno = nltk.stem.SnowballStemmer('english')
        self.ps = nltk.stem.PorterStemmer()
        self.columns = text_columns
        self.stemmers = stemmers

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
            for col in self.columns:
                if 'ps' in self.stemmers:
                    NX[f"{col}_ps"] = NX[col].apply(self.create_lemma_column, method='ps')
                if 'sno' in self.stemmers:
                    NX[f"{col}_sno"] = NX[col].apply(self.create_lemma_column, method = 'sno')
                if 'wnl' in self.stemmers:
                    NX[f"{col}_wnl"] = NX[col].apply(self.create_lemma_column, method = 'wnl')
        except Exception as err:
            print('LemmantizerTransformer.transform(): {}'.format(err))
        return NX

    def fit_transform(self, X, Y=None):
        self.fit(X, Y)
        return self.transform(X, Y)

class FeatureExtractionTwitts(BaseEstimator,TransformerMixin):
    def __init__(self, text_column="content_min", features_to_extract = []):
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('words',download_dir='.')

        self.dictionary = stopwords.words('english')
        self.twit_text_column = text_column
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
                NX[f"var_arrobas_count"] = NX[self.twit_text_column].apply(self.regex_count, patt= r'(\@[a-zA-Z0-9\-\_]*)', threshold=3)
            if "hashtag_count" in self.features_to_extract:
                NX[f"var_hashtag_count"] = NX[self.twit_text_column].apply(self.regex_count, patt= r'(\#[a-zA-Z0-9\-\_]*)', threshold=1)
            if "is_reply" in self.features_to_extract:
                NX[f"var_is_reply"] = NX[self.twit_text_column].apply(self.is_reply)
            if "is_rt" in self.features_to_extract:
                NX[f"var_is_rt"] = NX[self.twit_text_column].apply(self.is_rt)
            if "subjectivity" in self.features_to_extract:
                NX[f"var_subjectivity"] = NX[self.twit_text_column].apply(self.getSubjectivity) 
            if "polarity" in self.features_to_extract:
                NX[f"var_polarity"] = NX[self.twit_text_column].apply(self.getPolarity)
            if "twitt_length" in self.features_to_extract:
                NX[f"var_twit_length"] = NX[self.twit_text_column].apply(lambda x: len(x))
            

        except Exception as err:
            print('FeatureExtractionTwitts.transform(): {}'.format(err))
        return NX

    def fit_transform(self, X, Y=None):
        self.fit(X, Y)
        return self.transform(X, Y)
