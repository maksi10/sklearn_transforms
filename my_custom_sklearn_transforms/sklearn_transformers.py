from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
import numpy as np


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')


class ZeroNan(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        values = {self.columns[0]: 0}
        data = data.fillna(value=values)
        print(50*'-')
        print('Transformed Column ' + self.columns[0] + ' from NaN to 0.')
        print(50*'-')
        return data
    

class InglesDT(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        #Creating DF with populated INGLES
        df_notnull = X[X['INGLES'].notnull()]
        features = [
                    "MATRICULA", 'REPROVACOES_DE', 'REPROVACOES_EM', "REPROVACOES_MF", "REPROVACOES_GO",
                    "NOTA_DE", "NOTA_EM", "NOTA_MF", "NOTA_GO",
                    "H_AULA_PRES", "TAREFAS_ONLINE", "FALTAS", 
                   ]

        # Definição da variável-alvo
        target = ["INGLES"]

        # Preparação dos argumentos para os métodos da biblioteca ``scikit-learn``
        Xt = df_notnull[features]
        yt = df_notnull[target]

        # Criação da árvore de decisão com a biblioteca ``scikit-learn``:
        dtc_model = DecisionTreeClassifier()  # O modelo será criado com os parâmetros padrões da biblioteca

        # Treino do modelo 
        dtc_model.fit(Xt,yt)

        #Reset X to original Dataframe (X)
        Xt = X[features]

        # Loop Dataframe populating INGLES with DT Predicted Value 
        predcount = 0 
        for i, row in X.iterrows():
            if np.isnan(X.at[i,'INGLES']):   #If INGLES is nan
                X.at[i,'INGLES'] = dtc_model.predict(Xt.loc[i].to_numpy().reshape(1, -1))
                predcount =  predcount + 1
        
        print(50*'-')
        print('Populated ' + str(predcount) + ' ENGLISH values on nan rows using a DecisionTree.')
        print(50*'-')
        return X

class Encode_localtrab(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print('Encoding localtrab')
        data = X.copy()
        data.loc[data['Local de trabalho'] == 'Cliente'   , 'Local de trabalho'] = 0
        data.loc[data['Local de trabalho'] == 'Misto'     , 'Local de trabalho'] = 1
        data.loc[data['Local de trabalho'] == 'Escritório', 'Local de trabalho'] = 2
        return data


class Encode_depart(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print('Encoding depart')
        data = X.copy()
        data.loc[data['Departmento'] == 'Engenharia', 'Departmento'] = 0
        data.loc[data['Departmento'] == 'Vendas'    , 'Departmento'] = 1
        data.loc[data['Departmento'] == 'RH'        , 'Departmento'] = 2
        return data

class Encode_educ(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print('Encoding educ')
        data = X.copy()
        data.loc[data['Educacao'] == 'Médio completo'                , 'Educacao'] = 0
        data.loc[data['Educacao'] == 'Superior incompleto - cursando', 'Educacao'] = 1
        data.loc[data['Educacao'] == 'Superior incompleto'           , 'Educacao'] = 2
        data.loc[data['Educacao'] == 'Superior completo'             , 'Educacao'] = 3
        data.loc[data['Educacao'] == 'Pós-graduação'                 , 'Educacao'] = 4
        return data

class Encode_area(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print('Encoding area')
        data = X.copy()
        data.loc[data['Area'] == 'Ciências das natureza'   , 'Area'] = 0
        data.loc[data['Area'] == 'Medicina'                , 'Area'] = 1
        data.loc[data['Area'] == 'Outros'                  , 'Area'] = 2
        data.loc[data['Area'] == 'Marketing'               , 'Area'] = 3
        data.loc[data['Area'] == 'Faculdade Técnica'       , 'Area'] = 4
        data.loc[data['Area'] == 'Ciências humanas'        , 'Area'] = 5
        return data

class Encode_genero(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        self.le = le
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print('Encoding genero')
        data = X.copy()
         data.loc[data['Genero'] == 'M'  , 'Genero'] = 0
         data.loc[data['Genero'] == 'F'  , 'Genero'] = 1
        return data

class Encode_contrat(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print('Encoding contrat')
        data = X.copy()
        data.loc[data['Contratar'] == 'Não'  , 'Contratar'] = 0
        data.loc[data['Contratar'] == 'Sim'  , 'Contratar'] = 1
        return data

class Encode_cargo(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print('Encoding cargo')
        data = X.copy()
        data.loc[data['Cargo'] == 'Engenheiro'      , 'Cargo'] = 0
        data.loc[data['Cargo'] == 'Tecnico'         , 'Cargo'] = 1
        data.loc[data['Cargo'] == 'Supervisor'      , 'Cargo'] = 2
        data.loc[data['Cargo'] == 'Analista'        , 'Cargo'] = 3
        data.loc[data['Cargo'] == 'Gerente'         , 'Cargo'] = 4
        data.loc[data['Cargo'] == 'Diretor'         , 'Cargo'] = 5
        data.loc[data['Cargo'] == 'Vendedo senior'  , 'Cargo'] = 6
        data.loc[data['Cargo'] == 'Vendedor junior' , 'Cargo'] = 7
        data.loc[data['Cargo'] == 'Assistente'      , 'Cargo'] = 8
        return data

class Encode_estcivil(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.le = le
        self.le.fit(['Casado','Solteiro','Divorciado'])
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print('Encoding estcivil')
        data = X.copy()
        data.loc[data['Estado civil'] == 'Casado'      , 'Estado civil'] = 0
        data.loc[data['Estado civil'] == 'Solteiro'    , 'Estado civil'] = 1
        data.loc[data['Estado civil'] == 'Divorciado'  , 'Estado civil'] = 2
        return data

class Encode_he(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print('Encoding HE')
        data = X.copy()
        data.loc[data['Necessita de hora extra'] == 'Não'  , 'Necessita de hora extra'] = 0
        data.loc[data['Necessita de hora extra'] == 'Sim'  , 'Necessita de hora extra'] = 1
        return data
