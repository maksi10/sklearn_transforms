from sklearn.base import BaseEstimator, TransformerMixin


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


