import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy import stats
import umap.umap_ as umap

class Pretraitement :


    def __init__(self, path: str):

        self.path = path

    def Charger_donnees(self) :
        
        return pd.read_csv(self.path)
    
    def Encoder_donnees(self, df: pd.DataFrame, etiquette :str):

        le = LabelEncoder()
        etiquette = le.fit_transform(df[etiquette])  
        classes = list(le.classes_)
        return etiquette,classes
    
    def Diviser_donnees(self,df :pd.DataFrame,etiquette : list) :

        x_train, x_test, y_train, y_test = train_test_split(df,etiquette, test_size=0.2, random_state=1)

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)
        return x_train, x_test, y_train, y_test,x_val, y_val


    def Normaliser(self,df : pd.DataFrame) :
        scaler=preprocessing.MinMaxScaler()
        df1=scaler.fit_transform(df)
        final = pd.DataFrame(df1,df.columns)
        return final

    def Normalize_dataframe(self,df : pd.DataFrame):
        col_num = df.select_dtypes(include=[np.number]).columns.tolist()
        df_normalizer = df.copy()
        scaler = preprocessing.MinMaxScaler()
        df_normalizer[col_num] = scaler.fit_transform(df_normalizer[col_num])

        return df_normalizer


 
    def Umap(self, data : pd.DataFrame, n_components : int):

        umap = umap(n_components=n_components)
        reduced_data = umap.fit_transform(data)
        return reduced_data


    def Indice_outliers(self, df :pd.DataFrame,seuil : float) :
        z_scores=stats.zscore(df)
        outliers = []
        for i in range (z_scores.shape[0]):
            cpt=0
            for j in range (z_scores.shape[1]):
                if (z_scores.iloc[i,j] > seuil or z_scores.iloc[i,j] < - seuil ) :
                    cpt += 1
            if (cpt>0.1*z_scores.shape[1]):
                outliers.append(i)
        
        return outliers

