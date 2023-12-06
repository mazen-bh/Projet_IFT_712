import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy import stats

class Pretraitement:

    def __init__(self, path: str):
        self.path = path

    def Charger_donnees(self):
        # Fonction pour charger les donnees a partir d'un fichier CSV
        return pd.read_csv(self.path)

    def Encoder_donnees(self, df: pd.DataFrame, etiquette: str):
        # Fonction pour encoder une colonne du dataframe en utilisant LabelEncoder
        le = LabelEncoder()
        etiquette_encoded = le.fit_transform(df[etiquette])
        classes = list(le.classes_)
        return etiquette_encoded, classes

    def Diviser_donnees(self, df: pd.DataFrame, etiquette: list):
        #Fonction pour diviser les donnees en ensembles d'entrainement, de test et de validation
        x_train, x_test, y_train, y_test = train_test_split(df, etiquette, test_size=0.2, random_state=1)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)
        return x_train, x_test, y_train, y_test, x_val, y_val

    def Indice_outliers(self, df: pd.DataFrame, seuil: float):
        # Fonction pour trouver les indices des outliers avec Z-score
        z_scores = stats.zscore(df)
        outliers = []
        for i in range(z_scores.shape[0]):
            cpt = 0
            for j in range(z_scores.shape[1]):
                if (z_scores[i, j] > seuil or z_scores[i, j] < -seuil):
                    cpt += 1
            if (cpt > 0.1 * z_scores.shape[1]):
                outliers.append(i)
        return outliers

    def get_num_classes(data_frame, class_column='class'):
        # Fonction qui retourne le nombre de classes uniques
        num_classes = data_frame[class_column].nunique()
        return num_classes




