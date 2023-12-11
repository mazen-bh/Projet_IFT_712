#Importation des librairies
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
from pandas.plotting import table

# Classe pour Le pretraitement des données
class Pretraitement:
    # Constructeur de la classe
    def __init__(self, path: str):
        self.path = path

    # Fonction pour charger les donnees a partir d'un fichier CSV
    def Charger_donnees(self):
        
        return pd.read_csv(self.path)

    # Fonction pour encoder une colonne du dataframe en utilisant LabelEncoder
    def Encoder_donnees(self, df: pd.DataFrame, etiquette: str):
        
        le = LabelEncoder()
        etiquette_encoded = le.fit_transform(df[etiquette])
        classes = list(le.classes_)
        return etiquette_encoded, classes

    #Fonction pour diviser les donnees en ensembles d'entrainement, de test et de validation
    def Diviser_donnees(self, df: pd.DataFrame, etiquette: list):
        
        x_train, x_test, y_train, y_test = train_test_split(df, etiquette, test_size=0.2, random_state=1)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)
        return x_train, x_test, y_train, y_test, x_val, y_val


    # Fonction pour trouver les indices des outliers avec Z-score
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
    
    # Fonction qui retourne le nombre de classes uniques
    def get_num_classes(data_frame, class_column='class'):
        
        num_classes = data_frame[class_column].nunique()
        return num_classes

    # Compter le nombre d'éléments dans chaque classe
    def plot_class_distribution(self, df, class_column):
        class_counts = df[class_column].value_counts()

        table_df = pd.DataFrame({
            'Classe': class_counts.index,
            'Nombre d\'éléments': class_counts.values
        })

        # Ajuster la taille de la figure et la largeur des colonnes
        fig, ax = plt.subplots(figsize=(12, 6))  # Augmentation de la taille de la figure
        ax.axis('off')
        colWidths = [0.3, 0.3]  # Ajuster en fonction de la longueur des noms de vos classes
        tbl = table(ax, table_df, loc='center', cellLoc='center', colWidths=colWidths)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1.2, 1.2)

        plt.show()




