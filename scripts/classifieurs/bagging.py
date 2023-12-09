#Importation des librairies
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Classe pour modele Bagging
class Bagging(object):

    # Constructeur de la classe
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train # Données d'entrainement
        self.y_train = y_train # Etiquettes
        self.x_val = x_val # Données de validation
        self.y_val = y_val # Etiquettes de validation
        self.x_test = x_test # Données de test
        self.y_test = y_test # Etiquettes de test
        self.bg_classifier = BaggingClassifier() # Initialisation du classificateur Bagging

    # Fonction pour effectuer la validation croisée et la recherche des meilleurs hyperparametres
    # avec gridsearch
    def validation_croisee_gridsearch(self):
        parameters = {
            'n_estimators': [30, 50, 100, 200],
            'max_samples': [0.5, 0.7, 0.9, 1.0],
            'max_features': [0.5, 0.7, 0.9, 1.0],
            'base_estimator': [DecisionTreeClassifier(max_depth=1), 
                               DecisionTreeClassifier(max_depth=2)]
        }

        # Création d'une instance de GridSearchCV avec 5 folds de validation croisée
        clf = GridSearchCV(self.bg_classifier, parameters, cv=5)
        # Entrainement du modele avec les paramétres
        clf.fit(self.x_train, self.y_train)
        # Mise à jour du classificateur avec les meilleurs hyperparametres
        self.bg_classifier.set_params(**clf.best_params_)
        # Affichage des meilleurs hyperparamètres
        print("Meilleurs hyperparamètres pour Bagging:", clf.best_params_)

    # Fonction pour entrainer le modèle
    def entrainement(self):
        self.validation_croisee_gridsearch()
        self.bg_classifier.fit(self.x_train, self.y_train)

    # Fonction pour  faire des prediction sur les données de test
    def prediction(self):
        return self.bg_classifier.predict(self.x_test)
    
    # Fonction pour obtenir les probabilités de la prédiction
    def prediction_proba(self):
        return self.bg_classifier.predict_proba(self.x_test)
