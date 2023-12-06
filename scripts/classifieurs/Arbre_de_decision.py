#Importation des librairies
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import confusion_matrix, classification_report 

# Classe pour modele d'arbre de décision
class Arbre_de_decision(object):
    
    # Constructeur de la classe 
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        
        self.x_train = x_train # Données d'entrainement
        self.y_train = y_train # Etiquettes 
        self.x_val = x_val # Données de validation
        self.y_val = y_val # Etiquettes de validation
        self.x_test = x_test # Données de test
        self.y_test = y_test # Etiquettes de test
        self.dt_classifier = DecisionTreeClassifier() # Initialisation du classificateur d'arbre de décision

    # Fonction pour effectuer la validation croisée et la recherche des meilleurs hyperparametres
    # avec gridsearch
    def validation_croisee_gridsearch(self):

        parameters = {
            'criterion': ['gini', 'entropy'], 
            'max_depth': [None, 10], 
            'min_samples_split': [2, 10], 
            'min_samples_leaf': [1, 4], 
            'max_features': ['auto'] 
        }

        # Création d'une instance de GridSearchCV avec 5 folds de validation croisée
        clf = GridSearchCV(self.dt_classifier, parameters, cv=5, n_jobs=-1, scoring='accuracy')
        # Entrainement du modele avec les paramétres
        clf.fit(self.x_train, self.y_train) 
        # Mise à jour du classificateur avec les meilleurs hyperparametres
        self.dt_classifier = clf.best_estimator_ 
        # Affichage des meilleurs hyperparamètres
        print("Meilleurs hyperparamètres:", clf.best_params_) 

    # Fonction pour entrainer le modèle
    def entrainement(self):
        
        self.validation_croisee_gridsearch() # Appel de la validation croisée
        self.dt_classifier.fit(self.x_train, self.y_train) # Entraînement du modèle

     # Fonction pour  faire des prediction sur les données de test
    def prediction(self):
       
        return self.dt_classifier.predict(self.x_test) # Retourne les prédictions

    # Fonction pour obtenir les probabilités de la prédiction
    def prediction_proba(self):
        
        return self.dt_classifier.predict_proba(self.x_test) # Retourne les probabilités de prédiction
