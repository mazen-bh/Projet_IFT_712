#Importation des librairies
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.neural_network import MLPClassifier

# Classe pour modele Reseaux de neurones
class Reseaux_de_neurones(object):
    # Constructeur de la classe
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):

        self.hidden_layer_sizes = (100, 100)  
        self.activation = 'relu'
        self.alpha = 0.0001
        self.learning_rate_init = 0.001
        self.x_train = x_train # Données d'entrainement
        self.y_train = y_train # Etiquettes
        self.x_val = x_val # Données de validation
        self.y_val = y_val # Etiquettes de validation
        self.x_test = x_test # Données de test
        self.y_test = y_test # Etiquettes de test
        self.learning_curve_data = None 
        self.y_test = y_test
        self.best_hyperparameters = None 
        self.nn_classifier = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=10,
            solver='lbfgs',  

        )
    # Standardise les données d'entraînement, de validation et de test en utilisant StandardScaler pour 
    # une cohérence dans le modèle.
    def preprocess_data(self):
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_val = scaler.transform(self.x_val)
        self.x_test = scaler.transform(self.x_test)
    
    # Fonction pour effectuer la validation croisée et la recherche des meilleurs hyperparametres
    # avec gridsearch
    def validation_croisee_gridsearch(self):
        # Configuration des hyperparamètres pour la recherche par grille
        param_grid = {
            'hidden_layer_sizes': [(50, 50), (50, 30), (30,)],
            'alpha': [0.001, 0.01, 0.1],
            'activation': ['relu', 'tanh', 'logistic'],
            'learning_rate_init': [0.1, 0.01, 0.001],
        }

        # Création et exécution de la recherche par grille
        grid_search = GridSearchCV(MLPClassifier(solver='lbfgs'), param_grid, cv=2, n_jobs=-1)
        combined_x = np.concatenate([self.x_train, self.x_val])
        combined_y = np.concatenate([self.y_train, self.y_val])
        # Entrainement du modele avec les paramétres
        grid_search.fit(combined_x, combined_y)
        # Mise à jour des meilleurs hyperparamètres
        self.best_hyperparameters = grid_search.best_params_
        print("Best hyperparameters:", self.best_hyperparameters)

        self.nn_classifier = MLPClassifier(
            hidden_layer_sizes=self.best_hyperparameters['hidden_layer_sizes'],
            activation=self.best_hyperparameters['activation'],
            alpha=self.best_hyperparameters['alpha'],
            learning_rate_init=self.best_hyperparameters['learning_rate_init'],
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=10,
            solver='lbfgs',
        )

        
    # Fonction pour entrainer le modèle
    def entrainement(self):
        self.nn_classifier.fit(self.x_train, self.y_train)
        train_sizes, train_scores, test_scores = learning_curve(
            self.nn_classifier, self.x_train, self.y_train, cv=5, scoring="accuracy"
        )

    # Fonction pour  faire des prediction sur les données de test
    def prediction(self):
        return self.nn_classifier.predict(self.x_test)
    # Fonction pour obtenir les probabilités de la prédiction
    def prediction_proba(self):
        return self.nn_classifier.predict_proba(self.x_test)
