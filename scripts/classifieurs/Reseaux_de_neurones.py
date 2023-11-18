from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.neural_network import MLPClassifier

class Reseaux_de_neurones(object):
 
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):

        self.hidden_layer_sizes = (100, 100)  
        self.activation = 'relu'
        self.alpha = 0.0001
        self.learning_rate_init = 0.001
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
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
 
    def preprocess_data(self):
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_val = scaler.transform(self.x_val)
        self.x_test = scaler.transform(self.x_test)
        
    def validation_croisee_gridsearch(self):
        param_grid = {
            'hidden_layer_sizes': [(50, 50), (100, 100), (100, 50), (100,)],
            'alpha': [0.0001, 0.001, 0.01],
            'activation': ['relu', 'tanh', 'logistic'],
            'learning_rate_init': [0.1, 0.01, 0.001],
        }
 
        grid_search = GridSearchCV(MLPClassifier(solver='lbfgs'),
                                   param_grid, cv=2, n_jobs=-1)
        combined_x = np.concatenate([self.x_train, self.x_val])
        combined_y = np.concatenate([self.y_train, self.y_val])
        grid_search.fit(combined_x, combined_y)
        self.best_hyperparameters = grid_search.best_params_  
        print("Best hyperparameters:", self.best_hyperparameters)
        return combined_x, combined_y
 
    def garder_meilleur_hyperparameters(self):
        combined_x, combined_y = self.validation_croisee_gridsearch()
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

    def entrainement(self):
        self.nn_classifier.fit(self.x_train, self.y_train)
        train_sizes, train_scores, test_scores = learning_curve(
            self.nn_classifier, self.x_train, self.y_train, cv=5, scoring="accuracy"
        )

        learning_curve_data = {
            "train_sizes": train_sizes,
            "train_accuracy": np.mean(train_scores, axis=1),
            "val_accuracy": np.mean(test_scores, axis=1),
            "train_loss": np.mean(train_scores, axis=1), 
            "val_loss": np.mean(test_scores, axis=1)  
        }
        self.learning_curve_data = learning_curve_data

    def prediction(self):
        return self.nn_classifier.predict(self.x_test)

    def prediction_proba(self):
        return self.nn_classifier.predict_proba(self.x_test)

    def resultats_model(self):
        y_pred = self.nn_classifier.predict(self.x_test)
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
