import importlib
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import learning_curve , validation_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelBinarizer 

class Evaluation(object):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.x_train = x_train
        self.y_train = y_train
        #self.learning_curve_data = learning_curve_data 


    def calculate_metrics(self, y_true, y_pred):
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        return f1, precision, recall

    def plt_roc_curve(self):
        if hasattr(self.model, 'predict_proba') and callable(getattr(self.model, 'predict_proba', None)):
            y_pred_proba = self.model.predict_proba(self.x_test)
 
            lb = LabelBinarizer()
            y_test_bin = lb.fit_transform(self.y_test)
 
            n_classes = y_test_bin.shape[1]
            mean_roc_auc = 0.0
 
            # Agrandir la figure
            plt.figure(figsize=(12, 8))
 
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                mean_roc_auc += roc_auc
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
 
            mean_roc_auc /= n_classes  # Calculer la moyenne de ROC AUC
 
            # Tracer la ligne en pointillés
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
 
            # Ajouter les étiquettes à côté du graphe
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.subplots_adjust(right=0.75)  # Ajuster la disposition pour accueillir les étiquettes
 
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
           
            # Ajouter la moyenne de ROC AUC dans le titre
            plt.suptitle(f'Mean AUC = {mean_roc_auc:.2f}', y=0.92)
 
            plt.show()
        else:
            print("Model does not support predict_proba method.")
            


    def plt_confusion_matrix(self):
        y_pred = self.model.predict(self.x_test)
        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    def generate_classification_report(self):
        y_pred = self.model.predict(self.x_test)

        report = classification_report(self.y_test, y_pred)
        print("Classification Report:\n", report)


 
    # def plot_learning_curves(self, train_sizes, scoring='accuracy'):

    #     train_sizes, train_scores, val_scores = learning_curve(
    #         self.model, self.x_train, self.y_train, cv=5,
    #         train_sizes=train_sizes, scoring=scoring)
 
    #     train_mean = np.mean(train_scores, axis=1)
    #     train_std = np.std(train_scores, axis=1)
    #     val_mean = np.mean(val_scores, axis=1)
    #     val_std = np.std(val_scores, axis=1)
        
    #     plt.figure(figsize=(12, 8))
    #     plt.plot(train_sizes, train_mean, label='Score d\'entraînement')
    #     plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    #     plt.plot(train_sizes, val_mean, label='Score de validation')
    #     plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    #     plt.title('Courbes d\'apprentissage')
    #     plt.xlabel('Taille de l\'ensemble d\'entraînement')
    #     plt.ylabel(scoring.capitalize())

    #     plt.legend(loc='best')
    #     plt.grid()
    #     plt.show()

    def plot_learning_curves_loss(self, train_sizes):
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, self.x_train, self.y_train, cv=5,
            train_sizes=train_sizes, scoring='accuracy')

        # Calcul de la perte comme 1 - accuracy
        train_loss = 1 - np.mean(train_scores, axis=1)
        train_loss_std = np.std(train_scores, axis=1)
        test_loss = 1 - np.mean(test_scores, axis=1)
        test_loss_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(12, 8))
        plt.plot(train_sizes, train_loss, 'o-', color="r", label="Perte d'entraînement")
        plt.fill_between(train_sizes, train_loss - train_loss_std, train_loss + train_loss_std, alpha=0.1, color="r")
        plt.plot(train_sizes, test_loss, 'o-', color="g", label="Perte de test")
        plt.fill_between(train_sizes, test_loss - test_loss_std, test_loss + test_loss_std, alpha=0.1, color="g")

        plt.title('Courbes de perte d\'apprentissage')
        plt.xlabel('Taille de l\'ensemble d\'entraînement')
        plt.ylabel('Perte')
        plt.legend(loc='best')
        plt.grid()
        plt.show()
    
    def plot_learning_curves_accuracy(self, train_sizes):
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, self.x_train, self.y_train, cv=5,
            train_sizes=train_sizes, scoring='accuracy')

        # Calcul de la moyenne et de l'écart-type des scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(12, 8))
        plt.plot(train_sizes, train_mean, 'o-', color="r", label="Précision d'entraînement")
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
        plt.plot(train_sizes, test_mean, 'o-', color="g", label="Précision de test")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")

        plt.title('Courbes d\'apprentissage de précision')
        plt.xlabel('Taille de l\'ensemble d\'entraînement')
        plt.ylabel('Précision')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

