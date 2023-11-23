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


