import importlib
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelBinarizer

class Evaluation(object):
    def __init__(self, model, x_test, y_test,learning_curve_data):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.learning_curve_data = learning_curve_data 


    def calculate_metrics(self, y_true, y_pred):
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        return f1, precision, recall

    def plot_roc_curve(self):
        if hasattr(self.model, 'predict_proba') and callable(getattr(self.model, 'predict_proba', None)):
            y_pred_proba = self.model.predict_proba(self.x_test)

            lb = LabelBinarizer()
            y_test_bin = lb.fit_transform(self.y_test)

            n_classes = y_test_bin.shape[1]

            plt.figure(figsize=(8, 6))
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            plt.show()
        else:
            print("Model does not support predict_proba method.")
            
    def plot_learning_curves(self):
        if self.learning_curve_data is not None:
            data = self.learning_curve_data
            epochs = data['train_sizes']

            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(epochs, data['train_accuracy'], 'b', label='Training accuracy')
            plt.plot(epochs, data['val_accuracy'], 'r', label='Validation accuracy')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Training Size')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(epochs, data['train_loss'], 'b', label='Training loss')
            plt.plot(epochs, data['val_loss'], 'r', label='Validation loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Training Size')
            plt.ylabel('Loss')
            plt.legend()

            plt.show()
        else:
            print("No learning curve data available.")

    def plot_confusion_matrix(self):
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


