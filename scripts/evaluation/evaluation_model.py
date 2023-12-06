import numpy as np
from sklearn.model_selection import learning_curve
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

    def calculate_metrics(self, y_true, y_pred):
        # Fonction pour calculer F1, precision et recall 
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        return f1, precision, recall

    def plt_roc_curve(self):
        # Fonction qui trace la courbe ROC du modele
        if hasattr(self.model, 'predict_proba') and callable(getattr(self.model, 'predict_proba', None)):
            y_pred_proba = self.model.predict_proba(self.x_test)
            lb = LabelBinarizer()
            y_test_bin = lb.fit_transform(self.y_test)
            n_classes = y_test_bin.shape[1]
            mean_roc_auc = 0.0
            plt.figure(figsize=(12, 8))
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                mean_roc_auc += roc_auc
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
            mean_roc_auc /= n_classes 

            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.subplots_adjust(right=0.75)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.suptitle(f'Mean AUC = {mean_roc_auc:.2f}', y=0.92)
            plt.show()
        else:
            print("Model does not support predict_proba method")

    def plt_confusion_matrix(self):
        # Fonction qui trace la matrice de confusion du modele
        y_pred = self.model.predict(self.x_test)
        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    def generate_classification_report(self):
        # Fonction pour afficher le rapport de classification du modele
        y_pred = self.model.predict(self.x_test)
        report = classification_report(self.y_test, y_pred)
        print("Rapport de classification :\n", report)

    def plot_learning_curves(self, train_sizes, scoring='accuracy'):
        # Fonctions pour tracee les courbes d'apprentissage pour evaluer la performance du modele
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, self.x_train, self.y_train, cv=5,
            train_sizes=train_sizes, scoring=scoring)
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=(12, 8))
        plt.plot(train_sizes, train_mean, label='Score d\'entraînement')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.plot(train_sizes, val_mean, label='Score de validation')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
        plt.title('Courbes d\'apprentissage')
        plt.xlabel('Taille de l\'ensemble d\'entraînement')
        plt.ylabel(scoring.capitalize())
        plt.legend(loc='best')
        plt.grid()
        plt.show()

    def plot_learning_curves_loss(self, train_sizes, scoring='neg_log_loss'):
        # Fonction pour tracer les courbes d'apprentissage de la perte pour evaluer la performance du modele
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, self.x_train, self.y_train, cv=5,
            train_sizes=train_sizes, scoring=scoring, shuffle=True, random_state=42)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=(12, 8))
        plt.plot(train_sizes, train_mean, label='Perte d\'entraînement')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.plot(train_sizes, val_mean, label='Perte de validation')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
        plt.title('Courbes d\'apprentissage - Perte')
        plt.xlabel('Taille de l\'ensemble d\'entraînement')
        plt.ylabel('Perte')
        plt.legend(loc='best')
        plt.grid()
        plt.show()
