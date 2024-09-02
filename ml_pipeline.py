import sklearn.datasets as ds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (r2_score,
                             accuracy_score, precision_score,
                             recall_score, f1_score,
                             silhouette_score)
import numpy as np
import csv
import os
from pathlib import Path
import pickle
import pandas as pd
os.chdir(Path(__file__).parent)

from typing import Literal

class MLPipeline:
    Datasets = ["Iris", "Digits", "Breast_cancer"]
    ModelNames = ["Linear Regression", "Clustering",
                        "Logistic Regression", "Decission Tree"]

    Data = Literal["Iris", "Digits", "Breast_cancer"]
    ModelName = Literal["Linear Regression", "Clustering",
                    "Logistic Regression", "Decission Tree"]

    def __init__(self, name:Data) -> None:
        data = self.load_dataset(name)
        self.name = name
        self.X_origin = data.data
        self.X = self.X_origin.copy()
        self.y_origin = data.target
        self.y = self.y_origin.copy()
        self.feature_names = data.feature_names
        self.target_names = data.target_names
        self.model = None
        self.docu = {"dataset": name}

    def load_dataset(self,choice:Data):
        """load one of Datasets: "Iris", "Breast_cancer" or "Digits"."""
        if choice == "Iris":
            data = ds.load_iris(as_frame=True)
            self.clusters = 3
        elif choice == "Breast_cancer":
            data = ds.load_breast_cancer(as_frame=True)
            self.clusters = 2
        else:        # "Digits"
            data = ds.load_digits(as_frame=True)
            self.clusters = 10
        return data
    
    def preprocess_data(self,
            choose_scaling:None|Literal["MinMaxScaler","StandardScaler"]=None,
            pca_choice:None|str=None,
            pca_arg:None|int|float=None) -> bool:
        """split and scale data, perform PCA as selected.
        
        Args:
            choose_scaling: None|Literal["MinMaxScaler","StandardScaler"]  
            pca_choice: None|Literal["Percentage of Variance","Number of components"]  
            pca_arg: None|int|float  
            - between 0 and 100 (exclusive) if pca_choice is 'Percentage of Variance'
            - between 1 and the minimum number of features/instances if pca_choice is 'Number of components'."""
        # 0 for documentation:
        self.docu["scaling"] = choose_scaling
        self.docu["pca_method"] = pca_choice
        self.docu["pca_arg"] = pca_arg
        
        
        # 1 split in train & test:
        X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
                )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # 2 Scale
        check = True
        if choose_scaling:
            if choose_scaling == "MinMaxScaler":
                scaler = MinMaxScaler()
            elif choose_scaling == "StandardScaler":
                scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)

        # 3 PCA    
            if pca_choice:
                if pca_choice == "Percentage of Variance":
                    arg = pca_arg / 100
                else:
                    arg = pca_arg
                try:
                    pca = PCA(n_components=arg)
                    self.X_train = pca.fit_transform(self.X_train)
                    self.X_test = pca.transform(self.X_test)
                except Exception as e:
                    print(e)
                    check = False

        self.docu["columns_after_pca"] = self.X_train.shape[1]
        return check
    
    def create_fit_model(self,modelname:ModelName):
        """Create a Machine Learning Model trained on X_train."""
        X,y = self.X_train, self.y_train
        if modelname == "Linear Regression":
            model = LinearRegression()
            model.fit(X,y)

        elif modelname == "Logistic Regression":
            model = LogisticRegression(solver="liblinear", tol=0.0001)
            model.fit(X,y)

        elif modelname == "Decission Tree":
            model = DecisionTreeClassifier()
            model.fit(X,y)

        elif modelname == "Clustering":
            n = self.clusters
            model = KMeans(n_clusters=n,
                           n_init="auto").fit(X)
        self.model = model
        self.docu["model"] = str(self.model)
        self.modelname = modelname

    def evaluate(self) -> tuple[float]:
        self.y_pred = self.model.predict(self.X_test)
        if self.modelname == "Linear Regression":
            evaluation = r2_score(self.y_test, self.y_pred)
            self.ev_info = "R²-score"
            
        elif self.modelname == "Clustering":
            evaluation = silhouette_score(self.X_test, self.y_test)
            self.ev_info = ("**silhouette score** ranges from -1 to 1,\n\n"
                            "where a higher value indicates better-defined clusters.")
        else:
            self.y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, self.y_pred)
            precision = precision_score(self.y_test, self.y_pred,average='macro', zero_division=np.nan)
            recall = recall_score(self.y_test, self.y_pred,average='macro', zero_division=np.nan)
            f1score = f1_score(self.y_test, self.y_pred,average='macro', zero_division=np.nan)
            evaluation = tuple(np.array([accuracy,precision,recall,f1score]).round(3))
            self.ev_info = "accuracy, precision, recall, f1score"
        self.docu["evaluation"] = evaluation
        return evaluation
    
    def documentation_to_csv(self, ev,
                             file:Path="./Evaluation/model_comparison.csv") -> None:
        fieldnames = ["dataset","model",
                      "scaling","pca_method","pca_arg","columns_after_pca",
                      "evaluation"]
        
        if os.path.exists(file):
            with open(file, mode="a", newline="") as f:
                writer = csv.DictWriter(f,fieldnames=fieldnames)
                writer.writerow(self.docu)
        else:
            with open(file, mode="a", newline="") as f:
                writer = csv.DictWriter(f,fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(self.docu)
        return None
    
    def save_model(self, filename:Path="./model1.pkl"):
        with open(filename, mode="wb") as file:
            pickle.dump(self.model, file)

    @staticmethod
    def load_model(filename:Path):
        """just for completeness"""
        with open(filename, mode="rb") as file:
            model = pickle.load(file)
        return model


if __name__ == "__main__":
    models = ["Linear Regression", "Clustering",
              "Logistic Regression", "Decission Tree"]
    
    print('Evaluation:')
    for el in MLPipeline.Datasets:
        mlp = MLPipeline(el)
        mlp.preprocess_data()
        mlp.create_fit_model("Logistic Regression")
        ev = mlp.evaluate()
        print(f'{el}:\t', ev)
        mlp.documentation_to_csv(ev)
        # mlp.save_model()
