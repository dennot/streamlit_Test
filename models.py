import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree

# Llamamos a la clase del módulo de creamos previamente:
from utils import Utils

class Models:
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    def tree_training(self, X,y):
        
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .2, random_state = 99)
        
        # Definimos las características que deseamos se integren en nuestro árbol: 
        t = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 7)
        
        # Entrenamos el modelo con nuestros dataset de prueba:
        model = t.fit(X_train, y_train)
        
        # Calculamos las métricas de nuestro modelo
        score_entrenamiento = model.score(X_train, y_train)
        score_prueba = model.score(X_test, y_test)
        
        return score_entrenamiento, score_prueba