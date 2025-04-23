import math
import numpy as np

class MeuKnn:
    """Classe para calcular o K-vizinhos mais próximos, usando distância euclidiana.
    """
    def __init__(self, k=5):
        """

        Args:
            k (int, optional): Número de vizinhos próximos que será usado pelo algortimo. Valor padrão = 5.
        """
        self.k = k
        
    def fit(self,x_treino, y_treino):
        """Função para armazenar os valores de treino na classe

        Args:
            x_treino (Numpy array): Um array numpy contendo os atributos dos elementos
            y_treino (Numpy array): Um array numpy contendo as classes dos elementos
        """
        self.x_treino = x_treino
        self.y_treino = y_treino


    def calcular_distancia(self,x,y):
        d = 0
        for x1,x2 in zip(x,y):
            d += (x1 - x2)**2
        return math.sqrt(d)