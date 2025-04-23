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

    
    def achar_vizinhos(self,x_novo):
        dist = []
        for x in self.x_treino:
            dist.append(self.calcular_distancia(x_novo,x))
        dist_array = np.array(dist)
        d_indices_ord = np.argsort(dist_array)
        vizinhos_ind = d_indices_ord[:self.k]
        vizinhos_dist = dist_array[vizinhos_ind]
        return vizinhos_ind,vizinhos_dist
    
    def _votar(self,vizinhos_labels,vizinhos_distancias=None):
        
        votos = {}
        if vizinhos_distancias is None:
            for labels in vizinhos_labels:
                votos[labels] = votos.get(labels,0)+1
        else:
            for label,dist in zip(vizinhos_labels,vizinhos_distancias):
                peso = 1 / (dist + 1e-5)
                votos[label] = votos.get(label,0)+peso

        max_votos = max(votos.values())
        candidatos = [label for label,cont in votos.items() if cont == max_votos]
        mais_votado = max(candidatos)
        
        return mais_votado
    
    def predict(self,x_teste):
        lista = []
        
        for x_novo in x_teste:
            vizinhos_ind, vizinhos_dist = self.achar_vizinhos(x_novo)
            vizinhos_labels = self.y_treino[vizinhos_ind]
            lista.append(self._votar(vizinhos_labels,vizinhos_dist))
        
        return np.array(lista)