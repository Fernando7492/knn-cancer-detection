import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.MeuKnn import MeuKnn
from src.preprocessing import binarizar_col, escalar_col


def testar_parametros(criar_modelo,nome):
    data = pd.read_csv("data/breast-cancer-winsconsin-data.csv",sep=",",encoding="utf-8")
    data = data.drop(columns=['id'])

    #Transformando a coluna diagnosis em binário
    data = binarizar_col("diagnosis",data)

    df_escalado = escalar_col(data)

    x = df_escalado.drop(columns='diagnosis')
    y = df_escalado['diagnosis']
    x_treino, x_teste, y_treino, y_teste = train_test_split(x,y,test_size=0.2,random_state=42)

    lista_acuracias = []
    resultado = {
        "peso": [],
        "uniforme": []
    }
    for k in range(1,21):
        for opt in [True,False]:
            knn = criar_modelo(k,opt)
            knn.fit(x_treino,y_treino)
            if nome == "sklearn":
                y_pred = knn.predict(x_teste)
            else: 
                y_pred = knn.predict(x_teste, dist=opt)
            acuracia = accuracy_score(y_teste.values,y_pred)
            lista_acuracias.append(acuracia)
            if opt:
                resultado["peso"].append((k,acuracia))
            else:
                resultado["uniforme"].append((k,acuracia))
    exibir_resultados(resultado)
    
def exibir_resultados(resultado):
    for chave, valores in resultado.items():
        print(f"\nResultados para ponderação: {chave}")
        for k, acc in valores:
            print(f"k={k}, acurácia={acc:.4f}")

def criar_meu_knn(k, dist):
    return MeuKnn(k)

def criar_sklearn_knn(k, dist):
    return KNeighborsClassifier(n_neighbors=k, weights="distance" if dist else "uniform")

if __name__ == "__main__":
    print("Resultados MeuKnn:")
    testar_parametros(criar_meu_knn, "meu")

    print("\nResultados KNeighborsClassifier:")
    testar_parametros(criar_sklearn_knn, "sklearn")