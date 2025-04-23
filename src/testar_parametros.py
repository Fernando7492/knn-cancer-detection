import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.MeuKnn import MeuKnn
from src.preprocessing import binarizar_col, escalar_col


def testar_parametros():
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
    for k in range(1,20):
        knn = MeuKnn(k)
        knn.fit(x_treino.values,y_treino.values)
        for opt in [True,False]:
            y_pred = knn.predict(x_teste.values,opt)
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


if __name__ == "__main__":
    testar_parametros()