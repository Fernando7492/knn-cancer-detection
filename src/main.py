import pandas as pd
from src.outliers_detection import outliers_por_coluna
from src.preprocessing import binarizar_col, escalar_col
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("data/breast-cancer-winsconsin-data.csv",sep=",",encoding="utf-8")
data = data.drop(columns=['id'])

#Transformando a coluna diagnosis em binário
data = binarizar_col("diagnosis",data)

#Verificando quantos outliers existem em cada coluna
print("Outliers por coluna:")
print(outliers_por_coluna(data))

df_escalado = escalar_col(data)

x = df_escalado.drop(columns='diagnosis')
y = df_escalado['diagnosis']

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y,test_size=0.2,random_state=42)
modelo_knn = KNeighborsClassifier(n_neighbors=5)
modelo_knn.fit(x_treino,y_treino)
