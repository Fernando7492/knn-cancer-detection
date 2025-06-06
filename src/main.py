import pandas as pd
from src.preprocessing import binarizar_col, escalar_col
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from src.testar_modelo import testar_modelo
from src.MeuKnn import MeuKnn

k = 9

data = pd.read_csv("data/breast-cancer-winsconsin-data.csv",sep=",",encoding="utf-8")
data = data.drop(columns=['id'])

#Transformando a coluna diagnosis em binário
data = binarizar_col("diagnosis",data)

df_escalado = escalar_col(data)

x = df_escalado.drop(columns='diagnosis')
y = df_escalado['diagnosis']
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y,test_size=0.2,random_state=42)

#Sklearn - KNN
modelo_knn = KNeighborsClassifier(n_neighbors=k)
modelo_knn.fit(x_treino,y_treino)
y_pred = modelo_knn.predict(x_teste)

#MeuKnn
meuKnn = MeuKnn(k=k)
meuKnn.fit(x_treino,y_treino)
y_pred_manual = meuKnn.predict(x_teste)


testar_modelo(y_teste,y_pred)
testar_modelo(y_teste,y_pred_manual)

