import pandas as pd
from src.outliers_detection import outliers_por_coluna
from src.preprocessing import binarizar_col

data = pd.read_csv("data/breast-cancer-winsconsin-data.csv",sep=",",encoding="utf-8")
data = data.drop(columns=['id'])

#Transformando a coluna diagnosis em binário
data = binarizar_col("diagnosis",data)

#Verificando quantos outliers existem em cada coluna
print("Outliers por coluna:")
print(outliers_por_coluna(data))

print(data.shape)
print(data.dtypes)
print(data.describe().T)
print(data.isna().sum())


