from sklearn.preprocessing import StandardScaler
import pandas as pd

def binarizar_col(column,data):
    data[column] = data[column].map({'M':1, 'B':0})
    data[column].value_counts(normalize=True)
    return data

def escalar_col(data):
    scaler = StandardScaler()
    data_para_escalar = data.drop(columns='diagnosis')
    data_escalado = scaler.fit_transform(data_para_escalar)
    df_escalado = pd.DataFrame(data_escalado, columns=data_para_escalar.columns, index=data.index)
    df_escalado["diagnosis"] = data["diagnosis"]
    return df_escalado
