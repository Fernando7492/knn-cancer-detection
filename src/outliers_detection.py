import pandas as pd

def calcular_iqr(data_num):
    q1 = data_num.quantile(0.25)
    q3 = data_num.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5*iqr,q3 + 1.5*iqr

def calcular_outliers(data):
    data_num = data.select_dtypes(include="number")
    lower, upper = calcular_iqr(data_num)
    outlier_por_coluna = {}
    for col in data_num.columns:
        outlier_col = ((data_num[col] > upper[col]) | (data_num[col]<lower[col])).sum()
        outlier_por_coluna[col] = outlier_col
    return outlier_por_coluna


def outliers_por_coluna(data):
    outliers_dict = calcular_outliers(data)
    return pd.Series(outliers_dict).sort_values(ascending=False)
    