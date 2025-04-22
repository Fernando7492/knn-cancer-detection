def binarizar_col(column,data):
    data[column] = data[column].map({'M':1, 'B':0})
    data[column].value_counts(normalize=True)
    return data