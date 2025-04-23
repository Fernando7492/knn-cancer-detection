from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def testar_modelo(y_teste,y_pred):
    #Acuracia
    acuracia = accuracy_score(y_teste,y_pred)

    #Report de classificação
    report = classification_report(y_teste,y_pred,target_names=['Benigno','Maligno'])

    #matriz de confusão
    cm = confusion_matrix(y_teste,y_pred)


    print(f'ACURACIA: {acuracia}')
    print(report)
    print("Matriz de Confusão:")
    print("                  Predito")
    print("              Benigno  Maligno")
    print(f"Real Benigno     {cm[0][0]:<8} {cm[0][1]}")
    print(f"Real Maligno     {cm[1][0]:<8} {cm[1][1]}")