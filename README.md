# 🔍 KNN do Zero - Classificador de Câncer de Mama

Este projeto implementa o algoritmo K-Nearest Neighbors (KNN) **do zero** em Python, comparando seus resultados com o `KNeighborsClassifier` da biblioteca `scikit-learn`. A base utilizada é a de diagnóstico de câncer de mama (Breast Cancer Wisconsin).

---


## 🧠 Funcionalidades

- Implementação personalizada do KNN (`MeuKnn`)
- Suporte a ponderação por distância (peso) ou voto uniforme
- Comparação com o KNN da `scikit-learn`
- Avaliação da acurácia para diferentes valores de `k`
- Visualização dos resultados com gráficos

---

## 🚀 Como Rodar
```
# Clone o repositório
git clone https://github.com/Fernando7492/knn-cancer-detection
cd knn-cancer-detection

# (Recomenda-se o uso de um ambiente virtual)
python3 -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate no Windows

# Instale as dependências
pip install -r requirements.txt

# Execute o algoritmo com K = 9 para ambos os modelos e verificar diversas métricas, como  acuracia e matriz de confusão
python3 -m src.main

# Execute os testes de acurácia
python3 -m src.testar_parametros
```
---

## 📁 ESTRUTURA DO PROJETO

```
.
├── data
│   └── breast-cancer-winsconsin-data.csv
├── LICENSE
├── README.md
├── requirements.txt
└── src
    ├── __init__.py
    ├── main.py
    ├── MeuKnn.py
    ├── outliers_detection.py
    ├── preprocessing.py
    ├── testar_modelo.py
    └── testar_parametros.py
```
## 🧰 Tecnologias Usadas

    Python 3.10.12

    Pandas

    NumPy

    scikit-learn

## 📊 RESULTADOS OBTIDOS

Melhor acurácia: 96.49% (para k=9) em ambos os modelos e votações

## 👤 Autor

[<img loading="lazy" src="https://avatars.githubusercontent.com/u/112771403?v=4" width=115><br><sub>Fernando Emidio</sub>](https://github.com/Fernando7492)

---

## ⚖️ Licença

Este projeto está sob a licença [MIT](./LICENSE).  

Veja o arquivo [LICENSE](./LICENSE) para detalhes.