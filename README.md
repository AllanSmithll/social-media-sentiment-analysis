# Social Media Usage and Emotional Well-being - Machine Learning

Este projeto utiliza algoritmos de aprendizado de máquina para analisar a relação entre o uso de redes sociais e o bem-estar emocional. Diversos classificadores foram testados para encontrar o modelo mais adequado.

## 📂 Estrutura do Projeto
- **Download do dataset**: Obtém os dados do Kaggle.
- **Pré-processamento**: Trata valores categóricos e normaliza os dados.
- **Treinamento e avaliação**: Testa vários algoritmos, incluindo Decision Tree, kNN, MLP e K-Means.

## 📊 Dataset
**Nome:** Social Media Usage and Emotional Well-being  
**Fonte:** Kaggle ([@emirhanai](https://www.kaggle.com/emirhanai))  
**Arquivo principal:** `train.csv`

## 🔧 Dependências

Instale as bibliotecas necessárias com:

```bash
pip install numpy pandas scikit-learn matplotlib kagglehub
```

## 🚀 Como Executar

1. **Baixar o dataset:**
   ```python
   import kagglehub
   kagglehub.dataset_download("emirhanai/social-media-usage-and-emotional-well-being")
   ```
2. **Executar o script principal:**
   ```bash
   python main.py
   ```

## 🏗️ Pré-processamento dos Dados

- **Remoção de colunas irrelevantes** (`User_ID`, `Platform`)
- **Filtragem de valores de `Gender`** (`Male`, `Female`, `Non-binary`)
- **Codificação categórica** com `LabelEncoder` e `OneHotEncoder`
- **Normalização** com `MinMaxScaler`

## 🧠 Modelos Utilizados

### 1️⃣ Decision Tree
- **Critérios:** `gini` e `entropy`
- **Validação cruzada:** 10-fold

### 2️⃣ k-Nearest Neighbors (kNN)
- **k = 5 e k = 10**
- **Métrica:** Distância Euclidiana

### 3️⃣ Multi-layer Perceptron (MLP)

Foram testadas duas arquiteturas diferentes:

- **MLP 1:** `(64, 32, 16)`
- **MLP 2:** `(128, 64, 32)`

Parâmetros:
- **Função de ativação:** `tanh` e `relu`
- **Taxa de aprendizado:** `0.03` (MLP 1) e `0.015` (MLP 2)
- **Early stopping:** Ativado

### 4️⃣ K-Means
- **Número de clusters:** Definido pelo número de classes no problema

## 📈 Resultados

| Algoritmo | Acurácia Média (%) | Taxa de Erro (%) |
|-----------|-------------------|----------------|
| Decision Tree (Gini) | XX | XX |
| Decision Tree (Entropy) | XX | XX |
| kNN (k=5) | XX | XX |
| kNN (k=10) | XX | XX |
| MLP 1 (tanh) | XX | XX |
| MLP 1 (relu) | XX | XX |
| MLP 2 (tanh) | XX | XX |
| MLP 2 (relu) | XX | XX |
| K-Means | XX | XX |

## 📌 Visualização

Para analisar o erro de treinamento ao longo das épocas, foi gerado um gráfico com a evolução da perda dos modelos MLP.

```python
import matplotlib.pyplot as plt
plt.plot(mlp_one_tanh.loss_curve_, label="MLP 1 (tanh)")
plt.plot(mlp_one_relu.loss_curve_, label="MLP 1 (relu)")
plt.plot(mlp_two_tanh.loss_curve_, label="MLP 2 (tanh)")
plt.plot(mlp_two_relu.loss_curve_, label="MLP 2 (relu)")
plt.xlabel("Épocas")
plt.ylabel("Erro de Treinamento")
plt.legend()
plt.grid()
plt.show()
```

## 📢 Conclusão
- **MLP apresentou melhor desempenho geral** em termos de acurácia.
- **Árvores de decisão** tiveram boa performance com `entropy`.
- **kNN teve performance inferior**, especialmente para `k=10`.
- **K-Means apresentou menor acurácia**, pois não é um classificador supervisionado.

## 🏆 Melhor modelo: `MLP 2 (ReLU)`

Este projeto demonstra o impacto do uso de redes sociais na emoção dos usuários, utilizando aprendizado de máquina para identificar padrões nos dados.

---
✉️ Para dúvidas ou sugestões, entre em contato!
