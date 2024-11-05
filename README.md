# USP Data ICMC
Neste projeto é feita a classificação do Iris Dataset utilizando os modelos de machine learning Multi-layer Perceptron Classifier (MLPClassifier) e Support Vector Machine (SVM).

## Classificação de Espécies de Íris

Este projeto aplica técnicas de aprendizado de máquina para classificar diferentes espécies de flores do gênero Iris com base em medidas das sépalas e pétalas.

## Estrutura do Projeto

O projeto é organizado em uma classe chamada `Modelo` que contém métodos para carregar o conjunto de dados, realizar o pré-processamento, treinar e testar os modelos de machine learning.

### Métodos

1. **CarregarDataset(path)**: Carrega o conjunto de dados a partir de um arquivo CSV.

2. **TratamentoDeDados()**: Realiza o pré-processamento dos dados, incluindo a visualização inicial, análise de correlação e limpeza dos dados.

3. **Treinamento()**: Treina dois modelos de machine learning: um `MLPClassifier` (rede neural) e um `SVC` (classificador de vetores de suporte). Este método também inclui validação cruzada para avaliar a acurácia dos modelos.

4. **Teste(mlp, svc, X_test, y_test)**: Avalia o desempenho dos modelos treinados nos dados de teste, calculando métricas como acurácia, precisão, recall e F1-score.

5. **Train()**: Método principal que encapsula o fluxo de treinamento do modelo, chamando os métodos anteriores na ordem correta.

## Instalação

Para executar o código, você deve ter as seguintes bibliotecas instaladas:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Você pode instalar as dependências usando `pip` ou utilizar o Anaconda no ambiente de desenvolvimento.
