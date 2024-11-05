import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


class Modelo():

    def __init__(self):
        pass

    def CarregarDataset(self, path):
        """
        Carrega o conjunto de dados a partir de um arquivo CSV.

        Parâmetros:
        - path (str): Caminho para o arquivo CSV contendo o dataset.
        
        O dataset é carregado com as seguintes colunas: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm e Species.
        """
        names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
        self.df = pd.read_csv(path, names=names)


    def TratamentoDeDados(self):
        """
        Realiza o pré-processamento dos dados carregados.

        Sugestões para o tratamento dos dados:
            * Utilize `self.df.head()` para visualizar as primeiras linhas e entender a estrutura.
            * Verifique a presença de valores ausentes e faça o tratamento adequado.
            * Considere remover colunas ou linhas que não são úteis para o treinamento do modelo.
        
        Dicas adicionais:
            * Explore gráficos e visualizações para obter insights sobre a distribuição dos dados.
            * Certifique-se de que os dados estão limpos e prontos para serem usados no treinamento do modelo.
        """
        
        print(self.df.head(), '\n')
        print(self.df.info(), '\n')
        print(f'Shape: {self.df.shape}', '\n')

        # Os dados têm shape (150, 5) com zero valores nulos.

        # Criando uma figura para o heatmap
        plt.figure(figsize=(9, 7))
        # Gera um gráfico matricial para apresentar a correção entre as variáveis de entrada do dataset
        ax = sns.heatmap(self.df.corr(numeric_only=True), annot=True, cmap=sns.cubehelix_palette(as_cmap=True))
        ax.set_title('Heatmap — Correlação entre Variáveis')
        ax=ax
        plt.show()
        plt.close()

        # O gráfico mostrou uma correlação de 0.96 (96%) entre as variáveis PetalLengthCm e PetalWidthCm, portanto, vamos remover a coluna referente à variável PetalWidthCm
        self.df.drop(self.df.columns[3], axis=1, inplace=True)

        # Criar uma figura e uma grade de subplots
        fig, axes = plt.subplots(1, 3, figsize=(17, 7))  # 1 linha, 3 colunas

        # Scatter plot 1: SepalLengthCm vs SepalWidthCm
        sns.scatterplot(data=self.df, x='SepalLengthCm', y='SepalWidthCm', hue='Species', palette='Dark2', ax=axes[0])
        axes[0].set_xlabel('SepalLengthCm')
        axes[0].set_ylabel('SepalWidthCm')
        axes[0].set_title('SepalLengthCm vs SepalWidthCm (with Species as color)')

        # Scatter plot 2: SepalLengthCm vs PetalLengthCm
        sns.scatterplot(data=self.df, x='SepalLengthCm', y='PetalLengthCm', hue='Species', palette='Dark2', ax=axes[1])
        axes[1].set_xlabel('SepalLengthCm')
        axes[1].set_ylabel('PetalLengthCm')
        axes[1].set_title('SepalLengthCm vs PetalLengthCm (with Species as color)')

        # Scatter plot 3: PetalLengthCm vs SepalWidthCm
        sns.scatterplot(data=self.df, x='PetalLengthCm', y='SepalWidthCm', hue='Species', palette='Dark2', ax=axes[2])
        axes[2].set_xlabel('PetalLengthCm')
        axes[2].set_ylabel('SepalWidthCm')
        axes[2].set_title('PetalLengthCm vs SepalWidthCm (with Species as color)')

        # Ajustar o layout
        plt.tight_layout()
        # Exibir o gráfico
        plt.show()
        plt.close()

        # Percebe-se que a espécie Iris-Setosa se destaca bastante das outras duas, tendo seus valores distantes das outras duas classes nos scatter plots.
        # As espécies Iris-Versicolor e Iris-Virginica se mesclam em alguns plots, principalmente no SepalLengthCm vs SepalWidthCm.


    def Treinamento(self):
        """
        Treina o modelo de machine learning.

        Detalhes:
            * Utilize a função `train_test_split` para dividir os dados em treinamento e teste.
            * Escolha o modelo de machine learning que queira usar. Lembrando que não precisa ser SVM e Regressão linear.
            * Experimente técnicas de validação cruzada (cross-validation) para melhorar a acurácia final.
        
        Nota: Esta função deve ser ajustada conforme o modelo escolhido.
        """
        X = self.df.drop('Species', axis=1)
        y = self.df['Species']

        scaler = MinMaxScaler().fit(X) # Cria o modelo para o ajuste
        X = scaler.transform(X) # Aplica a normalização/padronização no dataset de treinamento baseado nos dados de treinamento

        # Converter as espécies para valores numéricos 0, 1 e 2
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(self.df['Species'])

        # Separação do dataset em amostras para treino e teste, considerando 30% dos valores para teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Cria e treina o modelo de machine learning
        mlp = MLPClassifier(hidden_layer_sizes=(10), max_iter=5000, epsilon=1e-10, activation='tanh', learning_rate='constant', solver='adam') # rede neural MLPClassifier
        mlp.fit(X_train, y_train)

        svc = SVC(C=1.0, gamma='scale', kernel='rbf') # machine learning Support Vector Classifier
        svc.fit(X_train, y_train)

        print('TRAINING:\n')

        # Validação cruzada com 10 divisões
        scores_mlp = cross_val_score(mlp, X, y, cv=10)  # cv define o número de "folds" na validação cruzada
        print("Scores da validação cruzada para cada fold do MLPClassifier:\n", scores_mlp)
        print("Acurácia média da validação cruzada do MLPClassifier:", np.mean(scores_mlp), '\n')

        scores_svc = cross_val_score(svc, X, y, cv=10)
        print("Scores da validação cruzada para cada fold do SVC:\n", scores_svc)
        print("Acurácia média da validação cruzada do SVC:", np.mean(scores_svc))

        return mlp, svc, X_test, y_test


    def Teste(self, mlp, svc, X_test, y_test):
        """
        Avalia o desempenho do modelo treinado nos dados de teste.

        Esta função deve ser implementada para testar o modelo e calcular métricas de avaliação relevantes, 
        como acurácia, precisão, ou outras métricas apropriadas ao tipo de problema.
        """

        def classification_metrics(model):

            y_pred = model.predict(X_test)

            # Nome do modelo
            if model == mlp:
                name = 'MLPClassifier'
            elif model == svc:
                name = 'Support Vector Classifier'

            print('\n', 80*'=', '\n')

            print(f"********** {name} **********")
            print('')
            print("CLASSIFICATION METRICS:")

            # Métricas de acurácia
            print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")
            print(f"Balanced accuracy score: {balanced_accuracy_score(y_test, y_pred)}")
            print('')

            # Métricas de precisão, recall, F1-score
            print(f"Precision (macro): {precision_score(y_test, y_pred, average='macro')}")
            print(f"Recall (macro): {recall_score(y_test, y_pred, average='macro')}")
            print(f"F1 Score (macro): {f1_score(y_test, y_pred, average='macro')}")
            print('')

            # Matriz de confusão
            print(f"Confusion matrix\n {confusion_matrix(y_test, y_pred)}")
            print('')

            relatorio = classification_report(y_test, y_pred, target_names=["Iris Setosa", "Iris Virginica", "Iris Versicolor"])
            print("Relatório de classificação das amostras de teste:\n")
            print(relatorio)

            conf_matrix = confusion_matrix(y_test, y_pred)
            cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = ['Iris Setosa', 'Iris Virginica', 'Iris Versicolor'])
            cm_display.plot()
            plt.title(f"Confusion Matrix — {name}", fontsize=14)
            plt.show()
            plt.close()

            if model == mlp:
                # Plotando o gráfico de erros no processo de treinamento
                plt.plot(model.loss_curve_)
                plt.title(f"Training Loss Curve — {name}", fontsize=14)
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.show()
                plt.close()

        classification_metrics(mlp)
        classification_metrics(svc)


    def Train(self):
        """
        Função principal para o fluxo de treinamento do modelo.

        Este método encapsula as etapas de carregamento de dados, pré-processamento e treinamento do modelo.
        Sua tarefa é garantir que os métodos `CarregarDataset`, `TratamentoDeDados` e `Treinamento` estejam implementados corretamente.
        
        Notas:
            * O dataset padrão é "iris.data", mas o caminho pode ser ajustado.
            * Caso esteja executando fora do Colab e enfrente problemas com o path, use a biblioteca `os` para gerenciar caminhos de arquivos.
        """
        self.CarregarDataset('iris.data')

        self.TratamentoDeDados()

        mlp, svc, X_test, y_test = self.Treinamento()

        self.Teste(mlp, svc, X_test, y_test)


modelo = Modelo()
modelo.Train()