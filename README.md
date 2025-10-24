# Comparação de Arquiteturas CNN para Classificação CIFAR-10

Este projeto treina e compara duas arquiteturas de Redes Neurais Convolucionais (CNN) para classificar imagens do dataset CIFAR-10.

O notebook principal (`CnnModel.ipynb`) demonstra todo o processo, desde o carregamento dos dados até a avaliação final e a análise comparativa dos resultados.

## Dataset

O [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) é um dataset de 60.000 imagens coloridas de 32x32 pixels, divididas em 10 classes:
* Avião
* Automóvel
* Pássaro
* Gato
* Cervo
* Cachorro
* Sapo
* Cavalo
* Navio
* Caminhão

Os dados foram pré-processados da seguinte forma:
* Normalização dos pixels para a escala [0, 1].
* Codificação dos rótulos em formato one-hot.
* Separação de 10% dos dados de treino para um conjunto de validação (5.000 imagens).

## Arquiteturas Testadas

Foram testadas duas arquiteturas, ambas treinadas por 15 épocas com tamanho de lote (batch size) de 32:

### 1. Modelo A (Baseline)
Uma CNN simples com 3 camadas convolucionais e 1 camada densa, para estabelecer uma linha de base de desempenho.
* `Conv2D(32) -> MaxPooling`
* `Conv2D(64) -> MaxPooling`
* `Conv2D(64)`
* `Flatten -> Dense(64) -> Dense(10, activation='softmax')`
* **Total de Parâmetros:** 122.570

### 2. Modelo B (Melhorado)
Uma arquitetura mais profunda que utiliza **Batch Normalization** (para estabilizar o treinamento e acelerar a convergência) e **Dropout** (para reduzir o overfitting).
* `Conv2D(32) -> BatchNormalization -> MaxPooling -> Dropout(0.25)`
* `Conv2D(64) -> BatchNormalization -> MaxPooling -> Dropout(0.25)`
* `Conv2D(128) -> BatchNormalization -> MaxPooling -> Dropout(0.25)`
* `Flatten -> Dense(512) -> BatchNormalization -> Dropout(0.5) -> Dense(10, activation='softmax')`
* **Total de Parâmetros:** 1.150.410 (Treináveis: 1.148.938)

## Resultados e Análise

Os modelos foram compilados com o otimizador 'adam' e a função de perda 'categorical_crossentropy', monitorando a acurácia.

### Desempenho no Conjunto de Teste
Após 15 épocas de treinamento, os modelos foram avaliados no conjunto de teste:

| Modelo             | Acurácia no Teste | Perda no Teste |
| :----------------- | :---------------: | :------------: |
| Modelo A (Baseline) | **70.20%** | 1.0168         |
| Modelo B (Melhorado)| 68.72%        | 0.9596         |

### Análise das Curvas de Treinamento
Os gráficos de acurácia e perda na validação ao longo das épocas mostram o seguinte:

* **(Acurácia):** O Modelo A atingiu uma acurácia de validação ligeiramente superior e mais estável ao final das 15 épocas em comparação com o Modelo B, que apresentou mais oscilações.
* **(Perda):** A perda de validação do Modelo A tendeu a aumentar nas épocas finais, sugerindo um leve overfitting. O Modelo B, apesar de oscilar, mostrou momentos de perda menor que o Modelo A, indicando potencial com mais treinamento ou ajuste de hiperparâmetros (como a taxa de dropout ou o otimizador).

*(**Sugestão:** Insira aqui uma captura de tela (screenshot) dos seus gráficos da Célula 6 do notebook)*

![Comparação Acurácia e Perda](link_para_imagem_graficos.png)

### Análise das Matrizes de Confusão
As matrizes de confusão mostram como cada modelo classifica as imagens do conjunto de teste:

* **Modelo A (Baseline):** Apresentou bom desempenho geral, com a diagonal principal concentrando a maioria das previsões corretas. Confusões notáveis ocorreram entre classes visualmente similares (ex: Gato vs Cachorro, Caminhão vs Automóvel).
* **Modelo B (Melhorado):** Apesar da acurácia final ligeiramente menor no teste, a matriz de confusão pode revelar se ele teve melhor desempenho em classes específicas ou se as confusões foram distribuídas de forma diferente. No seu caso, o desempenho foi inferior ao Modelo A. As confusões mais significativas ocorreram entre Gato/Cachorro e Pássaro/Cervo/Sapo.

*(**Sugestão:** Insira aqui uma captura de tela (screenshot) dos seus heatmaps da Célula 8 do notebook)*

![Comparação Matrizes de Confusão](link_para_imagem_matrizes.png)

### Conclusão da Comparação
Neste experimento específico com 15 épocas, o **Modelo A (Baseline)** apresentou uma acurácia ligeiramente superior no conjunto de teste (70.20%) em comparação ao Modelo B (68.72%). Embora o Modelo B utilize técnicas de regularização como Batch Normalization e Dropout, ele pode necessitar de mais épocas para convergir melhor ou de um ajuste fino nos hiperparâmetros para superar o modelo mais simples. O overfitting observado no Modelo A nas últimas épocas também sugere que técnicas como *early stopping* poderiam ser benéficas.

## Parâmetros Medidos e Analisados

* **Acurácia (Accuracy):** Percentual de classificações corretas (monitorada no treino, validação e teste).
* **Perda (Loss):** Valor da função de perda ('categorical_crossentropy'), indicando quão "erradas" estão as previsões (monitorada no treino, validação e teste).
* **Curvas de Treinamento/Validação:** Gráficos da acurácia e perda por época, usados para visualizar a convergência e detectar overfitting/underfitting.
* **Matriz de Confusão:** Tabela que detalha os acertos e erros por classe no conjunto de teste, permitindo identificar quais classes são mais confundidas.

## Como Executar

1.  Clone este repositório.
2.  (Opcional, mas recomendado) Crie e ative um ambiente virtual:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    .\venv\Scripts\Activate.ps1 # Windows PowerShell
    ```
3.  Instale as dependências (certifique-se de ter o `requirements.txt`):
    ```bash
    pip install -r requirements.txt
    ```
4.  Inicie o Jupyter Lab:
    ```bash
    jupyter-lab
    ```
5.  Abra e execute o notebook `CnnModel.ipynb`. O dataset CIFAR-10 será baixado automaticamente na primeira execução, se necessário.
