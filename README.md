#  Detecção de Fraude em Cartão de Crédito

Projeto de classificação em Machine Learning para identificar transações fraudulentas em um dataset real e **extremamente desbalanceado** (só 0,17% das transações são fraude). O foco não é só treinar modelos, mas mostrar por que **acurácia é uma métrica enganosa** nesse tipo de problema, e o trade-off entre Precision e Recall que qualquer projeto de fraude precisa enfrentar.

##  Sobre o dataset

Transações feitas por portadores de cartão europeus em setembro de 2013.

| | |
|---|---|
| Total de transações | 284.807 |
| Fraudes | 492 (0,173%) |
| Features | 28 componentes PCA (`V1`-`V28`) + `Time` + `Amount` |

O dataset não vem incluído neste repositório (arquivo de ~98MB). Baixe por uma das opções:

- **Kaggle** (precisa de conta gratuita): [creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Mirror direto no GitHub** (sem login): [creditcard.csv](https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv)

Depois de baixar, coloque o arquivo `creditcard.csv` na raiz do projeto.

##  Metodologia

1. **EDA**: análise da distribuição das classes e do valor das transações
2. **Pré-processamento**: escalonamento de `Amount` e `Time` (as colunas `V1`-`V28` já vêm normalizadas via PCA)
3. **Modelagem**: 4 algoritmos clássicos de classificação, todos com `class_weight='balanced'` (exceto KNN, que não suporta esse parâmetro):
   - Regressão Logística
   - Árvore de Decisão
   - K-Nearest Neighbors (KNN)
   - SVM Linear
4. **Avaliação**: Precision, Recall, F1-Score e ROC-AUC (nunca acurácia sozinha)

##  Resultados

| Modelo | Precision | Recall | F1-Score | ROC-AUC | Tempo |
|---|---|---|---|---|---|
| **KNN** | 0.919 | 0.806 | **0.859** | 0.944 | ~20-120s |
| SVM (Linear) | 0.072 | 0.918 | 0.133 | **0.975** | ~2-4s |
| Árvore de Decisão | 0.069 | 0.827 | 0.128 | 0.908 | ~6-9s |
| Regressão Logística | 0.061 | 0.918 | 0.114 | 0.972 | ~2-3s |

>  Adicione aqui os gráficos gerados pelo script:
> `01_distribuicao_classes.png` · `02_distribuicao_amount.png` · `03_comparacao_modelos.png` · `04_matrizes_confusao.png` · `05_curva_roc.png`

##  Principais insights

- Com 99,8% das transações sendo normais, um modelo que nunca prevê fraude já "acerta" quase tudo, por isso acurácia não serve como métrica aqui.
- **Regressão Logística e SVM** com `class_weight='balanced'` pegam ~92% das fraudes (recall alto), mas geram muitíssimos falsos positivos (precision de só ~6-7%), na prática, isso significa bloquear um monte de compra legítima.
- **KNN**, mesmo sem nenhum ajuste de balanceamento, equilibra melhor Precision e Recall neste dataset, mas é o modelo mais lento e menos escalável dos quatro.
- A escolha do "melhor" modelo depende do que custa mais caro pro negócio: deixar passar uma fraude ou incomodar um cliente legítimo.

##  Como rodar

```bash
git clone https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
cd SEU-REPOSITORIO

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

# baixe o creditcard.csv (veja seção "Sobre o dataset") e coloque na raiz

python3 predicao.py
```

O script também tem marcadores `# %%`, então pode ser rodado célula por célula no VS Code ou Jupyter para ver os gráficos inline.

##  Tecnologias

- Python 3
- pandas / numpy
- scikit-learn
- matplotlib / seaborn

##  Licença

Este projeto é de uso livre para fins de estudo. O dataset original pertence à [ULB Machine Learning Group](https://www.kaggle.com/mlg-ulb/creditcardfraud).