#  Detecção de Fraude em Cartão de Crédito

Projeto de classificação em Machine Learning para identificar transações fraudulentas em um dataset real e **extremamente desbalanceado** (só 0,17% das transações são fraude). O foco aqui não é apenas treinar modelos, e sim mostrar por que **acurácia é uma métrica enganosa** nesse tipo de problema, além do trade-off entre Precision e Recall que todo projeto de fraude precisa encarar.

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
3. **Modelagem**: 5 algoritmos de classificação, todos com `class_weight='balanced'` (exceto KNN, que não suporta esse parâmetro):
   - Regressão Logística
   - Árvore de Decisão
   - K-Nearest Neighbors (KNN)
   - SVM Linear
   - Gradient Boosting (`HistGradientBoostingClassifier`), adicionado na iteração 2
4. **Avaliação**: Precision, Recall, F1-Score, F2-Score e ROC-AUC (nunca acurácia sozinha)

##  Resultados

| Modelo | Precision | Recall | F1-Score | F2-Score | ROC-AUC | Tempo |
|---|---|---|---|---|---|---|
| **KNN** | 0.919 | 0.806 | **0.859** | 0.826 | 0.944 | ~20-120s |
| **Gradient Boosting** 🆕 | 0.330 | 0.888 | 0.481 | **0.663** | 0.968 | ~2-3s |
| SVM (Linear) | 0.072 | 0.918 | 0.133 | 0.273 | **0.975** | ~2-4s |
| Árvore de Decisão | 0.069 | 0.827 | 0.128 | 0.260 | 0.908 | ~6-9s |
| Regressão Logística | 0.061 | 0.918 | 0.114 | 0.241 | 0.972 | ~2-3s |

##  Iteração 2: Gradient Boosting + F2-score

A primeira versão deste projeto comparava os 4 modelos vistos no curso da IBM (Regressão Logística, KNN, Árvore de Decisão e SVM). Depois de mostrar o resultado para um colega, recebi dois feedbacks que me fizeram evoluir o projeto:

1. **"Testa Gradient Boosting"**: um modelo de *ensemble* mais avançado, que treina várias árvores em sequência, cada uma corrigindo os erros da anterior. É conhecido por ter ótimo desempenho em problemas com dados tabulares como este.
2. **"O F1 tem que ser seu objetivo, não a precisão"**: otimizar só por precisão é uma armadilha, já que um modelo que quase nunca prevê fraude tem precisão altíssima e é inútil. Adicionei o **F2-score**, uma variante que dá ainda mais peso ao recall. Essa é a métrica mais alinhada com fraude, onde deixar passar um caso costuma custar mais caro do que um alarme falso.

**O que mudou:** o Gradient Boosting mais que triplicou o F1 dos modelos lineares (de ~0.11-0.13 para 0.48), mantendo recall alto, sem virar um "detector paranoico" que bloqueia todo mundo. Ele roda em cerca de 2 segundos, aproximadamente 100x mais rápido que o KNN, que ainda lidera em F1 puro mas é impraticável em escala real.

##  Principais insights

- Com 99,8% das transações sendo normais, um modelo que nunca prevê fraude já "acerta" quase tudo. É por isso que acurácia não serve como métrica aqui.
- **Regressão Logística e SVM** com `class_weight='balanced'` pegam ~92% das fraudes (recall alto), mas geram muitíssimos falsos positivos (precision de só ~6-7%). Na prática, isso significa bloquear um monte de compra legítima.
- **Gradient Boosting** teve o melhor equilíbrio entre os modelos: recall alto sem sacrificar tanto a precision, e treino rapidíssimo.
- **KNN** ainda lidera em F1 puro, mas é de longe o mais lento (cerca de 100x mais devagar que o Gradient Boosting), o que o torna pouco escalável para uso real.
- A escolha do "melhor" modelo depende do que custa mais caro para o negócio: deixar passar uma fraude ou incomodar um cliente legítimo. Por isso comparo F1 **e** F2.

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

##  Próximos passos

- [x] Testar Gradient Boosting (feito na iteração 2)
- [ ] Testar SMOTE (`imbalanced-learn`) em vez de `class_weight`
- [ ] Testar XGBoost / LightGBM e comparar com o HistGradientBoosting
- [ ] Ajustar hiperparâmetros com `GridSearchCV`
- [ ] Ajustar o threshold de decisão para otimizar o custo de negócio

## 📄 Licença

Este projeto é de uso livre para fins de estudo. O dataset original pertence ao [ULB Machine Learning Group](https://www.kaggle.com/mlg-ulb/creditcardfraud).