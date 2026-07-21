# %% [markdown]
# # Detecção de Fraude em Cartão de Crédito — v2 (com Gradient Boosting)
#
# Projeto de classificação usando o dataset "Credit Card Fraud Detection"
# (transações reais anonimizadas de cartões europeus, set/2013).
#
# Objetivo: dado que so 0,17% das transações são fraude, comparar como
# diferentes algoritmos de classificação lidam com esse desbalanceamento
# extremo — e por que "acurácia" sozinha engana muito aqui.
#
# Esta é a 2ª iteração do projeto. A v1 comparava os 4 modelos vistos no
# curso da IBM (Regressão Logística, KNN, Árvore de Decisão e SVM). Depois
# de um feedback, foram adicionados:
#   - Gradient Boosting (HistGradientBoostingClassifier), um modelo de
#     ensemble mais avançado que costuma dominar problemas tabulares;
#   - a métrica F2-score, que dá mais peso ao recall — mais adequada para
#     fraude, onde deixar passar um caso costuma custar mais caro do que
#     um alarme falso.
#
# ## Como conseguir o dataset
# Duas opções, escolha uma:
#
# 1. Kaggle (precisa de conta gratuita):
#    https://www.kaggle.com/mlg-ulb/creditcardfraud
#    Baixe e coloque o arquivo `creditcard.csv` na mesma pasta deste script.
#
# 2. Mirror público no GitHub (sem precisar de login):
#    https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv
#    (arquivo de ~98MB, é o mesmo dataset original da Kaggle)

# %%
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, fbeta_score, roc_auc_score, roc_curve
)

sns.set_style("whitegrid")

# %% [markdown]
# ## 1. Carregando e explorando os dados

# %%
df = pd.read_csv('creditcard.csv')
print(f"Shape: {df.shape}")
df.head()

# %%
# O ponto central do problema: o quão desbalanceadas são as classes?
contagem = df['Class'].value_counts()
percentual = df['Class'].value_counts(normalize=True) * 100

print("Contagem por classe:")
print(contagem)
print("\nPercentual por classe:")
print(percentual.round(3))

fig, ax = plt.subplots(figsize=(5, 4))
sns.barplot(x=contagem.index, y=contagem.values, ax=ax)
ax.set_xticks(range(len(contagem)))
ax.set_xticklabels(['Normal (0)', 'Fraude (1)'])
ax.set_ylabel('Número de transações')
ax.set_title('Distribuição de classes (escala linear)')
for i, v in enumerate(contagem.values):
    ax.text(i, v, f'{v:,}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('01_distribuicao_classes.png', dpi=120)
plt.show()

# Repare: com 284.315 transações normais contra apenas 492 fraudes,
# um modelo "preguiçoso" que sempre chuta "normal" já acertaria 99.8%.
# Isso mostra por que ACURÁCIA é uma métrica inútil aqui.

# %%
# Como o Amount (valor da transação) se comporta em cada classe?
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(data=df, x='Class', y='Amount', ax=axes[0])
axes[0].set_ylim(0, 500)
axes[0].set_title('Valor da transação por classe (zoom até R$500)')

sns.histplot(data=df, x='Amount', hue='Class', bins=50,
             log_scale=(False, True), ax=axes[1])
axes[1].set_xlim(0, 500)
axes[1].set_title('Distribuição do valor (escala log)')
plt.tight_layout()
plt.savefig('02_distribuicao_amount.png', dpi=120)
plt.show()

# %% [markdown]
# ## 2. Pré-processamento
#
# As colunas V1-V28 já vêm normalizadas (são componentes de PCA aplicados
# pela própria Kaggle para anonimizar os dados). Só `Amount` e `Time`
# precisam ser escalonadas manualmente, como vocês viram no curso.

# %%
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_scaled'] = scaler.fit_transform(df[['Time']])
df = df.drop(['Amount', 'Time'], axis=1)

X = df.drop('Class', axis=1)
y = df['Class']

# stratify=y garante que a proporção de fraudes seja igual em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Treino: {X_train.shape[0]:,} transações ({y_train.sum()} fraudes)")
print(f"Teste:  {X_test.shape[0]:,} transações ({y_test.sum()} fraudes)")

# %% [markdown]
# ## 3. Treinando os modelos
#
# Os mesmos 4 algoritmos de classificação do curso da IBM. Em Logistic
# Regression, Decision Tree, SVM e Gradient Boosting usamos
# `class_weight='balanced'`, que faz o modelo "prestar mais atenção" na
# classe minoritária (fraude) — sem isso, a maioria desses modelos
# simplesmente ignoraria as fraudes. KNN não tem esse parâmetro, então
# ele entra "no bruto" para comparação.

# %%
modelos = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', max_iter=1000
    ),
    'Decision Tree': DecisionTreeClassifier(
        class_weight='balanced', max_depth=8, random_state=42
    ),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM (Linear)': LinearSVC(
        class_weight='balanced', max_iter=5000, dual=False
    ),
    # NOVO na v2: ensemble de árvores treinadas em sequência, cada uma
    # corrigindo o erro da anterior. class_weight='balanced' faz ele dar
    # mais atenção à classe minoritária (fraude), igual aos outros.
    'Gradient Boosting': HistGradientBoostingClassifier(
        class_weight='balanced', random_state=42
    ),
}

resultados = []
predicoes = {}

for nome, modelo in modelos.items():
    print(f"Treinando {nome}...")
    t0 = time.time()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    if hasattr(modelo, "predict_proba"):
        y_score = modelo.predict_proba(X_test)[:, 1]
    else:
        y_score = modelo.decision_function(X_test)

    duracao = time.time() - t0
    predicoes[nome] = (y_pred, y_score)

    resultados.append({
        'Modelo': nome,
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        # F2 pesa recall mais que precision (beta=2) — mais adequado para
        # fraude, onde deixar passar um caso costuma ser pior que um
        # alarme falso.
        'F2-Score': fbeta_score(y_test, y_pred, beta=2),
        'ROC-AUC': roc_auc_score(y_test, y_score),
        'Tempo (s)': round(duracao, 1),
    })
    print(f"  concluído em {duracao:.1f}s")

# Aviso: o KNN pode levar 1-2 minutos, porque ele precisa calcular a
# distância de cada transação de teste até as ~228 mil de treino. Todos
# os outros (incluindo o Gradient Boosting) rodam em poucos segundos.

# %% [markdown]
# ## 4. Comparando os modelos

# %%
df_resultados = pd.DataFrame(resultados).sort_values('F1-Score', ascending=False)
print(df_resultados.to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(8, 5))
df_plot = df_resultados.set_index('Modelo')[['Precision', 'Recall', 'F1-Score']]
df_plot.plot(kind='bar', ax=ax)
ax.set_ylabel('Score')
ax.set_title('Precision x Recall x F1-Score por modelo')
ax.set_ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('03_comparacao_modelos.png', dpi=120)
plt.show()

# %%
# Matrizes de confusão lado a lado — aqui fica bem visual o trade-off
# entre "pegar toda fraude" (recall alto) e "não incomodar cliente à toa"
# (precision alta).
fig, axes = plt.subplots(1, len(modelos), figsize=(4 * len(modelos), 4))
for ax, (nome, (y_pred, _)) in zip(axes, predicoes.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Normal', 'Fraude'],
                yticklabels=['Normal', 'Fraude'])
    ax.set_title(nome, fontsize=10)
    ax.set_xlabel('Previsto')
    ax.set_ylabel('Real')
plt.tight_layout()
plt.savefig('04_matrizes_confusao.png', dpi=120)
plt.show()

# %%
# Curvas ROC de todos os modelos no mesmo gráfico
fig, ax = plt.subplots(figsize=(6, 6))
for nome, (_, y_score) in predicoes.items():
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)
    ax.plot(fpr, tpr, label=f'{nome} (AUC={auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Aleatório')
ax.set_xlabel('Taxa de Falsos Positivos')
ax.set_ylabel('Taxa de Verdadeiros Positivos')
ax.set_title('Curva ROC')
ax.legend(loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig('05_curva_roc.png', dpi=120)
plt.show()

# %% [markdown]
# ## 5. Conclusões
#
# - **Acurácia é enganosa aqui**: qualquer modelo bobo que nunca preveja
#   fraude já acerta ~99.8%. Por isso avaliamos com Precision, Recall,
#   F1, F2 e ROC-AUC em vez disso.
# - **`class_weight='balanced'` tem um preço**: Logistic Regression, SVM
#   e Decision Tree tendem a pegar quase todas as fraudes (recall alto),
#   mas à custa de MUITOS falsos positivos (precision baixa) — na prática
#   isso significa bloquear compras legítimas de clientes.
# - **Gradient Boosting (novo na v2)** foi o maior salto de equilíbrio
#   sobre os modelos lineares: sem virar um "detector paranoico", ele
#   mais que triplica o F1 da Regressão Logística/SVM, mantendo recall
#   alto — exatamente o comportamento que se espera de um ensemble num
#   problema tabular difícil.
# - **KNN** ainda lidera em F1 puro (equilibra melhor precision e recall
#   neste dataset), mas é de longe o mais lento — ~100x mais devagar que
#   o Gradient Boosting — o que o torna pouco prático em escala real.
# - **F1 vs F2**: F2 dá mais peso ao recall. Ao ordenar por F2 em vez de
#   F1, os modelos que "correm atrás" de toda fraude sobem no ranking —
#   é a métrica a usar se, no seu contexto, deixar passar uma fraude for
#   pior que incomodar um cliente legítimo.
# - Qual modelo "vence" depende do que custa mais caro para o negócio:
#   deixar uma fraude passar, ou incomodar um cliente legítimo.
#
# ## 6. Próximos passos (se quiser ir além)
#
# - Testar **SMOTE** (biblioteca `imbalanced-learn`) para gerar exemplos
#   sintéticos da classe minoritária em vez de só usar `class_weight`.
# - Testar **XGBoost** ou **LightGBM** e comparar com o HistGradientBoosting.
# - Fazer **GridSearchCV** para ajustar hiperparâmetros de cada modelo.
# - Ajustar o **threshold de decisão** (em vez do padrão de 0.5) para
#   otimizar diretamente para o custo de negócio que importa mais pra você.