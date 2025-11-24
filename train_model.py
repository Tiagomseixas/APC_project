import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# --- 1. CARREGAR DADOS ---
print("1. A carregar o dataset...")
df = pd.read_csv("dataset_final.csv", index_col=0)

# Separar Inputs (X) e Alvo (y)
X = df.drop(columns=['target'])
y = df['target']

# XGBoost gosta que as classes sejam 0, 1, 2.
# O nosso target é -1, 0, 1. Vamos converter.
# -1 (Venda) -> 0
#  0 (Neutro) -> 1
#  1 (Compra) -> 2
le = LabelEncoder()
y_encoded = le.fit_transform(y) 

# --- 2. DIVISÃO SEM BATOTA (Time Series Split) ---
# Usamos os primeiros 80% para treinar e os ultimos 20% para testar
split_point = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y_encoded[:split_point], y_encoded[split_point:]

print(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")

# --- 3. TREINAR O MODELO ---
print("2. A treinar o XGBoost (O Cérebro)...")
model = XGBClassifier(
    n_estimators=100,     # Número de árvores
    learning_rate=0.1,    # Velocidade de aprendizagem
    max_depth=3,          # Profundidade das árvores (evita decorar demais)
    random_state=42
)

model.fit(X_train, y_train)

# --- 4. AVALIAÇÃO ---
print("3. A avaliar performance...")
y_pred = model.predict(X_test)

# Converter de volta para os nomes originais (-1, 0, 1) para lermos melhor
target_names = [str(cls) for cls in le.classes_]
print("\n--- RELATÓRIO DE CLASSIFICAÇÃO ---")
print(classification_report(y_test, y_pred, target_names=target_names))

# --- 5. O QUE É QUE O MODELO APRENDEU? (Feature Importance) ---
plt.figure(figsize=(10, 6))
importances = model.feature_importances_
indices = np.argsort(importances)

plt.title('Importância das Features (O que o modelo valoriza?)')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Importância Relativa')
plt.show()

# --- 6. MATRIZ DE CONFUSÃO ---
plt.figure(figsize=(6,5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title("Matriz de Confusão (Real vs Previsto)")
plt.ylabel("Realidade")
plt.xlabel("Previsão do Modelo")
plt.show()
