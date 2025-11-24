import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- 1. PREPARAR DADOS (Igual ao anterior) ---
print("1. A preparar dados...")
df = pd.read_csv("dataset_final.csv", index_col=0)
X = df.drop(columns=['target'])
y = df['target']

# Codificar target: -1, 0, 1 -> 0, 1, 2
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Dividir Treino/Teste (80/20 cronológico)
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y_encoded[:split], y_encoded[split:]

# --- 2. TREINAR MODELO ---
print("2. A treinar modelo...")
# Vamos usar os mesmos parâmetros que deram bom resultado
model = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=3, 
    random_state=42
)
model.fit(X_train, y_train)

# --- 3. ANÁLISE DE CALIBRAÇÃO (Foco na Classe COMPRA) ---
print("3. A calcular curva de calibração...")

# Probabilidades de cada classe [P(Venda), P(Neutro), P(Compra)]
probs = model.predict_proba(X_test)

# Queremos analisar apenas a probabilidade de COMPRA (Classe 2 no encoded, ou 1.0 no original)
# O índice da classe Compra depende do LabelEncoder. 
# Normalmente: -1->0, 0->1, 1->2. Logo Compra é índice 2.
prob_buy = probs[:, 2]

# Transformar y_test num binário: 1 se for COMPRA, 0 se não for
y_test_binary = (y_test == 2).astype(int)

# Calcular a curva (Bina as probabilidades em 10 grupos e vê a precisão real de cada grupo)
fraction_of_positives, mean_predicted_value = calibration_curve(y_test_binary, prob_buy, n_bins=10)

# --- 4. VISUALIZAÇÃO ---
plt.figure(figsize=(10, 10))

# Plot 1: A Curva de Calibração
plt.subplot(2, 1, 1)
plt.plot([0, 1], [0, 1], "k:", label="Perfeitamente Calibrado")
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="O Teu Modelo (XGBoost)", color='blue')

plt.ylabel("Fração Real de Positivos (Precisão)")
plt.xlabel("Confiança Média do Modelo")
plt.title("Reliability Diagram (Diagrama de Confiabilidade) - Classe COMPRA")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# Plot 2: Histograma de Confiança (Quantas vezes ele é arrojado?)
plt.subplot(2, 1, 2)
plt.hist(prob_buy, range=(0, 1), bins=10, histtype="step", lw=2, color='blue')
plt.xlabel("Confiança Média do Modelo")
plt.ylabel("Contagem de Amostras")
plt.title("Histograma de Confiança (Quantas vezes ele aposta?)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n--- ANÁLISE RÁPIDA ---")
print("Olha para o gráfico de cima:")
print("1. Se a linha azul estiver ABAIXO da pontilhada -> O modelo é 'Excesso de Confiança' (Arrogante).")
print("2. Se a linha azul estiver ACIMA da pontilhada -> O modelo é 'Sub-confiante' (tímido).")
print("3. Se a linha azul seguir a pontilhada -> Perfeito.")
