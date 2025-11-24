import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
import datetime

# --- FUNÇÕES MATEMÁTICAS ---
def get_weights_ffd(d, thres, lim):
    w, k = [1.], 1
    while True:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres: break
        w.append(w_k)
        k += 1
        if k >= lim: break
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series, d=0.4, thres=1e-5):
    w = get_weights_ffd(d, thres, len(series))
    width = len(w) - 1
    output = []
    series_vals = series.values
    for i in range(width, len(series_vals)):
        window0 = series_vals[(i - width):(i + 1)]
        output.append(np.dot(window0.T, w)[0])
    return pd.Series(output, index=series.index[width:], name='frac_diff')

def get_triple_barrier_labels(price, vol, horizon=10, barrier_width=2.0):
    labels = []
    indices = []
    limit = len(price) - horizon
    for i in range(limit):
        p = price.iloc[i]; v = vol.iloc[i]; t = price.index[i]
        if np.isnan(v): continue
        top = p * (1 + v * barrier_width)
        bot = p * (1 - v * barrier_width)
        future = price.iloc[i+1 : i+1+horizon]
        lbl = 0
        for _, fut_p in future.items():
            if fut_p >= top: lbl = 1; break
            elif fut_p <= bot: lbl = -1; break
        labels.append(lbl)
        indices.append(t)
    return pd.Series(labels, index=indices, name='target')

# --- INTERFACE ---
print("\n" + "="*40)
print("   ORÁCULO DE MERCADO (PREVISÃO LIVE)   ")
print("="*40)

ticker = input("Qual o Ticker para analisar agora? (ex: BTC-USD, AAPL): ").upper()
if ticker == "": ticker = "BTC-USD"

print(f"\n>>> A analisar o futuro imediato de {ticker}...")

# 1. BAIXAR DADOS ATÉ AO ÚLTIMO SEGUNDO
df = yf.download(ticker, period="max", progress=False)

if isinstance(df.columns, pd.MultiIndex): 
    df.columns = df.columns.get_level_values(0)

close = df['Close']
last_date = str(close.index[-1])[:10]
last_price = close.iloc[-1]

print(f"Dados mais recentes: {last_date} | Preço: ${last_price:.2f}")

# 2. PREPARAR FEATURES
print("-> A calcular indicadores...")
feat_frac = frac_diff_ffd(close, d=0.4)
delta = close.diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
feat_rsi = 100 - (100 / (1 + gain/loss))
feat_vol = close.pct_change().rolling(20).std()
sma_50 = close.rolling(50).mean()
feat_dist = (close - sma_50) / sma_50
# Targets (só servem para treino, o último dia não terá target conhecido)
targets = get_triple_barrier_labels(close, feat_vol)

data = pd.concat([feat_frac, feat_rsi, feat_vol, feat_dist, targets], axis=1)
data.columns = ['frac_diff', 'rsi', 'vol', 'dist_sma', 'target']

# 3. PREPARAR TREINO E PREVISÃO
# Removemos NaNs
dataset = data.dropna()

# O "Hoje" (última linha) não tem Target (porque não sabemos o futuro 10 dias à frente)
# Então usamos TUDO menos o último para TREINAR
X_train = dataset.drop(columns=['target']).iloc[:-1] # Tudo menos a última linha
y_train = dataset['target'].iloc[:-1]                # Tudo menos a última linha

# O "Amanhã" é baseado na última linha de dados que temos (Features de Hoje)
X_today = dataset.drop(columns=['target']).iloc[-1:] # Apenas a última linha

# Codificar
y_train_encoded = y_train.map({-1:0, 0:1, 1:2})

# 4. TREINAR MODELO (Em todo o histórico disponível)
print("-> A treinar o modelo com toda a história...")
model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, class_weight='balanced')
model.fit(X_train, y_train_encoded)

# 5. PREVER
prediction_encoded = model.predict(X_today)[0]
probability = model.predict_proba(X_today)[0] # Probabilidades [Venda, Neutro, Compra]

mapping = {0: "VENDA (-1)", 1: "NEUTRO (0)", 2: "COMPRA (1)"}
sinal = mapping[prediction_encoded]
confianca = probability[prediction_encoded] * 100

print("\n" + "="*40)
print(f"   PREVISÃO PARA {ticker}")
print("="*40)
print(f"Sinal do Modelo:   {sinal}")
print(f"Confiança:         {confianca:.2f}%")
print("-" * 40)
print("Probabilidades Detalhadas:")
print(f"   Venda (Bearish): {probability[0]*100:.1f}%")
print(f"   Neutro (Wait):   {probability[1]*100:.1f}%")
print(f"   Compra (Bullish):{probability[2]*100:.1f}%")
print("="*40)

if prediction_encoded == 2 and probability[2] > 0.6:
    print("🔥 ALERTA: Sinal de Compra Forte!")
elif prediction_encoded == 0 and probability[0] > 0.6:
    print("⚠️ ALERTA: Sinal de Venda Forte!")
else:
    print("💤 Conclusão: O mercado está incerto ou o sinal é fraco.")
