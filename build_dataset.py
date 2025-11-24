import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# --- 1. FUNÇÕES MATEMÁTICAS (Feitas à mão para não depender de bibliotecas externas) ---

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
    # Diferenciação Fracionária
    w = get_weights_ffd(d, thres, len(series))
    width = len(w) - 1
    output = []
    series_vals = series.values
    for i in range(width, len(series_vals)):
        window0 = series_vals[(i - width):(i + 1)]
        output.append(np.dot(window0.T, w)[0])
    return pd.Series(output, index=series.index[width:], name='frac_diff')

def get_triple_barrier_labels(price, vol, horizon=10, barrier_width=2.0):
    # Triple Barrier Method
    labels = []
    indices = []
    limit = len(price) - horizon
    
    for i in range(limit):
        p = price.iloc[i]
        v = vol.iloc[i]
        t = price.index[i]
        
        if np.isnan(v): continue
            
        top = p * (1 + v * barrier_width)
        bot = p * (1 - v * barrier_width)
        future = price.iloc[i+1 : i+1+horizon]
        
        lbl = 0
        for _, fut_p in future.items():
            if fut_p >= top:
                lbl = 1; break
            elif fut_p <= bot:
                lbl = -1; break
        
        labels.append(lbl)
        indices.append(t)
        
    return pd.Series(labels, index=indices, name='target')

def calculate_rsi(series, period=14):
    # Cálculo manual do RSI (para não precisar do pandas_ta)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- 2. EXECUÇÃO DO PIPELINE ---

print("1. A baixar dados da AAPL...")
df = yf.download("AAPL", start="2015-01-01", end="2024-01-01", progress=False)

# Correção para yfinance recente
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

close = df['Close']
print(f"Dados: {len(close)} linhas.")

print("2. A criar Features (Inputs)...")

# Feature 1: FracDiff (Memória Estacionária)
# d=0.4 preserva a memória mas torna os dados estacionários
feat_frac = frac_diff_ffd(close, d=0.4)

# Feature 2: RSI (Calculado manualmente)
feat_rsi = calculate_rsi(close, period=14)
feat_rsi.name = 'rsi'

# Feature 3: Volatilidade (Risco)
feat_vol = close.pct_change().rolling(20).std()
feat_vol.name = 'volatility'

# Feature 4: Distância para Média Móvel (Tendência)
sma_50 = close.rolling(50).mean()
feat_dist_sma = (close - sma_50) / sma_50
feat_dist_sma.name = 'dist_sma50'

print("3. A criar Labels (Target)...")
# Target: Triple Barrier (O output real: 1, -1 ou 0)
targets = get_triple_barrier_labels(close, feat_vol, horizon=10, barrier_width=2.0)

print("4. A juntar tudo no Dataset Final...")
# Unir todas as colunas
dataset = pd.concat([feat_frac, feat_rsi, feat_vol, feat_dist_sma, targets], axis=1)

# Remover linhas com valores em falta (NaNs) criados pelos indicadores
dataset.dropna(inplace=True)

print("\n--- DATASET FINAL PRONTO ---")
print(dataset.head())
print(f"\nTamanho Total: {len(dataset)} linhas")
print("Distribuição dos Labels:")
print(dataset['target'].value_counts())

# Guardar em CSV
dataset.to_csv("dataset_final.csv")
print("\nSucesso! Ficheiro 'dataset_final.csv' criado.")
