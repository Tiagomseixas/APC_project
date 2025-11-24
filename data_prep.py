import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# --- 1. FUNÇÕES MATEMÁTICAS AVANÇADAS (FracDiff) ---
# Nota: Esta é a implementação "Fixed Window" para preservar dados

def get_weights_ffd(d, thres, lim):
    """
    Calcula os pesos para a diferenciação fracionária.
    d: Ordem da diferenciação (ex: 0.4)
    thres: Limite para cortar pesos insignificantes (ex: 1e-5)
    lim: Tamanho máximo da janela
    """
    w, k = [1.], 1
    while True:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
        if k >= lim:
            break
    w = np.array(w[::-1]).reshape(-1, 1)
    return w

def frac_diff_ffd(series, d, thres=1e-5):
    """
    Aplica a diferenciação fracionária numa série temporal (Close price).
    Esta técnica torna os dados estacionários mantendo a memória da tendência.
    """
    # 1. Calcular pesos
    w = get_weights_ffd(d, thres, len(series))
    width = len(w) - 1
    
    # 2. Aplicar os pesos aos dados (dot product)
    df = {}
    name = series.name
    series_vals = series.values  # Converter para numpy array para velocidade
    
    # Loop para aplicar a janela deslizante
    # Nota: Em produção, isto pode ser otimizado, mas aqui é explícito para clareza
    output = []
    # Começamos apenas onde temos dados suficientes (width)
    for i in range(width, len(series_vals)):
        # Janela de dados: do ponto i-width até i
        window0 = series_vals[(i - width):(i + 1)]
        # Produto escalar (dot product) entre os dados e os pesos
        dot_prod = np.dot(window0.T, w)[0]
        output.append(dot_prod)
        
    return pd.Series(output, index=series.index[width:], name=name)

# --- 2. EXECUÇÃO DO PROJETO ---

print("1. A baixar dados da AAPL...")
df = yf.download("AAPL", start="2015-01-01", end="2024-01-01", progress=False)
close_prices = df['Close']

# Se o yfinance devolver MultiIndex (comum em versões novas), resolvemos assim:
if isinstance(close_prices, pd.DataFrame):
    close_prices = close_prices.iloc[:, 0] # Pega na primeira coluna

print(f"Dados baixados: {len(close_prices)} linhas.")

# 3. Aplicar Transformações
print("2. A calcular Diferenciação Fracionária (isto pode demorar uns segundos)...")

# d=0.4 é um valor comum que geralmente passa no teste de estacionaridade (ADF)
# mantendo muita memória.
frac_diff_series = frac_diff_ffd(close_prices, d=0.4)

# Vamos criar também os retornos simples para comparar
simple_returns = close_prices.pct_change().dropna()

# --- 4. VISUALIZAÇÃO ---
print("3. A gerar gráficos comparativos...")

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Gráfico 1: Preço Original (Não Estacionário - Tem Tendência)
axes[0].plot(close_prices, color='blue')
axes[0].set_title('Preço Original (AAPL) - Memória Total, Não Estacionário')
axes[0].grid(True)

# Gráfico 2: Retornos Simples (Estacionário - Sem Memória)
axes[1].plot(simple_returns, color='green', alpha=0.7)
axes[1].set_title('Retornos Simples (d=1.0) - Sem Memória (Ruído Branco)')
axes[1].grid(True)

# Gráfico 3: Diferenciação Fracionária (O Melhor dos Dois Mundos)
axes[2].plot(frac_diff_series, color='red')
axes[2].set_title('FracDiff (d=0.4) - Estacionário mas COM Memória de Tendência')
axes[2].grid(True)
axes[2].axhline(0, color='black', linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.show()

print("Concluído! Analisa o terceiro gráfico.")
