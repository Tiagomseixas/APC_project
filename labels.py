import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# --- CONFIGURAÇÕES ---
TICKER = "AAPL"
START_DATE = "2018-01-01"
END_DATE = "2023-01-01"
BARRIER_WIDTH = 2.0        # Multiplicador de Volatilidade (2 desvios padrão)
TIME_HORIZON = 10          # Dias máximos (Barreira Vertical)

# --- 1. PREPARAÇÃO DE DADOS ---
print(f"1. A baixar dados de {TICKER}...")
df = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)

# CORREÇÃO PARA YFINANCE RECENTE:
# Se as colunas tiverem múltiplos níveis (ex: Ticker e Price), achatamos tudo.
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Garantir que temos a coluna Close limpa
close = df['Close']
print(f"Dados baixados: {len(close)} linhas.")

# --- 2. CÁLCULO DA VOLATILIDADE ---
# Volatilidade móvel de 20 dias
daily_vol = close.pct_change().rolling(window=20).std()

# --- 3. ALGORITMO TRIPLE BARRIER METHOD ---
print("2. A aplicar Triple Barrier Method (Simulando trades... aguarda)...")

labels = []
entry_dates = []

# Loop (excluímos o final onde não há horizonte futuro suficiente)
limit_index = len(close) - TIME_HORIZON

for i in range(limit_index):
    
    # Preço e Volatilidade no dia da entrada
    current_price = close.iloc[i]
    current_vol = daily_vol.iloc[i]
    current_date = close.index[i]
    
    # Se volatilidade for NaN (início dos dados), saltamos
    if np.isnan(current_vol):
        continue
        
    # DEFINIR BARREIRAS (Túnel)
    # Topo e Fundo baseados na volatilidade do dia
    top_barrier = current_price * (1 + current_vol * BARRIER_WIDTH)
    bottom_barrier = current_price * (1 - current_vol * BARRIER_WIDTH)
    
    # OLHAR PARA O FUTURO
    future_prices = close.iloc[i+1 : i+1+TIME_HORIZON]
    
    label = 0 # Default: Neutro (Tempo Esgotado)
    
    # Verificar quem foi tocado primeiro
    for future_date, price in future_prices.items():
        if price >= top_barrier:
            label = 1  # Lucro (Compra)
            break
        elif price <= bottom_barrier:
            label = -1 # Perda (Venda/Short)
            break
            
    labels.append(label)
    entry_dates.append(current_date)

# --- 4. CONSTRUÇÃO DO DATAFRAME FINAL ---
# Criamos um novo DF limpo com os resultados
labeled_data = pd.DataFrame({'label': labels}, index=entry_dates)

# Adicionamos o preço original a este novo DF para podermos desenhar
# Usamos .loc para garantir que pegamos nas datas certas
labeled_data['Close'] = close.loc[labeled_data.index]

# --- 5. VISUALIZAÇÃO E ESTATÍSTICAS ---
print("3. A gerar gráfico de Sinais...")
print(f"Colunas disponíveis: {labeled_data.columns}") # Debug para garantir

plt.figure(figsize=(14, 7))
plt.plot(labeled_data.index, labeled_data['Close'], color='gray', alpha=0.5, label='Preço AAPL')

# Filtrar para desenhar as setas
buys = labeled_data[labeled_data['label'] == 1]
sells = labeled_data[labeled_data['label'] == -1]

plt.scatter(buys.index, buys['Close'], color='green', marker='^', s=30, label='Label 1 (Upside)')
plt.scatter(sells.index, sells['Close'], color='red', marker='v', s=30, label='Label -1 (Downside)')

plt.title(f'Triple Barrier Method (Volatilidade Dinâmica)\nHorizonte: {TIME_HORIZON} dias | Largura: {BARRIER_WIDTH}x Vol')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ESTATÍSTICAS FINAIS
counts = labeled_data['label'].value_counts()
print("\n--- DISTRIBUIÇÃO DAS CLASSES ---")
print(f"Neutros (0 - Tempo Esgotado): {counts.get(0, 0)}")
print(f"Compras (1 - Tocou Topo):     {counts.get(1, 0)}")
print(f"Vendas (-1 - Tocou Fundo):    {counts.get(-1, 0)}")
print("--------------------------------")
