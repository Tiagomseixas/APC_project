import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import datetime # <--- IMPORTANTE: Para saber o dia de hoje

# --- FUNÇÕES MATEMÁTICAS (O motor do sistema) ---
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

# --- INTERAÇÃO COM O UTILIZADOR ---
print("\n" + "="*40)
print("   SIMULADOR DE TRADING COM IA   ")
print("   (Agora compatível com 2025)   ")
print("="*40)

ticker = input("Qual o Ticker que queres testar? (Enter para AAPL): ").upper()
if ticker == "": ticker = "AAPL"

ano_teste = input("Qual o ano que queres simular? (ex: 2024, 2025): ")

try:
    capital_inicial = float(input("Qual o capital inicial? (Enter para 10000): ") or 10000)
except ValueError:
    capital_inicial = 10000

print(f"\n>>> A carregar a Máquina do Tempo para {ano_teste}...")

# --- 1. PREPARAR DADOS ---
print("1. A baixar histórico completo...")

# DATA DINÂMICA: Calcula o dia de amanhã para garantir que pega os dados de hoje
hoje = datetime.date.today() + datetime.timedelta(days=1)
str_hoje = hoje.strftime("%Y-%m-%d")

# Baixa dados desde 2015 até HOJE
df = yf.download(ticker, start="2015-01-01", end=str_hoje, progress=False)

# Correção para novas versões do yfinance
if isinstance(df.columns, pd.MultiIndex): 
    df.columns = df.columns.get_level_values(0)

if df.empty:
    print("ERRO: Ticker não encontrado ou sem dados.")
    exit()

close = df['Close']

# Criar Features
print("2. A aplicar matemática financeira (FracDiff, Triple Barrier)...")
feat_frac = frac_diff_ffd(close, d=0.4)

# Indicadores Técnicos
delta = close.diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
feat_rsi = 100 - (100 / (1 + gain/loss))
feat_vol = close.pct_change().rolling(20).std()
sma_50 = close.rolling(50).mean()
feat_dist = (close - sma_50) / sma_50
targets = get_triple_barrier_labels(close, feat_vol)

# Dataset
data = pd.concat([feat_frac, feat_rsi, feat_vol, feat_dist, targets], axis=1)
data.columns = ['frac_diff', 'rsi', 'vol', 'dist_sma', 'target']
data.dropna(inplace=True)

# --- 2. DIVIDIR O TEMPO ---
# O treino é tudo ANTES do ano escolhido
train_data = data[data.index < f"{ano_teste}-01-01"]
# O teste é O ano escolhido
test_data = data[data.index.astype(str).str.contains(ano_teste)]

if len(train_data) < 100:
    print(f"ERRO: Não há dados históricos suficientes antes de {ano_teste} para treinar o modelo.")
    exit()

if len(test_data) == 0:
    print(f"ERRO: Não há dados para o ano de {ano_teste} (Verifica se o ano já começou!).")
    exit()

print(f"   -> Treino: {len(train_data)} dias (Passado)")
print(f"   -> Simulação: {len(test_data)} dias (Ano de {ano_teste})")

X_train = train_data.drop(columns=['target'])
y_train = train_data['target']
# Mapear para 0, 1, 2
y_train_encoded = y_train.map({-1:0, 0:1, 1:2})

# --- 3. TREINAR ---
print("3. A treinar a Inteligência Artificial (Random Forest)...")
model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, class_weight='balanced')
model.fit(X_train, y_train_encoded)

# --- 4. SIMULAR ---
print(f"4. A executar trades dia-a-dia em {ano_teste}...")
X_test = test_data.drop(columns=['target'])
preds_encoded = model.predict(X_test)
mapping = {0:-1, 1:0, 2:1} 
signals = [mapping[p] for p in preds_encoded]

# Carteira
capital = capital_inicial
posicao = 0 
historico_capital = []
precos_teste = close.loc[test_data.index]
qtd_acoes = 0
trades_feitos = 0

for i, data_atual in enumerate(test_data.index):
    sinal = signals[i]
    preco = precos_teste.iloc[i]
    
    # Regras de Trading
    if sinal == 1 and posicao == 0: # COMPRA
        qtd_acoes = capital / preco
        capital = 0 
        posicao = 1
        trades_feitos += 1
        
    elif sinal == -1 and posicao == 1: # VENDA
        capital = qtd_acoes * preco
        posicao = 0
        qtd_acoes = 0
        trades_feitos += 1
    
    valor_atual = capital if posicao == 0 else (qtd_acoes * preco)
    historico_capital.append(valor_atual)

# Forçar venda no último dia para contabilizar saldo
if posicao == 1:
    capital = qtd_acoes * precos_teste.iloc[-1]
    historico_capital[-1] = capital

# --- RESULTADOS ---
lucro = historico_capital[-1] - capital_inicial
roi = (lucro / capital_inicial) * 100
buy_hold_return = ((precos_teste.iloc[-1] - precos_teste.iloc[0]) / precos_teste.iloc[0]) * 100

print("\n" + "="*40)
print(f"   RESULTADOS FINAIS DE {ano_teste}")
print("="*40)
print(f"Capital Inicial:    {capital_inicial:.2f}€")
print(f"Capital Final (IA): {historico_capital[-1]:.2f}€")
print(f"Lucro/Prejuízo:     {roi:.2f}%")
print(f"Total de Trades:    {trades_feitos}")
print("-" * 40)
print(f"Retorno do Mercado (Buy & Hold): {buy_hold_return:.2f}%")

if roi > buy_hold_return:
    print("🏆 A IA BATEU O MERCADO! Excelente trabalho.")
else:
    print("📉 A IA perdeu para o mercado neste ano.")

# Gráfico
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, historico_capital, color='blue', label='A Tua IA', linewidth=2)
# Adicionar linha do Buy & Hold para comparação
norm_market = (precos_teste / precos_teste.iloc[0]) * capital_inicial
plt.plot(test_data.index, norm_market, color='gray', linestyle='--', label='Mercado (Buy & Hold)', alpha=0.6)

plt.title(f'Performance: IA vs Mercado em {ano_teste}')
plt.ylabel('Valor da Carteira (€)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
