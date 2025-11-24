# 📈 Advanced Quantitative Trading Strategy
> **Machine Learning Financeiro com Triple Barrier Method & Diferenciação Fracionária**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Sobre o Projeto

Este projeto implementa um pipeline de Machine Learning avançado para previsão de movimentos em mercados financeiros (Ações e Criptomoedas). Ao contrário de abordagens ingénuas que tentam prever o preço exato, este sistema foca-se na **gestão de risco** e na **probabilidade de eventos**.

A arquitetura baseia-se nas metodologias modernas de **Marcos Lopez de Prado** (*Advances in Financial Machine Learning*), utilizando técnicas robustas para evitar *overfitting* e capturar padrões não-lineares.

## 🚀 Key Features (A "Magia" Técnica)

### 1. Triple Barrier Method (Gestão de Risco Dinâmica)
Em vez de rótulos fixos ("Sobe 1%"), o modelo aprende a prever o resultado de três barreiras:
* **Take Profit:** Limite Superior (Dinâmico, baseado na Volatilidade).
* **Stop Loss:** Limite Inferior (Dinâmico, baseado na Volatilidade).
* **Time Horizon:** Limite de Tempo (O trade expira).
> *Resultado:* O modelo adapta-se a mercados calmos e voláteis, evitando falsos sinais em períodos de caos.

### 2. Diferenciação Fracionária (FracDiff)
Resolve o dilema "Estacionariedade vs. Memória".
* Dados brutos têm memória mas não são estacionários.
* Retornos simples são estacionários mas perdem a memória.
* **FracDiff ($d=0.4$):** Torna a série estacionária mantendo a correlação com o histórico de preços, permitindo ao modelo "ver" tendências de longo prazo.

### 3. Validação & Calibração
* **Walk-Forward Split:** Sem "olhar para o futuro" (Data Leakage).
* **Calibration Curves:** Análise de "Excesso de Confiança" do modelo para garantir que as probabilidades emitidas são realistas.

---

## 🛠️ Instalação e Requisitos

Clone o repositório e instale as dependências:

```bash
git clone [https://github.com/teu-usuario/nome-do-repo.git](https://github.com/teu-usuario/nome-do-repo.git)
cd nome-do-repo
pip install -r requirements.txt
