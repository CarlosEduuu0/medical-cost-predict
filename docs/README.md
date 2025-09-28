# 💰 MedCost Predict
---

## 📌 Descrição
Este projeto tem como objetivo analisar o dataset **Insurance** (custos médicos) e construir modelos de **regressão** para prever os valores de despesas médicas (`charges`).  
Foram aplicadas técnicas de **EDA (Exploração de Dados)**, pré-processamento e diferentes modelos de machine learning, como **Regressão Linear, Random Forest, XGBoost e SVR**.  

---
## 📊 dashboard
-6 graficos para entender o perfil etario e como certas variaveis afetam os charges
-acesso aos modelos para rever valores de acordo com o registro
-mudança de tema

---

## 📂 Estrutura do Projeto
```plaintext
medical-cost-predict/
├── .venv/                 # Ambiente virtual
├── __pycache__/           # Cache de Python
├── dashboard/             # Aplicação Dash (frontend interativo)
├── docs/                  # Documentação
├── modelos-encoder/       # Modelos treinados / encoders salvos
├── anlysis.ipynb          # Notebook de análise e modelagem
├── insurance.csv          # Dataset


## ⚙️ Instalação

1-Clone este repositório:

git clone https://github.com/seu-usuario/medcost-predict.git
cd medcost-predict

2-crie um ambiente vitual com pyton pelo menos 3.10

3-instale um requiremnets.txt em docs

pip install -r requirements.txt

4-rode o dashboard.py e acessa url
