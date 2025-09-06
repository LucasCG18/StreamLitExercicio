# Breast Cancer Streamlit App

Aplicação Streamlit com 4 páginas: Exploração, Dados, RandomForest e XGBoost.

## Como rodar (Windows / PowerShell)

1. Criar e ativar venv:
   - `python -m venv .venv`
   - `./.venv/Scripts/Activate.ps1`
2. Instalar dependências:
   - `pip install -r requirements.txt`
3. Rodar o app:
   - `streamlit run app.py`

## Estrutura
- `app.py` – navegação entre páginas
- `pages/` – páginas Streamlit e dataset CSV
- `utils.py` – utilitários de dados
- `requirements.txt` – dependências
- `.gitignore` – ignora venv e artefatos

## Observações
- O XGBoost é opcional; se não estiver instalado, a página informa como instalar.
