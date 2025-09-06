from pathlib import Path
import pandas as pd
import streamlit as st


def _find_dataset_path(nome: str = "Breast_cancer_dataset.csv") -> Path:
    candidatos = [
        Path(__file__).parent / 'pages' / nome,   # pages/
        Path(__file__).parent / nome,             # raiz do app
        Path.cwd() / nome                         # diretório corrente
    ]
    for p in candidatos:
        if p.exists():
            return p
    # padrão na pasta pages/
    return Path(__file__).parent / 'pages' / nome


@st.cache_data
def load_data(path=None):
    nome = "Breast_cancer_dataset.csv"
    resolved = Path(path) if path else _find_dataset_path(nome)
    if not resolved.exists():
        st.error(f"Arquivo '{nome}' não encontrado em {resolved}.")
        return None, resolved
    df = pd.read_csv(resolved)
    df.drop(columns=['id'], errors='ignore', inplace=True)
    if 'diagnosis' in df.columns:
        if df['diagnosis'].dtype == object:
            df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
        try:
            df['diagnosis'] = df['diagnosis'].astype(int)
        except Exception:
            pass
    return df, resolved


def save_data(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def get_numerical_cols(df: pd.DataFrame):
    cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns.tolist()
    if 'diagnosis' in cols:
        cols.remove('diagnosis')
    return cols

