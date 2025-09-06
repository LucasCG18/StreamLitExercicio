import pandas as pd
import numpy as np
import streamlit as st
from utils import load_data, save_data, get_numerical_cols

st.set_page_config(layout="wide", page_title="Dados - Breast Cancer")
st.title("Gerenciar dados")

df, data_path = load_data()
st.caption(f"Arquivo de dados: {data_path}")

with st.expander("Adicionar manualmente", expanded=False):
    if df is None:
        st.info("Se não houver dataset, criaremos um novo começando pela primeira linha inserida.")
        current_df = pd.DataFrame(columns=['diagnosis'])
        num_cols = []
    else:
        current_df = df.copy()
        num_cols = get_numerical_cols(current_df)
    with st.form("manual_add_form"):
        diag_label = st.selectbox("Diagnóstico", options=[0, 1], format_func=lambda x: "Benigno (0)" if x == 0 else "Maligno (1)")
        cols_left, cols_right = st.columns(2)
        values = {}
        for i, c in enumerate(num_cols):
            default_val = float(current_df[c].median()) if (df is not None and c in current_df.columns and not current_df[c].empty) else 0.0
            if i % 2 == 0:
                values[c] = cols_left.number_input(c, value=default_val)
            else:
                values[c] = cols_right.number_input(c, value=default_val)
        submitted = st.form_submit_button("Adicionar linha")
    if submitted:
        new_row = {**values, 'diagnosis': int(diag_label)}
        if df is None or current_df.empty:
            new_df = pd.DataFrame([new_row])
        else:
            # garantir que todas as colunas existentes permaneçam
            full_row = {col: (new_row[col] if col in new_row else (int(diag_label) if col == 'diagnosis' else np.nan)) for col in current_df.columns}
            # incluir colunas novas que não existiam ainda
            for k, v in new_row.items():
                if k not in full_row:
                    full_row[k] = v
            new_df = pd.concat([current_df, pd.DataFrame([full_row])], ignore_index=True)

        try:
            save_data(new_df, data_path)
            load_data.clear()
            st.success("Linha adicionada e salva.")
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao salvar: {e}")
