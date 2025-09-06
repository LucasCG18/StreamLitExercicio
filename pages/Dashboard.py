import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data, get_numerical_cols

st.set_page_config(layout="wide", page_title="Exploração - Breast Cancer")

df, data_path = load_data()
if df is None:
    st.stop()
numerical_cols = [c for c in get_numerical_cols(df) if c in df.columns]
st.title("Gráficos - Cancer de mama")

st.sidebar.title("Filtros e talvez alguma Opções 😊 ")
diagnosis_filter = st.sidebar.multiselect(
    "Diagnóstico", options=[0, 1], default=[0, 1],
    format_func=lambda x: "Benigno (0)" if x == 0 else "Maligno (1)"
)
radius_thresh = st.sidebar.slider(
    "Definir media do raio para o gráfico de barras (deixei como padrao 11.89 pois da para ver algo legal)",
    float(df['radius_mean'].min()), float(df['radius_mean'].max()), 11.89
)
box_features = st.sidebar.multiselect(
    "Selecione até 3 variaveis para boxplots", options=numerical_cols,
    default=numerical_cols[:3]
)
corr_method = st.sidebar.selectbox("Método de correlação", options=['pearson', 'spearman', 'kendall'], index=1)

df_filtrado = df[df['diagnosis'].isin(diagnosis_filter)]

st.header("Scatter: média do raio x média da textura")
if 'radius_mean' in df_filtrado.columns and 'texture_mean' in df_filtrado.columns:
    fig_scatter = px.scatter(
        df_filtrado, x='radius_mean', y='texture_mean',
    color=df_filtrado['diagnosis'].map({0: 'Benigno', 1: 'Maligno'}),
    labels={'color': 'Diagnóstico'},
    hover_data=[c for c in ['radius_worst', 'texture_worst'] if c in df_filtrado.columns],
    opacity=0.7
    )
    fig_scatter.update_layout(title="Relação entre Raio Médio e Textura Média")
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Para o scatter, é preciso 'radius_mean' e 'texture_mean'.")

st.header("Boxplots em grupos de até 3")
if not box_features:
    st.info("Selecione até 3 coisas na barra lateral.")
else:
    cols_to_show = [c for c in box_features[:3] if c in df.columns]
    if not cols_to_show:
        st.warning("Nenhuma das features selecionadas existe no DataFrame.")
    else:
        df_melt = df_filtrado.melt(
            id_vars=['diagnosis'], value_vars=cols_to_show, var_name='feature', value_name='value'
        )
        df_melt['diagnosis_label'] = df_melt['diagnosis'].map({0: 'Benigno', 1: 'Maligno'}).fillna('Desconhecido')

        fig_boxes = px.box(
            df_melt,
            x='diagnosis_label',
            y='value',
            color='diagnosis_label',
            facet_col='feature',
            facet_col_wrap=3,
            category_orders={'feature': cols_to_show},
            points='outliers',
            labels={'diagnosis_label': 'Diagnóstico', 'value': 'Valor', 'feature': 'Feature'},
            color_discrete_map={'Benigno': '#636EFA', 'Maligno': '#EF553B', 'Desconhecido': '#AAAAAA'}
        )
        fig_boxes.update_layout(title=f"Boxplots: {', '.join(cols_to_show)}", showlegend=False)
        st.plotly_chart(fig_boxes, use_container_width=True)

st.header("Histograma / Densidade: Média do Raio")

if 'radius_mean' not in df.columns:
    st.warning("Coluna 'radius_mean' não encontrada no DataFrame.")
else:
    df_hist = df_filtrado.copy()
    df_hist['radius_mean'] = pd.to_numeric(df_hist['radius_mean'], errors='coerce')
    df_hist = df_hist.dropna(subset=['radius_mean'])

    if df_hist.empty:
        st.warning("Nenhum dado válido para 'radius_mean' após filtragem.")
    else:
        if 'diagnosis' in df_hist.columns:
            df_hist['diagnosis_label'] = (
                df_hist['diagnosis'].map({0: 'Benigno', 1: 'Maligno'}).fillna(df_hist['diagnosis']).astype(str)
            )
        else:
            df_hist['diagnosis_label'] = 'Desconhecido'

        try:
            fig_hist = px.histogram(
                df_hist,
                x='radius_mean',
                color='diagnosis_label',
                marginal="density",
                barmode='overlay',
                opacity=0.6,
                nbins=40,
                labels={'diagnosis_label': 'Diagnóstico', 'radius_mean': 'radius_mean'}
            )
        except Exception:
            fig_hist = px.histogram(
                df_hist,
                x='radius_mean',
                color='diagnosis_label',
                barmode='overlay',
                opacity=0.6,
                nbins=40,
                labels={'diagnosis_label': 'Diagnóstico', 'radius_mean': 'radius_mean'}
            )

        fig_hist.update_layout(title="Distribuição da média do raio por diagnóstico")
        st.plotly_chart(fig_hist, use_container_width=True)

st.header(f"Matriz de Correlação ({corr_method})")
corr_cols = [c for c in numerical_cols if c in df.columns]
if 'diagnosis' in df.columns:
    corr_cols = corr_cols + ['diagnosis']
if len(corr_cols) >= 2:
    corr_matrix = df[corr_cols].corr(method=corr_method)
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r')
    fig_corr.update_layout(title=f"Matriz de Correlação ({corr_method})")
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("Colunas insuficientes para matriz de correlação.")

st.header(f"Contagem por Diagnóstico (Média do raio > {radius_thresh})")
col1, col2 = st.columns(2)
if 'radius_mean' in df.columns:
    df_gt = df[df['radius_mean'] > radius_thresh]
    df_le = df[df['radius_mean'] <= radius_thresh]
else:
    df_gt = df.copy()[:0]
    df_le = df.copy()[:0]

with col1:
    st.subheader(f"Raio > {radius_thresh}")
    vc_gt = (
        df_gt['diagnosis'].map({0: 'Benigno', 1: 'Maligno'}).value_counts().reindex(['Benigno', 'Maligno']).fillna(0)
    )
    fig_gt = px.bar(
        x=vc_gt.index, y=vc_gt.values, color=vc_gt.index,
        labels={'x': 'Diagnóstico', 'y': 'Contagem'}, title="Radius > threshold"
    )
    st.plotly_chart(fig_gt, use_container_width=True)
with col2:
    st.subheader(f"Raio ≤ {radius_thresh}")
    vc_le = (
        df_le['diagnosis'].map({0: 'Benigno', 1: 'Maligno'}).value_counts().reindex(['Benigno', 'Maligno']).fillna(0)
    )
    fig_le = px.bar(
        x=vc_le.index, y=vc_le.values, color=vc_le.index,
        labels={'x': 'Diagnóstico', 'y': 'Contagem'}, title="Radius ≤ threshold"
    )
    st.plotly_chart(fig_le, use_container_width=True)
