import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data, get_numerical_cols

st.set_page_config(layout="wide", page_title="XGBoost - Breast Cancer")
st.title("Modelo: XGBoost")

try:
    from xgboost import XGBClassifier  # type: ignore
    XGB_OK = True
except Exception as e:
    XGB_OK = False
    XGB_IMPORT_ERR = e

df, _ = load_data()
if df is None or 'diagnosis' not in df.columns:
    st.info("Precisamos de dados com a coluna 'diagnosis' para treinar.")
    st.stop()

feature_options = get_numerical_cols(df)
st.caption("Treinando com todas as colunas numéricas (imputação por mediana): " + ", ".join(feature_options))
selected_features = feature_options
test_size = st.slider("Tamanho do teste", 0.1, 0.4, 0.2, 0.05)
random_state = st.number_input("Random state", value=42)
st.caption("Random state define a semente aleatória usada na separação treino/teste e no modelo. \n"
          "Usar o mesmo valor torna os resultados reprodutíveis; mudar o valor altera a amostragem.")
train_btn = st.button("Treinar XGBoost")

if not XGB_OK:
    st.info("Pacote XGBoost não instalado. Instale com: pip install xgboost")

if train_btn and XGB_OK:
    if not selected_features:
        st.warning("Selecione ao menos uma feature.")
    else:
        X = df[selected_features].copy()
        y = df['diagnosis'].astype(int)
        imputer = SimpleImputer(strategy='median')
        X_imp = imputer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_imp, y, test_size=float(test_size), random_state=int(random_state), stratify=y
        )
        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=int(random_state),
            eval_metric='logloss',
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        try:
            auc_val = roc_auc_score(y_test, y_proba)
        except Exception:
            auc_val = float('nan')
        st.metric("Acurácia", f"{acc:.3f}")
        st.metric("ROC AUC", f"{auc_val:.3f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                           labels=dict(x="Predito", y="Verdadeiro", color="Contagem"))
        fig_cm.update_xaxes(tickvals=[0, 1], ticktext=["Benigno (0)", "Maligno (1)"])
        fig_cm.update_yaxes(tickvals=[0, 1], ticktext=["Benigno (0)", "Maligno (1)"])
        st.plotly_chart(fig_cm, use_container_width=True)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"ROC (AUC={roc_auc:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Aleatório', line=dict(dash='dash')))
        fig_roc.update_layout(xaxis_title='FPR', yaxis_title='TPR', title='Curva ROC')
        st.plotly_chart(fig_roc, use_container_width=True)
        st.markdown(
            """
            • ROC (Receiver Operating Characteristic) mostra o equilíbrio entre TPR (taxa de verdadeiros positivos) e FPR (taxa de falsos positivos) em vários limiares de decisão.
            
            • AUC (Área sob a curva) resume a performance: 0.5 ≈ aleatório, 1.0 ≈ perfeito. Quanto maior o AUC, melhor a separação entre classes.
            
            • O ponto ideal depende do custo de falsos positivos/negativos; use a curva para escolher o limiar adequado.
            """
        )

        # Feature Importances
        try:
            importances = model.feature_importances_
            imp_df = pd.DataFrame({
                'feature': selected_features,
                'importance': importances
            }).sort_values('importance', ascending=False)
            fig_imp = px.bar(imp_df, x='feature', y='importance', title='Importância das Features')
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception:
            pass

        st.session_state['xgb_model'] = model
        st.session_state['xgb_features'] = selected_features
        st.session_state['xgb_imputer'] = imputer

st.divider()
st.subheader("Prever um novo caso (XGBoost)")
feat_cols = st.session_state.get('xgb_features', selected_features)
if not feat_cols:
    st.info("Treine o modelo acima para habilitar a predição.")
else:
    defaults = {c: float(df[c].median()) if c in df.columns else 0.0 for c in feat_cols}
    with st.form("xgb_predict_form"):
        cols1, cols2 = st.columns(2)
        sample_vals = {}
        for i, c in enumerate(feat_cols):
            if i % 2 == 0:
                sample_vals[c] = cols1.number_input(c, value=defaults[c])
            else:
                sample_vals[c] = cols2.number_input(c, value=defaults[c])
        run_pred = st.form_submit_button("Prever com XGBoost")

    if run_pred:
        if not XGB_OK:
            st.warning("XGBoost não está disponível.")
        elif 'xgb_model' not in st.session_state or 'xgb_imputer' not in st.session_state:
            st.warning("Treine o modelo primeiro.")
        else:
            imputer: SimpleImputer = st.session_state['xgb_imputer']
            model: XGBClassifier = st.session_state['xgb_model']
            sample_df = pd.DataFrame([sample_vals])[feat_cols]
            X_new = imputer.transform(sample_df)
            proba = model.predict_proba(X_new)[0, 1]
            pred = int(proba >= 0.5)
            st.write(f"Prob. Maligno: {proba:.3f} — Predição: {'Maligno (1)' if pred==1 else 'Benigno (0)'}")
