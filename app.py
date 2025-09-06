import streamlit as st

pg = st.navigation([
	st.Page("pages/Dashboard.py", title="Exploração", icon="📊"),
	st.Page("pages/Data.py", title="Dados", icon="🧾"),
	st.Page("pages/RandomForest.py", title="RandomForest", icon="🎲"),
	st.Page("pages/XGBoost.py", title="XGBoost", icon="⚡"),
])
pg.run()