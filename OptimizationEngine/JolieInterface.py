"""
    Ce fichier va permettre de créer une interface utilisateur entre
    le programme d'optimisation de patrimoine et l'utilisateur.
"""

import streamlit as st
import numpy as np
import time
import threeDOptiTE as opti
from streamlit.runtime.scriptrunner import add_script_run_ctx
from threading import Thread

st.title("""*Optimisation de patrimoine*""")
st.header("version 1.0 : gestion indicielle avec Markowitz")

on = st.toggle('Approche génétique', value=False)

if on:
    st.write("Cette fonctionnalité n'est pas encore disponible")
    number_of_assets = st.text_input(label="Combien d'actifs souhaitez vous détenir dans votre portefeuille ?", value="", disabled=False, key= 2)
    if number_of_assets == "":
        st.stop()
    try:
        number_of_assets = int(number_of_assets)
    except:
        st.stop()


index_to_track = st.text_input(label="Quel indice souhaitez-vous suivre ? (ex : S&P500, CAC40, ...)", value="", disabled=False)
if index_to_track == "":
    st.stop()




df, tracking_error = opti.solver(index_to_track)


progress_bar = st.sidebar.progress(0, text="Optimisation en cours...")
chart = st.line_chart(df, x='date', y=['index', 'portfolio'], height=500, width=1000)
for i in range(1, len(df['index'])):
    if i % 10 == 0:
        progress_bar.progress(int(i / len(df['index']) * 100), text="Optimisation en cours...")
        new_row_df = df.iloc[:i]
        chart.line_chart(new_row_df, x='date', y=['index', 'portfolio'], height=500, width=1000)

st.metric(label="Tracking error (%)", value=round(tracking_error * 100, 2))
progress_bar.empty()
st.button("Recommencer")
