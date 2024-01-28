"""
    Ce fichier va permettre de créer une interface utilisateur entre
    le programme d'optimisation de patrimoine et l'utilisateur.
"""

import streamlit as st
import threeDOptiTE as opti
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("""*Optimisation de patrimoine*""")
st.header("version 1.0 : gestion indicielle avec Markowitz")

on = st.toggle('Approche génétique', value=False)

def clear_text():
    st.session_state.index_to_track = st.session_state.index_widget
    st.session_state.index_widget = ""
 

def asking_index_to_track():
    st.text_input(label="Quel indice souhaitez-vous suivre ? (ex : S&P500, CAC40, ...)", value="", key='index_widget' , on_change=clear_text)
    index_to_track = st.session_state.get('index_to_track', "")

    if index_to_track == "":
        st.stop()

    return index_to_track

def asking_esg_max():
    st.slider(label="Quel est le score ESG maximum que vous souhaitez avoir ?", min_value=7.24, max_value=100.0, value=7.24, step=0.5, key='esg_widget')
    esg_max = st.session_state.get('esg_widget', "")
    
    return esg_max

def asking_yield_min(min_yield, max_yield):
    st.slider(label="Quel est le rendement minimum que vous souhaitez avoir ?", min_value=min_yield, max_value=max_yield, value=0.0, step=0.01, key='yield_widget')
    yield_min = st.session_state.get('yield_widget', "")

    return yield_min


if on:
    st.write("Cette fonctionnalité n'est pas encore disponible")

    number_of_assets = st.text_input(label="Combien d'actifs souhaitez vous détenir dans votre portefeuille ?", value="", disabled=False, key= 2)

    index_to_track = asking_index_to_track()

    st.session_state.index_to_track = st.session_state.index_widget
    

    if number_of_assets == "":
        st.stop()
    try:
        number_of_assets = int(number_of_assets)
    except:
        st.stop()

else:
    esg_max = asking_esg_max()
    min_yield_threshold = asking_yield_min(-1.0, 11.0)

    index_to_track = asking_index_to_track()


    df, performance, weights = opti.solver(index_to_track, esg_max, min_yield_threshold)

    progress_bar = st.sidebar.progress(0, text="Optimisation en cours...")
    chart = st.line_chart(df, x='date', y=['index', 'portfolio'], height=500, width=2000)
    
    for i in range(1, len(df['index'])):
        if i % 10 == 0:
            progress_bar.progress(int(i / len(df['index']) * 100), text="Optimisation en cours...")
            new_row_df = df.iloc[:i]
            chart.line_chart(new_row_df, x='date', y=['index', 'portfolio'], height=500, width=2000)
    
    progress_bar.empty()


    index_to_track = ""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Tracking error (%)", value=round(performance['tracking_error'] * 100, 3))
    col2.metric(label="ESG score", value=round(performance['esg_score'], 3))
    col3.metric(label="Rendement du portefeuille (%)", value=round(performance['portfolio_return'] * 100, 3))
    col4.metric(label="Volatilité du portefeuille (%)", value=round(performance['portfolio_risk'] * 100, 3))

    labels = list(weights.keys())
    values = list(weights.values())
    # print true if weights are all positive
    explode = [0.1] * len(values)

    fig1, ax1 = plt.subplots()
    ax1.pie(values, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90, explode=explode, textprops={'fontsize': 5})
    ax1.axis('equal')

    st.pyplot(fig1)
    

    st.button("Recommencer")
    st.stop()
