"""
    Ce fichier va permettre de créer une interface utilisateur entre
    le programme d'optimisation de patrimoine et l'utilisateur.
"""

import streamlit as st
import time as t
import OptimizationEngine.threeDOptiTE as opti
import matplotlib.pyplot as plt
import DataProvider.translate_symbol as translate_symbol

st.set_page_config(layout="wide")
st.title("""*Optimisation de patrimoine*""")
st.header("version 1.0 : gestion indicielle avec Markowitz")

on = st.toggle('Approche génétique', value=False)

def clear_text():
    st.session_state.index_to_track = st.session_state.index_widget
    st.session_state.index_widget = ""
 

def asking_index_to_track(col, i):
    col.text_input(label="Quel indice souhaitez-vous suivre ? (ex : S&P500, CAC40, ...)", value="", key='index_widget' + str(i))
    index_to_track = st.session_state.get('index_to_track' + str(i) , "")

    return index_to_track

def asking_weight(col, i):
    col.slider(label="Quel est le poids que vous souhaitez donner à cet indice ?", min_value=0.00, max_value=1.00, value=0.00, step=0.001, key='weight_widget'+ str(i))
    weight = st.session_state.get('weight_widget'+ str(i), "")

    return weight

def asking_esg_max():
    st.slider(label="Quel est le score ESG maximum que vous souhaitez avoir ?", min_value=6.6, max_value=44.9, value=10.0, step=0.5, key='esg_widget')
    esg_max = st.session_state.get('esg_widget', "")
    
    return esg_max

def asking_yield_min(min_yield, max_yield):
    st.slider(label="Quel est le rendement minimum que vous souhaitez avoir ?", min_value=min_yield, max_value=max_yield, value=0.0, step=0.01, key='yield_widget')
    yield_min = st.session_state.get('yield_widget', "")

    return yield_min

def asking_number_of_assets():
    st.selectbox(label="Combien d'actifs souhaitez vous détenir dans votre portefeuille ?", options=[1, 2, 3], key='number_of_assets_widget')
    option = st.session_state.get('number_of_assets_widget', "")

    return option



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

    option = asking_number_of_assets()

    index_to_track = []

    for i in range(option):
        col1, col2 = st.columns(2)
        index = asking_index_to_track(col1, i)
        weight = asking_weight(col2, i)

        index_to_track.append((index, weight))
        
    st.stop()

    df, performance, index_performance, weights = opti.solver(esg_max, index_to_track)

    progress_bar = st.sidebar.progress(0, text="Optimisation en cours...")
    chart = st.line_chart(df, x='date', y=['index', 'portfolio'], height=500, width=2000)
    
    for i in range(1, len(df['index'])):
        if i % 10 == 0:
            progress_bar.progress(int(i / len(df['index']) * 100), text="Optimisation en cours...")
            new_row_df = df.iloc[:i]
            chart.line_chart(new_row_df, x='date', y=['index', 'portfolio'], height=500, width=2000)
    
    progress_bar.empty()

    col1, col2, col3, col4 = st.columns(4)
    col3.metric(label="Rendement indice (%)", value=round(index_performance['index_return'] * 100, 3))
    col4.metric(label="Volatilité indice (%)", value=round(index_performance['index_risk'] * 100, 3))

    col1, col2, col3, col4 = st.columns(4)
    delta_return = round(performance['portfolio_return'] - index_performance['index_return'], 3)
    delta_risk = round(performance['portfolio_risk'] - index_performance['index_risk'], 4)

    col1.metric(label="Tracking error (%)", value=round(performance['tracking_error'] * 100, 3))
    col2.metric(label="ESG score", value=round(performance['esg_score'], 3))
    col3.metric(label="Rendement portefeuille (%)", value=round(performance['portfolio_return'] * 100, 3), delta=f"{delta_return*100}%")
    col4.metric(label="Volatilité portefeuille (%)", value=round(performance['portfolio_risk'] * 100, 3), delta=f"{delta_risk*100}%")

    labels = list(weights.keys())
    labels = [translate_symbol.translate_symbol(label) for label in labels]
    values = list(weights.values())
    explode = [0.1] * len(values)

    fig1, ax1 = plt.subplots()
    # delete labels with less than 2% of weight
    for i in range(len(values)):
        if values[i] < 0.02:
            labels[i] = ""

    ax1.pie(values, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90, explode=explode, textprops={'fontsize': 5})
    ax1.axis('equal')

    st.pyplot(fig1)
    

    st.button("Recommencer")
    st.stop()
