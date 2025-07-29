import sklearn 
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

modelo = joblib.load("modelo_fut.pkl")

df = pd.read_csv("dados_times_regressao.csv")

st.title("Seu time vai ficar bem no campeonato?")

st.sidebar.header("Faça suas escolhas")
Finalizacoes = st.sidebar.slider("Quantas finalizacoes?", 1, 1000)
Chutes_no_Alvo = st.sidebar.slider("Chutes no alvo", 1, 500)
Gols_Sofridos = st.sidebar.slider("Gols sofridos", 1, 600)
Saldo_de_Gols = st.sidebar.slider("Saldo de gols", 1, 700)
Vitorias = st.sidebar.slider("Quantas vitorias seu time tem?", 0, 38, 1)

dados_usuário = pd.DataFrame(
    {
        "Finalizacoes": [Finalizacoes],
        "Chutes_no_Alvo": [Chutes_no_Alvo],
        "Gols_Sofridos": [Gols_Sofridos],
        "Saldo_de_Gols": [Saldo_de_Gols],
        "Vitorias": [Vitorias],
    }
)

st.subheader("Seus dados")
st.write(dados_usuário)

if st.button("Processar"):
    previsao = modelo.predict(dados_usuário)[0]
    st.subheader("Resultado da previsão")
    st.write("A possivel classificação do seu time é: ", previsao)



