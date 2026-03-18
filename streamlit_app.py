import re
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Plataforma de Risco Educacional", layout="wide")


# ---------------------------------------------------
# LOCALIZAR ARQUIVO DO MODELO
# ---------------------------------------------------

def localizar_modelo():

    caminhos = [
        "modelo_passos_magicos.pkl",
        "modelo_passos_magicos.joblib",
        "Modelos/modelo_passos_magicos.pkl",
        "Modelos/modelo_passos_magicos.joblib",
        "data/modelo_passos_magicos.pkl",
        "data/modelo_passos_magicos.joblib"
    ]

    for caminho in caminhos:

        if Path(caminho).exists():
            return caminho

    raise FileNotFoundError(
        "Arquivo do modelo não encontrado no repositório."
    )


# ---------------------------------------------------
# CARREGAR MODELO COM JOBLIB
# ---------------------------------------------------

@st.cache_resource
def carregar_modelo():

    caminho = localizar_modelo()

    modelo = joblib.load(caminho)

    return modelo


# ---------------------------------------------------
# EXTRAIR FASE
# ---------------------------------------------------

def extrair_fase(valor):

    if pd.isna(valor):
        return np.nan

    valor = str(valor).lower()

    if "alfa" in valor:
        return 0

    m = re.search(r"\d+", valor)

    if m:
        return int(m.group())

    return np.nan


# ---------------------------------------------------
# PREPARAÇÃO DOS DADOS
# ---------------------------------------------------

def preparar_base(df):

    df = df.copy()

    df["fase_ideal"] = df["fase_ideal"].apply(extrair_fase)

    df["media_academica"] = df[["mat", "por", "ing"]].mean(axis=1)

    df["media_comportamental"] = df[["iaa", "ieg", "ips", "ipp"]].mean(axis=1)

    df["delta_inde"] = df["inde_2023"] - df["inde_2022"]

    return df


# ---------------------------------------------------
# GARANTIR COLUNAS DO MODELO
# ---------------------------------------------------

def garantir_colunas_modelo(df, model):

    try:
        colunas_modelo = list(model.feature_names_in_)
    except:
        return df

    for c in colunas_modelo:

        if c not in df.columns:
            df[c] = np.nan

    df = df[colunas_modelo]

    return df


# ---------------------------------------------------
# SHAP
# ---------------------------------------------------

@st.cache_resource
def shap_explainer(_model):

    try:
        modelo = _model.named_steps["model"]
    except:
        modelo = _model

    return shap.TreeExplainer(modelo)


def grafico_shap(model, df):

    try:

        if "prep" in model.named_steps:
            X = model.named_steps["prep"].transform(df)
        else:
            X = df

        explainer = shap_explainer(model)

        shap_values = explainer(X)

        values = shap_values.values[0]

        fig = plt.figure()

        shap.plots.waterfall(values, show=False)

        return fig

    except:

        fig = plt.figure()

        plt.text(
            0.5,
            0.5,
            "Explicação SHAP não disponível",
            ha="center"
        )

        plt.axis("off")

        return fig


# ---------------------------------------------------
# INTERPRETAÇÃO DO RISCO
# ---------------------------------------------------

def interpretar_risco(prob):

    if prob < 0.30:
        return "🟢 Baixo risco", "Aluno apresenta trajetória educacional estável."

    if prob < 0.60:
        return "🟡 Atenção", "Aluno pode apresentar dificuldades educacionais."

    return "🔴 Alto risco", "Aluno com forte probabilidade de defasagem."


# ---------------------------------------------------
# TERMÔMETRO DE RISCO
# ---------------------------------------------------

def grafico_risco(prob):

    fig = plt.figure()

    plt.barh(["Risco"], [prob])

    plt.xlim(0, 1)

    plt.xlabel("Probabilidade")

    return fig


# ---------------------------------------------------
# INPUT DO USUÁRIO
# ---------------------------------------------------

def input_usuario():

    st.sidebar.header("Dados do aluno")

    idade = st.sidebar.number_input("Idade", 6, 20, 12)

    genero = st.sidebar.selectbox("Gênero", ["menino", "menina"])

    fase = st.sidebar.selectbox(
        "Fase",
        ["Alfa", "Fase 1", "Fase 2", "Fase 3", "Fase 4", "Fase 5"]
    )

    mat = st.sidebar.slider("Matemática", 0.0, 10.0, 5.0)
    por = st.sidebar.slider("Português", 0.0, 10.0, 5.0)
    ing = st.sidebar.slider("Inglês", 0.0, 10.0, 5.0)

    iaa = st.sidebar.slider("IAA", 0.0, 10.0, 5.0)
    ieg = st.sidebar.slider("IEG", 0.0, 10.0, 5.0)
    ips = st.sidebar.slider("IPS", 0.0, 10.0, 5.0)
    ipp = st.sidebar.slider("IPP", 0.0, 10.0, 5.0)

    inde_2022 = st.sidebar.slider("INDE 2022", 0.0, 10.0, 5.0)
    inde_2023 = st.sidebar.slider("INDE 2023", 0.0, 10.0, 5.0)

    ida = st.sidebar.slider("IDA", 0.0, 10.0, 5.0)
    ipv = st.sidebar.slider("IPV", 0.0, 10.0, 5.0)

    n_av = st.sidebar.number_input("Número de avaliações", 0, 20, 5)

    dados = {

        "idade": idade,
        "genero": genero,
        "fase_ideal": fase,

        "mat": mat,
        "por": por,
        "ing": ing,

        "iaa": iaa,
        "ieg": ieg,
        "ips": ips,
        "ipp": ipp,

        "inde_2022": inde_2022,
        "inde_2023": inde_2023,

        "ida": ida,
        "ipv": ipv,

        "n_av": n_av
    }

    return pd.DataFrame(dados, index=[0])


# ---------------------------------------------------
# APP
# ---------------------------------------------------

def main():

    st.title("🎓 Plataforma de Risco Educacional")

    model = carregar_modelo()

    df = input_usuario()

    if st.button("Analisar aluno"):

        df = preparar_base(df)

        df = garantir_colunas_modelo(df, model)

        prob = model.predict_proba(df)[0][1]

        col1, col2 = st.columns(2)

        with col1:

            st.metric("Probabilidade de risco", f"{prob*100:.2f}%")

            fig = grafico_risco(prob)

            st.pyplot(fig)

        with col2:

            status, texto = interpretar_risco(prob)

            st.subheader(status)

            st.write(texto)

        st.subheader("Fatores que influenciaram a decisão")

        fig = grafico_shap(model, df)

        st.pyplot(fig)


if __name__ == "__main__":
    main()
