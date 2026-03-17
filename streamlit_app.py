import os
import re
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Análise de Risco Educacional",
    page_icon="🎓",
    layout="wide"
)

# =========================================================
# TRATAMENTO DE DADOS
# =========================================================

def extrair_fase(valor):

    if pd.isna(valor):
        return np.nan

    valor = str(valor).lower()

    if "alfa" in valor:
        return 0

    m = re.search(r"fase\s*(\d+)", valor)

    if m:
        return int(m.group(1))

    return np.nan


def preparar_base(df):

    df = df.copy()

    df["genero"] = df["genero"].str.lower()

    df["fase_ideal"] = df["fase_ideal"].apply(extrair_fase)

    df["media_academica"] = df[["mat","por","ing"]].mean(axis=1)

    df["media_comportamental"] = df[["iaa","ieg","ips","ipp"]].mean(axis=1)

    df["delta_inde"] = df["inde_2023"] - df["inde_2022"]

    return df


# =========================================================
# PROTEÇÃO DE COLUNAS
# =========================================================

def garantir_colunas(df):

    cols = [
        "iaa","ieg","ips","ipp",
        "inde_2022","inde_2023",
        "ida","ipv","n_av",
        "media_comportamental",
        "delta_inde"
    ]

    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    return df


def alinhar_colunas(df, config):

    ordem = config["features_modelo"]

    for c in ordem:

        if c not in df.columns:
            df[c] = np.nan

    return df[ordem]


# =========================================================
# CARREGAR MODELO
# =========================================================

@st.cache_resource
def carregar_modelo():

    modelo = joblib.load("Modelos/modelo_passos_magicos.joblib")

    config = joblib.load("Modelos/config_passos_magicos.joblib")

    return modelo, config


# =========================================================
# SHAP
# =========================================================

@st.cache_resource
def shap_explainer(model):

    return shap.TreeExplainer(model.named_steps["model"])


def grafico_shap(model, df):

    prep = model.named_steps["prep"]

    X = prep.transform(df)

    explainer = shap_explainer(model)

    shap_values = explainer(X)

    values = shap_values[0,:,1]

    fig = plt.figure(figsize=(8,5))

    shap.plots.waterfall(values, show=False)

    return fig


# =========================================================
# INTERPRETAÇÃO AUTOMÁTICA
# =========================================================

def interpretar_risco(prob):

    if prob < 0.30:
        return "🟢 Baixo risco", "Aluno apresenta trajetória educacional estável."

    if prob < 0.60:
        return "🟡 Atenção", "Aluno pode apresentar dificuldades futuras. Monitoramento recomendado."

    return "🔴 Alto risco", "Aluno com forte probabilidade de defasagem. Intervenção pedagógica recomendada."


# =========================================================
# TERMÔMETRO DE RISCO
# =========================================================

def grafico_risco(prob):

    fig = plt.figure()

    plt.barh(["Risco"], [prob])

    plt.xlim(0,1)

    plt.xlabel("Probabilidade")

    return fig


# =========================================================
# INPUT USUÁRIO
# =========================================================

def inputs_usuario():

    st.sidebar.header("Dados do Aluno")

    idade = st.sidebar.number_input("Idade",6,25,12)

    genero = st.sidebar.selectbox("Gênero",["menino","menina"])

    fase = st.sidebar.selectbox(
        "Fase",
        ["Alfa","Fase 1","Fase 2","Fase 3","Fase 4","Fase 5"]
    )

    mat = st.sidebar.slider("Matemática",0.0,10.0,5.0)
    por = st.sidebar.slider("Português",0.0,10.0,5.0)
    ing = st.sidebar.slider("Inglês",0.0,10.0,5.0)

    iaa = st.sidebar.slider("IAA",0.0,10.0,5.0)
    ieg = st.sidebar.slider("IEG",0.0,10.0,5.0)
    ips = st.sidebar.slider("IPS",0.0,10.0,5.0)
    ipp = st.sidebar.slider("IPP",0.0,10.0,5.0)

    inde_2022 = st.sidebar.slider("INDE 2022",0.0,10.0,5.0)
    inde_2023 = st.sidebar.slider("INDE 2023",0.0,10.0,5.0)
    inde_2024 = st.sidebar.slider("INDE Atual",0.0,10.0,5.0)

    ida = st.sidebar.slider("IDA",0.0,10.0,5.0)
    ipv = st.sidebar.slider("IPV",0.0,10.0,5.0)

    n_av = st.sidebar.number_input("Número de avaliações",0,50,10)

    data = {

        "idade":idade,
        "genero":genero,
        "fase_ideal":fase,

        "mat":mat,
        "por":por,
        "ing":ing,

        "iaa":iaa,
        "ieg":ieg,
        "ips":ips,
        "ipp":ipp,

        "inde_2022":inde_2022,
        "inde_2023":inde_2023,
        "inde_2024":inde_2024,

        "ida":ida,
        "ipv":ipv,
        "n_av":n_av
    }

    return pd.DataFrame(data,index=[0])


# =========================================================
# APP
# =========================================================

def main():

    st.title("🎓 Plataforma de Análise de Risco Educacional")

    st.write(
        "Sistema de apoio pedagógico para identificação de alunos "
        "com risco de defasagem escolar."
    )

    model, config = carregar_modelo()

    df = inputs_usuario()

    if st.button("Analisar aluno"):

        df = preparar_base(df)

        df = garantir_colunas(df)

        df = alinhar_colunas(df, config)

        prob = model.predict_proba(df)[0][1]

        col1,col2 = st.columns(2)

        with col1:

            st.subheader("Probabilidade de risco")

            st.metric("Risco",f"{prob*100:.1f}%")

            fig = grafico_risco(prob)

            st.pyplot(fig)

        with col2:

            status,msg = interpretar_risco(prob)

            st.subheader(status)

            st.write(msg)

        st.subheader("Fatores que influenciaram a decisão do modelo")

        fig = grafico_shap(model,df)

        st.pyplot(fig)


if __name__ == "__main__":
    main()
