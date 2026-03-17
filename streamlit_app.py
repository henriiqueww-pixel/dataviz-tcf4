import io
import re
import unicodedata
import os
import datetime

# Bibliotecas
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

validar_shap = 'n'

st.set_page_config(
    page_title="Predição de Risco de Defasagem",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =============================
# FUNÇÕES DE PREPARO DE DADOS
# =============================

def coerce_numeric(s):
    return pd.to_numeric(s, errors="coerce")

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


def padronizar_genero(df):

    df = df.copy()

    if "genero" in df.columns:
        df["genero"] = df["genero"].astype(str).str.strip().str.lower()

        map_genero = {
            "menino": "masculino",
            "masculino": "masculino",
            "menina": "feminino",
            "feminino": "feminino"
        }

        df["genero"] = df["genero"].map(map_genero)

    return df


def padronizar_idade(df):

    df = df.copy()

    if "idade" not in df.columns:
        return df

    s = df["idade"]

    dt = pd.to_datetime(s, errors="coerce")

    idade_from_date = np.where(
        dt.notna() & (dt.dt.year == 1900) & (dt.dt.month == 1),
        dt.dt.day,
        np.nan
    )

    idade_num = pd.to_numeric(s, errors="coerce")

    idade_final = pd.Series(idade_num, index=df.index)

    mask = idade_final.isna() & ~pd.isna(idade_from_date)

    idade_final.loc[mask] = idade_from_date[mask]

    idade_final = idade_final.where(idade_final.between(6, 30))

    df["idade"] = idade_final.round()

    return df


def tratar_inde_2024(df):

    df = df.copy()

    if "inde_2024" in df.columns:

        tmp = df["inde_2024"].astype(str).str.strip().str.upper()

        tmp = tmp.replace("INCLUIR", np.nan)

        df["inde_2024"] = coerce_numeric(tmp)

    return df


def preparar_base_app(df):

    df = df.copy()

    df = padronizar_genero(df)

    df = padronizar_idade(df)

    df = tratar_inde_2024(df)

    if "fase_ideal" in df.columns:
        df["fase_ideal"] = df["fase_ideal"].apply(extrair_fase)

    cols_acad = [c for c in ["mat", "por", "ing"] if c in df.columns]

    if len(cols_acad) >= 2:
        df["media_academica"] = df[cols_acad].mean(axis=1)

    cols_comp = [c for c in ["iaa", "ieg", "ips", "ipp"] if c in df.columns]

    if len(cols_comp) >= 2:
        df["media_comportamental"] = df[cols_comp].mean(axis=1)

    if ("inde_2022" in df.columns) and ("inde_2023" in df.columns):

        df["delta_inde"] = df["inde_2023"] - df["inde_2022"]

    return df


# =============================
# TRADUÇÃO FEATURES
# =============================

def traduzir_nomes_features(lista_nomes_tecnicos):

    mapa_nomes = {
        'num__idade': 'Idade do Aluno',
        'num__inde_2024': 'Índice INDE (Atual)',
        'num__media_academica': 'Média Acadêmica',
        'num__media_comportamental': 'Média Comportamental',
        'num__delta_inde': 'Evolução do INDE',
        'num__fase_ideal': 'Fase Ideal',
        'cat__genero_masculino': 'Gênero Masculino',
        'cat__genero_feminino': 'Gênero Feminino',
        'num__ida': 'Indicador Acadêmico',
        'num__ipv': 'Indicador Ponto de Virada',
        'num__n_av': 'Número de Avaliações'
    }

    nomes_traduzidos = []

    for nome in lista_nomes_tecnicos:

        if nome in mapa_nomes:
            nomes_traduzidos.append(mapa_nomes[nome])

        else:

            limpo = nome.replace('num__', '').replace('cat__', '').replace('_', ' ').title()

            nomes_traduzidos.append(limpo)

    return nomes_traduzidos


# =============================
# CARREGAR MODELO (.JOBLIB)
# =============================

@st.cache_resource
def load_models_and_config():

    caminho_modelo = os.path.join("Modelos", "modelo_passos_magicos.joblib")

    caminho_config = os.path.join("Modelos", "config_passos_magicos.joblib")

    try:

        modelo = joblib.load(caminho_modelo)

        config = joblib.load(caminho_config)

        return modelo, config

    except FileNotFoundError:

        st.error(
            "Arquivos .joblib não encontrados. "
            "Verifique se a pasta 'Modelos' contém os arquivos:\n"
            "- modelo_passos_magicos.joblib\n"
            "- config_passos_magicos.joblib"
        )

        return None, None


# =============================
# SHAP
# =============================

@st.cache_resource
def _get_shap_explainer(_classifier):

    return shap.TreeExplainer(_classifier)


def gerar_explicacao_shap(model, input_df_processed):

    try:

        preprocessor = model.named_steps['prep']

        classifier = model.named_steps['model']

        input_transformed = preprocessor.transform(input_df_processed)

        feature_names_raw = preprocessor.get_feature_names_out()

        feature_names_pt = traduzir_nomes_features(feature_names_raw)

        explainer = _get_shap_explainer(classifier)

        shap_values = explainer(input_transformed)

        if len(shap_values.shape) == 3:
            shap_values_to_plot = shap_values[0, :, 1]
        else:
            shap_values_to_plot = shap_values[0]

        shap_values_to_plot.feature_names = feature_names_pt

        fig, ax = plt.subplots(figsize=(10, 6))

        shap.plots.waterfall(shap_values_to_plot, show=False, max_display=10)

        return plt.gcf()

    except Exception as e:

        st.error(f"Erro SHAP: {e}")

        return None


# =============================
# APP PRINCIPAL
# =============================

def main():

    model, config = load_models_and_config()

    st.title("🎓 Previsão de Defasagem Educacional")

    st.markdown(
        "Analise o risco de defasagem educacional com base nos indicadores do aluno."
    )

    if model is None:
        return

    idade = st.number_input("Idade", 6, 30, 12)

    genero = st.selectbox("Gênero", ["Menino", "Menina"])

    fase = st.selectbox(
        "Fase",
        ["Alfa", "Fase 1", "Fase 2", "Fase 3", "Fase 4"]
    )

    mat = st.number_input("Matemática", 0.0, 10.0)

    por = st.number_input("Português", 0.0, 10.0)

    ing = st.number_input("Inglês", 0.0, 10.0)

    if st.button("Realizar Predição"):

        data = {

            "idade": idade,
            "genero": genero,
            "fase_ideal": fase,
            "mat": mat,
            "por": por,
            "ing": ing

        }

        df = pd.DataFrame(data, index=[0])

        processed = preparar_base_app(df)

        prob = model.predict_proba(processed)[0][1]

        st.metric("Probabilidade de Risco", f"{prob*100:.2f}%")

        fig = gerar_explicacao_shap(model, processed)

        if fig:
            st.pyplot(fig)


if __name__ == "__main__":
    main()
