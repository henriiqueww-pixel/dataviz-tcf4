import io
import re
import unicodedata
import os
import datetime

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
            "menino": "masculino", "masculino": "masculino",
            "menina": "feminino", "feminino": "feminino"
        }
        df["genero"] = df["genero"].map(map_genero)

    return df

def padronizar_idade(df):
    df = df.copy()
    if "idade" not in df.columns:
        return df
    
    s = df["idade"]
    dt = pd.to_datetime(s, errors="coerce")
    idade_from_date = np.where(dt.notna() & (dt.dt.year == 1900) & (dt.dt.month == 1), dt.dt.day, np.nan)
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

    cols_acad = [c for c in ["mat","por","ing"] if c in df.columns]
    if len(cols_acad) >= 2:
        df["media_academica"] = df[cols_acad].mean(axis=1)

    cols_comp = [c for c in ["iaa","ieg","ips","ipp"] if c in df.columns]
    if len(cols_comp) >= 2:
        df["media_comportamental"] = df[cols_comp].mean(axis=1)

    if ("inde_2022" in df.columns) and ("inde_2023" in df.columns):
        df["delta_inde"] = df["inde_2023"] - df["inde_2022"]

    return df

def garantir_colunas_modelo(df):

    colunas_esperadas = [
        'iaa','ieg','ips','ipp',
        'inde_2022','inde_2023',
        'ida','ipv','n_av',
        'media_comportamental',
        'delta_inde'
    ]

    for col in colunas_esperadas:
        if col not in df.columns:
            df[col] = np.nan

    return df

def traduzir_nomes_features(lista_nomes_tecnicos):

    mapa_nomes = {
        'num__idade': 'Idade do Aluno',
        'num__inde_2024': 'Índice INDE (Atual)',
        'num__media_academica': 'Média Acadêmica (Mat, Por, Ing)',
        'num__media_comportamental': 'Média Comportamental (IAA, IEG, IPS, IPP)',
        'num__delta_inde': 'Evolução do INDE (Últimos 2 anos)',
        'num__fase_ideal': 'Fase Ideal',
        'cat__genero_masculino': 'Gênero (Masculino)',
        'cat__genero_feminino': 'Gênero (Feminino)',
        'num__ida': 'Indicador de Desemp. Acad. (IDA)',
        'num__ipv': 'Indicador de Ponto de Virada (IPV)',
        'num__n_av': 'Número de Avaliações'
    }
    
    nomes_traduzidos = []

    for nome in lista_nomes_tecnicos:
        if nome in mapa_nomes:
            nomes_traduzidos.append(mapa_nomes[nome])

        else:
            limpo = nome.replace('num__', '').replace('cat__', '').replace('bin__', '').replace('_', ' ').title()
            nomes_traduzidos.append(limpo)
            
    return nomes_traduzidos

@st.cache_resource 
def load_models_and_config():
    
    caminho_modelo = os.path.join("Modelos", "modelo_passos_magicos.joblib")
    caminho_config = os.path.join("Modelos", "config_passos_magicos.joblib")
    
    try:
        modelo = joblib.load(caminho_modelo)
        config = joblib.load(caminho_config)
        return modelo, config
        
    except FileNotFoundError:
        st.error("Arquivos .joblib não encontrados na pasta Modelos.")
        return None, None

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
        st.error(f"Erro ao gerar explicabilidade SHAP: {e}")
        return None

def get_user_input_features():

    st.header("1. Dados do Aluno")

    col1, col2, col3 = st.columns(3)

    with col1:
        idade = st.number_input("Idade", min_value=6, max_value=30, value=12)

    with col2:
        genero = st.selectbox("Gênero", ["Menino", "Menina"])

    with col3:
        fase = st.selectbox("Fase Ideal", ["Alfa","Fase 1","Fase 2","Fase 3","Fase 4","Fase 5","Fase 6","Fase 7","Fase 8"])

    st.header("2. Notas Acadêmicas")

    mat = st.number_input("Matemática",0.0,10.0)
    por = st.number_input("Português",0.0,10.0)
    ing = st.number_input("Inglês",0.0,10.0)

    st.header("3. Indicadores")

    iaa = st.number_input("IAA",0.0,10.0)
    ieg = st.number_input("IEG",0.0,10.0)
    ips = st.number_input("IPS",0.0,10.0)
    ipp = st.number_input("IPP",0.0,10.0)

    inde_2022 = st.number_input("INDE 2022",0.0,10.0)
    inde_2023 = st.number_input("INDE 2023",0.0,10.0)
    inde_2024 = st.number_input("INDE Atual",0.0,10.0)

    ida = st.number_input("IDA",0.0,10.0)
    ipv = st.number_input("IPV",0.0,10.0)
    n_av = st.number_input("Número de Avaliações",0,50)

    data = {
        'idade': idade,
        'genero': genero,
        'fase_ideal': fase,
        'mat': mat,
        'por': por,
        'ing': ing,
        'iaa': iaa,
        'ieg': ieg,
        'ips': ips,
        'ipp': ipp,
        'inde_2022': inde_2022,
        'inde_2023': inde_2023,
        'inde_2024': inde_2024,
        'ida': ida,
        'ipv': ipv,
        'n_av': n_av
    }
    
    return pd.DataFrame(data, index=[0])

def main():

    model, config = load_models_and_config()
    
    st.title("🎓 Previsão de Defasagem Educacional")

    raw_input_df = get_user_input_features()

    if st.button("🔍 Realizar Predição"):

        processed_df = preparar_base_app(raw_input_df)

        processed_df = garantir_colunas_modelo(processed_df)

        probability = model.predict_proba(processed_df)

        proba_risco = probability[0][1]

        st.metric("Probabilidade de Risco", f"{proba_risco*100:.2f}%")

        fig = gerar_explicacao_shap(model, processed_df)

        if fig:
            st.pyplot(fig)

if __name__ == "__main__":
    main()
