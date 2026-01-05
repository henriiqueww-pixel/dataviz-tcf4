# =============================
# IMPORTS
# =============================
import io
import unicodedata

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st

# =============================
# CONFIG STREAMLIT
# =============================
st.set_page_config(
    page_title="Análise de Risco de Obesidade",
    layout="wide"
)

st.title("🍟 Análise de Risco de Obesidade")
st.info("Este aplicativo visa evidenciar as situações de risco analisadas de acordo com o banco de dados!")

# =============================
# FUNÇÕES AUXILIARES
# =============================
def ordenar_opcoes(lista):
    def normalizar(texto):
        return unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("utf-8").lower()
    return sorted(lista, key=normalizar)


def traduzir_nomes_features(lista_nomes_tecnicos):
    mapa_nomes = {
        'num__imc': 'Índice de Massa Corporal (IMC)',
        'num__idade': 'Idade',
        'bin__genero': 'Gênero',
        'bin__b_historico_familiar': 'Histórico Familiar',
        'bin__b_fuma': 'Hábito de Fumar',
        'bin__b_come_alimentos_caloricos': 'Consumo de Calóricos',
        'bin__b_monitora_calorias': 'Monitoramento de Calorias',
    }

    nomes = []
    for nome in lista_nomes_tecnicos:
        nomes.append(mapa_nomes.get(
            nome,
            nome.replace("num__", "").replace("cat__", "").replace("bin__", "").replace("_", " ").title()
        ))
    return nomes


@st.cache_resource
def load_model():
    try:
        return joblib.load("risco_obesidade_random_forest.joblib")
    except FileNotFoundError:
        url = "https://github.com/henriiqueww-pixel/dataviz-tcf4/raw/refs/heads/master/Modelos/risco_obesidade_random_forest.joblib"
        r = requests.get(url)
        return joblib.load(io.BytesIO(r.content))


@st.cache_resource
def get_explainer(model):
    return shap.TreeExplainer(model)


# =============================
# SIDEBAR
# =============================
def configurar_sidebar():
    with st.sidebar:
        st.header("📌 Sobre o Projeto")
        st.info("""
        🎓 Pós-Graduação em Data Analytics  
        🏫 FIAP + Alura  
        📊 Tech Challenge — Fase 4
        """)


# =============================
# INPUTS
# =============================
def get_user_input_features():
    st.header("1. Dados Pessoais")
    col1, col2 = st.columns(2)

    with col1:
        idade = st.number_input("Idade", 10, 100, 25)
        altura = st.number_input("Altura (m)", 1.0, 2.5, 1.70)

    with col2:
        genero_label = st.selectbox("Gênero", ["Masculino", "Feminino"])
        peso = st.number_input("Peso (kg)", 30.0, 200.0, 70.0)

    genero = 1 if genero_label == "Feminino" else 0
    imc = int(np.ceil(peso / (altura ** 2)))
    st.info(f"ℹ️ IMC calculado: **{imc}**")

    st.header("2. Histórico")
    historico = st.radio("Histórico familiar?", ["Sim", "Não"], horizontal=True)
    fuma = st.radio("Fuma?", ["Sim", "Não"], horizontal=True)

    st.header("3. Hábitos")
    caloricos = st.radio("Consome alimentos calóricos?", ["Sim", "Não"], horizontal=True)
    monitora = st.radio("Monitora calorias?", ["Sim", "Não"], horizontal=True)

    st.header("4. Estilo de Vida")

    mapa_atv = {
        "Sedentário": "Sedentario",
        "Baixa": "Baixa_frequencia",
        "Moderada": "Moderada_frequencia",
        "Alta": "Alta_frequencia"
    }

    mapa_net = {
        "Baixo (0-2h)": "Uso_baixo",
        "Moderado (3-5h)": "Uso_moderado",
        "Intenso (>5h)": "Uso_intenso"
    }

    mapa_transporte = {
        "Caminhada": "Walking",
        "Carro": "Automobile",
        "Bicicleta": "Bike",
        "Moto": "Motorbike",
        "Transporte Público": "Public_Transportation"
    }

    col3, col4 = st.columns(2)
    with col3:
        atv = st.selectbox("Atividade física", list(mapa_atv.keys()))
        net = st.selectbox("Tempo em telas", list(mapa_net.keys()))

    with col4:
        transporte = st.selectbox("Transporte", explicar := ordenar_opcoes(list(mapa_transporte.keys())))

    data = {
        "idade": idade,
        "genero": genero,
        "imc": imc,
        "qtd_atv_fisicas": mapa_atv[atv],
        "qtd_tmp_na_internet": mapa_net[net],
        "meio_de_transporte": mapa_transporte[transporte],
        "b_fuma": 1 if fuma == "Sim" else 0,
        "b_come_alimentos_caloricos": 1 if caloricos == "Sim" else 0,
        "b_monitora_calorias": 1 if monitora == "Sim" else 0,
        "b_historico_familiar": 1 if historico == "Sim" else 0,
        "freq_come_fora_refeicao": "Sometimes",
        "freq_alcool": "no",
        "qtd_refeicao": "Tres_refeicoes_principais_por_dia",
        "qtd_vegetais": "As_vezes",
        "qtd_agua": "Consumo_adequado"
    }

    return pd.DataFrame(data, index=[0])


# =============================
# SHAP
# =============================
def gerar_shap(model, input_df):
    pre = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]

    X = pre.transform(input_df)
    explainer = get_explainer(clf)
    shap_values = explainer(X)

    classe = clf.predict(X)[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0, :, classe], show=False)
    return fig


# =============================
# MAIN
# =============================
def main():
    configurar_sidebar()
    model = load_model()
    input_df = get_user_input_features()

    if st.button("🔍 Realizar Predição", use_container_width=True):
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        if pred == 1:
            st.error(f"⚠️ ALTO RISCO ({prob*100:.1f}%)")
        else:
            st.success(f"✅ BAIXO RISCO ({prob*100:.1f}%)")

        with st.spinner("Gerando explicação..."):
            fig = gerar_shap(model, input_df)
            st.pyplot(fig)


if __name__ == "__main__":
    main()

