import io
import unicodedata

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st

st.set_page_config(page_title="Análise de Risco de Obesidade", layout="wide")

st.title("🍟 Análise de Risco de Obesidade")
st.info("Este aplicativo visa evidenciar as situações de risco analisadas de acordo com o banco de dados!")

# =========================
# UTIL
# =========================
def ordenar_opcoes(lista):
    def normalizar(texto):
        return unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("utf-8").lower()
    return sorted(lista, key=normalizar)

# =========================
# MODEL
# =========================
@st.cache_resource
def load_model():
    try:
        return joblib.load("risco_obesidade_random_forest.joblib")
    except FileNotFoundError:
        url = "https://github.com/henriiqueww-pixel/dataviz-tcf4/raw/refs/heads/master/Modelos/risco_obesidade_random_forest.joblib"
        r = requests.get(url)
        return joblib.load(io.BytesIO(r.content))

@st.cache_resource
def get_explainer(clf):
    return shap.TreeExplainer(clf)

# =========================
# INPUTS
# =========================
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

    st.header("2. Histórico")
    historico = st.radio("Histórico familiar?", ["Sim", "Não"], horizontal=True)
    fuma = st.radio("Você fuma?", ["Sim", "Não"], horizontal=True)

    st.header("3. Hábitos Alimentares")
    refeicao_key = st.selectbox("Refeições/dia", ["1", "2", "3", "4+"], index=2)
    veg_key = st.selectbox("Vegetais", ["Raramente", "Às vezes", "Sempre"], index=1)
    agua_key = st.selectbox("Água", ["< 1 Litro", "1-2 Litros", "> 2 Litros"], index=1)
    fora_key = st.selectbox("Come entre refeições?", ["Não", "Às vezes", "Frequentemente", "Sempre"])

    st.header("4. Estilo de Vida")

    mapa_refeicao = {
        "1": "Uma_refeicao_principal_por_dia",
        "2": "Duas_refeicoes_principais_por_dia",
        "3": "Tres_refeicoes_principais_por_dia",
        "4+": "Quatro_ou_mais_refeicoes_principais_por_dia",
    }

    mapa_veg = {
        "Raramente": "Raramente",
        "Às vezes": "As_vezes",
        "Sempre": "Sempre",
    }

    mapa_agua = {
        "< 1 Litro": "Baixo_consumo",
        "1-2 Litros": "Consumo_adequado",
        "> 2 Litros": "Alto_consumo",
    }

    mapa_atv = {
        "Sedentário": "Sedentario",
        "Baixa": "Baixa_frequencia",
        "Moderada": "Moderada_frequencia",
        "Alta": "Alta_frequencia",
    }

    mapa_net = {
        "Baixo (0-2h)": "Uso_baixo",
        "Moderado (3-5h)": "Uso_moderado",
        "Intenso (>5h)": "Uso_intenso",
    }

    mapa_transporte = {
        "Transporte Público": "Public_Transportation",
        "Caminhada": "Walking",
        "Carro": "Automobile",
        "Bicicleta": "Bike",
        "Moto": "Motorbike",
    }

    col3, col4 = st.columns(2)
    with col3:
        atv_key = st.selectbox("Atividade física", list(mapa_atv.keys()))
        net_key = st.selectbox("Tempo em telas", list(mapa_net.keys()))

    with col4:
        transporte_key = st.selectbox("Transporte", ordenar_opcoes(list(mapa_transporte.keys())))

    data = {
        # Numéricas
        "idade": idade,
        "imc": imc,

        # Binárias
        "genero": genero,
        "b_fuma": 1 if fuma == "Sim" else 0,
        "b_historico_familiar": 1 if historico == "Sim" else 0,
        "b_come_alimentos_caloricos": 0,
        "b_monitora_calorias": 0,

        # Categóricas (TODAS AS ESPERADAS PELO MODELO)
        "qtd_refeicao": mapa_refeicao[refeicao_key],
        "qtd_vegetais": mapa_veg[veg_key],
        "qtd_agua": mapa_agua[agua_key],
        "qtd_atv_fisicas": mapa_atv[atv_key],
        "qtd_tmp_na_internet": mapa_net[net_key],
        "meio_de_transporte": mapa_transporte[transporte_key],
        "freq_come_fora_refeicao": (
            "no" if fora_key == "Não"
            else "Sometimes" if fora_key == "Às vezes"
            else "Frequently" if fora_key == "Frequentemente"
            else "Always"
        ),
        "freq_alcool": "no",
    }

    return pd.DataFrame(data, index=[0])

# =========================
# SHAP
# =========================
def gerar_shap(model, X):
    pre = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]

    Xt = pre.transform(X)
    explainer = get_explainer(clf)
    shap_values = explainer(Xt)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0, :, 1], show=False)
    return fig

# =========================
# MAIN
# =========================
def main():
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



