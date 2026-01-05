import io
import unicodedata

# Importar biblioteca completa - terceiro
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st

st.set_page_config(page_title="Análise de Risco de Obesidade", layout="wide")

st.title('🍟 Análise de Risco de Obesidade')
st.info('Este aplicativo visa evidenciar as situações de risco analisadas de acordo com o banco de dados!')

def ordenar_opcoes(lista):
    """Ordena uma lista de strings ignorando acentos e maiúsculas"""
    def normalizar(texto):
        if isinstance(texto, str):
            return unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8').lower()
        return str(texto)
    return sorted(lista, key=normalizar)

def traduzir_nomes_features(lista_nomes_tecnicos):
    """Traduz os nomes técnicos do Pipeline para Português legível."""
    mapa_nomes = {
        'num__imc': 'Índice de Massa Corporal (IMC)',
        'num__idade': 'Idade',
        'bin__genero': 'Gênero',
        'bin__b_historico_familiar': 'Histórico Familiar',
        'bin__b_fuma': 'Hábito de Fumar',
        'bin__b_come_alimentos_caloricos': 'Consumo de Calóricos',
        'bin__b_monitora_calorias': 'Monitoramento de Calorias',
        'cat__freq_come_fora_refeicao_no': 'Comer entre refeições (Nunca)',
        'cat__freq_come_fora_refeicao_Sometimes': 'Comer entre refeições (Às vezes)',
        'cat__freq_come_fora_refeicao_Frequently': 'Comer entre refeições (Frequentemente)',
        'cat__freq_come_fora_refeicao_Always': 'Comer entre refeições (Sempre)',
        'cat__qtd_atv_fisicas_Sedentario': 'Sedentarismo',
        'cat__qtd_atv_fisicas_Baixa_frequencia': 'Baixa Atividade Física',
        'cat__qtd_atv_fisicas_Moderada_frequencia': 'Atividade Física Moderada',
        'cat__qtd_atv_fisicas_Alta_frequencia': 'Alta Atividade Física',
        'cat__qtd_agua_Baixo_consumo': 'Baixo consumo de água',
        'cat__qtd_agua_Consumo_adequado': 'Consumo de água (Adequado)',
        'cat__qtd_agua_Alto_consumo': 'Alto consumo de água',
        'cat__meio_de_transporte_Automobile': 'Uso de Carro',
        'cat__meio_de_transporte_Public_Transportation': 'Transporte Público',
        'cat__meio_de_transporte_Motorbike': 'Uso de Moto',
        'cat__meio_de_transporte_Bike': 'Uso de Bicicleta',
        'cat__meio_de_transporte_Walking': 'Caminhada',
        'cat__qtd_refeicao_Tres_refeicoes_principais_por_dia': '3 Refeições principais/dia',
        'cat__qtd_refeicao_Duas_refeicoes_principais_por_dia': '2 Refeições principais/dia',
        'cat__qtd_refeicao_Uma_refeicao_principal_por_dia': '1 Refeição principal/dia',
        'cat__qtd_refeicao_Quatro_ou_mais_refeicoes_principais_por_dia': '4+ Refeições principais/dia',
        'cat__qtd_vegetais_Sempre': 'Consumo de Vegetais (Sempre)',
        'cat__qtd_vegetais_As_vezes': 'Consumo de Vegetais (Às vezes)',
        'cat__qtd_vegetais_Raramente': 'Consumo de Vegetais (Raramente)',
        'cat__qtd_tmp_na_internet_Uso_baixo': 'Tempo em Telas (Baixo)',
        'cat__qtd_tmp_na_internet_Uso_moderado': 'Tempo em Telas (Moderado)',
        'cat__qtd_tmp_na_internet_Uso_intenso': 'Tempo em Telas (Intenso)',
        'cat__freq_alcool_no': 'Consumo de Álcool (Não)',
        'cat__freq_alcool_Sometimes': 'Consumo de Álcool (Às vezes)',
        'cat__freq_alcool_Frequently': 'Consumo de Álcool (Frequentemente)',
        'cat__freq_alcool_Always': 'Consumo de Álcool (Sempre)'
    }

    return [
        mapa_nomes.get(
            nome,
            nome.replace('num__', '').replace('cat__', '').replace('bin__', '').replace('_', ' ').title()
        )
        for nome in lista_nomes_tecnicos
    ]

@st.cache_resource
def load_model():
    try:
        return joblib.load('risco_obesidade_random_forest.joblib')
    except FileNotFoundError:
        url = "https://github.com/henriiqueww-pixel/dataviz-tcf4/raw/refs/heads/master/Modelos/risco_obesidade_random_forest.joblib"
        r = requests.get(url)
        return joblib.load(io.BytesIO(r.content))

@st.cache_resource
def _get_shap_explainer(_classifier):
    return shap.TreeExplainer(_classifier)

def configurar_sidebar():
    with st.sidebar:
        st.header("📌 Sobre o Projeto")
        st.info("""
        🎓 Pós-Graduação em Data Analytics  
        🏫 FIAP + Alura  
        """)

def gerar_explicacao_shap(model, input_df):
    pre = model.named_steps['preprocess']
    clf = model.named_steps['clf']

    X = pre.transform(input_df)
    feature_names = traduzir_nomes_features(pre.get_feature_names_out())

    explainer = _get_shap_explainer(clf)
    shap_values = explainer(X)
    shap_values.feature_names = feature_names

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0, :, 1], show=False, max_display=10)
    return fig

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
    st.info(f"ℹ️ IMC Calculado: {imc}")

    st.header("2. Histórico")
    historico = st.radio("Histórico familiar?", ["Sim", "Não"], horizontal=True)
    fuma = st.radio("Você fuma?", ["Sim", "Não"], horizontal=True)

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
        "Transporte Público": "Public_Transportation",
        "Caminhada": "Walking",
        "Carro": "Automobile",
        "Bicicleta": "Bike",
        "Moto": "Motorbike"
    }

    col3, col4 = st.columns(2)

    with col3:
        atv_key = st.selectbox("Atividade física", list(mapa_atv.keys()))
        net_key = st.selectbox("Tempo em telas", list(mapa_net.keys()))

    with col4:
        transporte_key = st.selectbox("Transporte", ordenar_opcoes(list(mapa_transporte.keys())))

    fora_key = st.selectbox("Come entre refeições?", ["Não", "Às vezes", "Frequentemente", "Sempre"])

    data = {
        'idade': idade,
        'genero': genero,
        'imc': imc,

        'qtd_atv_fisicas': mapa_atv[atv_key],
        'qtd_tmp_na_internet': mapa_net[net_key],
        'meio_de_transporte': mapa_transporte[transporte_key],

        'b_fuma': 1 if fuma == "Sim" else 0,
        'b_come_alimentos_caloricos': 1 if caloricos == "Sim" else 0,
        'b_monitora_calorias': 1 if monitora == "Sim" else 0,
        'b_historico_familiar': 1 if historico == "Sim" else 0,

        'freq_come_fora_refeicao': (
            'no' if fora_key == 'Não'
            else 'Sometimes' if fora_key == 'Às vezes'
            else 'Frequently' if fora_key == 'Frequentemente'
            else 'Always'
        ),

        'freq_alcool': 'no'
    }

    return pd.DataFrame(data, index=[0])

def main():
    configurar_sidebar()
    model = load_model()

    input_df = get_user_input_features()

    if st.button("🔍 Realizar Predição", type="primary", use_container_width=True):
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        if pred == 1:
            st.error(f"⚠️ ALTO RISCO ({prob*100:.1f}%)")
        else:
            st.success(f"✅ BAIXO RISCO ({prob*100:.1f}%)")

        with st.spinner("Gerando explicação..."):
            fig = gerar_explicacao_shap(model, input_df)
            st.pyplot(fig)

if __name__ == "__main__":
    main()


